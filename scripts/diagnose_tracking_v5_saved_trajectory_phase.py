"""Compare saved TCP endpoints with a fixed reference phase grid; never simulate."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.geometry import validate_transform

HORIZON = 12
GRID_DIVISIONS = 1000
POSITION_CAP_M = 0.01
ROTATION_CAP_RAD = 0.05
PERIOD_SECONDS = 0.05
BOUND_DIAGNOSTIC_TOLERANCE = 1e-10
ROTATION_PROJECTION_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_tcp(value):
    """Preserve raw transforms and admit only a roundoff-sized SO(3) projection."""
    raw = validate_transform(value)
    rotations = Rotation.from_matrix(raw[..., :3, :3])
    canonical = raw.copy()
    canonical[..., :3, :3] = rotations.as_matrix()
    difference = float(np.max(np.abs(canonical[..., :3, :3] - raw[..., :3, :3])))
    defects = dict(
        orthogonality_max_abs=float(np.max(np.abs(raw[..., :3, :3].swapaxes(-1, -2) @ raw[..., :3, :3] - np.eye(3)))),
        determinant_max_abs_error=float(np.max(np.abs(np.linalg.det(raw[..., :3, :3]) - 1.0))),
        projection_max_abs_difference=difference,
    )
    require(difference <= ROTATION_PROJECTION_TOLERANCE, "SO(3) projection exceeds 1e-12 gate")
    return canonical, rotations, defects


def canonical_phase_grid(anchor, targets, rotations):
    require(anchor.shape == (4, 4) and targets.shape == (HORIZON, 4, 4), "One anchor and twelve TCP targets")
    knots = np.concatenate((anchor[None], targets))
    ticks = np.arange(HORIZON * GRID_DIVISIONS + 1)
    phases = ticks / GRID_DIVISIONS - 1.0
    segments = np.minimum(ticks // GRID_DIVISIONS, HORIZON - 1)
    fractions = (ticks - segments * GRID_DIVISIONS) / GRID_DIVISIONS
    translations = np.diff(knots[:, :3, 3], axis=0)
    rotation_vectors = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
    path_rotations = Rotation.from_rotvec(fractions[:, None] * rotation_vectors[segments]) * rotations[segments]
    path_rotations[::GRID_DIVISIONS] = rotations
    poses = np.broadcast_to(np.eye(4), (len(phases), 4, 4)).copy()
    poses[:, :3, 3] = knots[segments, :3, 3] + fractions[:, None] * translations[segments]
    poses[:, :3, :3] = path_rotations.as_matrix()
    # Keep every canonical integer knot exactly; original raw knots stay separate.
    poses[::GRID_DIVISIONS] = knots
    lipschitz = np.maximum(
        np.linalg.norm(translations, axis=1) / POSITION_CAP_M,
        np.linalg.norm(rotation_vectors, axis=1) / ROTATION_CAP_RAD,
    )
    return phases, poses, lipschitz, path_rotations


def interpolate_phase_grid(anchor_tcp, targets_tcp):
    """World-linear translation and shortest canonical SO(3) arcs on 12,001 phases."""
    anchor, anchor_rotation, _ = canonical_tcp(anchor_tcp)
    targets, target_rotations, _ = canonical_tcp(targets_tcp)
    phases, poses, lipschitz, _ = canonical_phase_grid(
        anchor, targets, Rotation.concatenate([anchor_rotation, target_rotations])
    )
    return phases, poses, lipschitz


def diagnose_saved_tcp(predicted_tcp, anchor_tcp, targets_tcp):
    """Independent per-endpoint geometric optima, with no phase/order repair."""
    raw_predicted = validate_transform(predicted_tcp)
    raw_targets = validate_transform(targets_tcp)
    predicted, predicted_rotations, predicted_defects = canonical_tcp(raw_predicted)
    anchor, anchor_rotation, anchor_defects = canonical_tcp(anchor_tcp)
    targets, target_rotations, target_defects = canonical_tcp(raw_targets)
    require(predicted.shape == (HORIZON, 4, 4), "Twelve saved predicted TCP endpoints")
    phases, path, lipschitz, path_rotations = canonical_phase_grid(
        anchor, targets, Rotation.concatenate([anchor_rotation, target_rotations])
    )
    raw_translation_error = raw_targets[:, :3, 3] - raw_predicted[:, :3, 3]
    raw_rotation_error = Rotation.from_matrix(
        raw_targets[:, :3, :3] @ raw_predicted[:, :3, :3].swapaxes(-1, -2)
    ).as_rotvec()
    raw_positions = np.linalg.norm(raw_translation_error, axis=1)
    raw_angles = np.linalg.norm(raw_rotation_error, axis=1)
    raw_scores = np.maximum(
        np.sum(raw_translation_error**2, axis=1) / POSITION_CAP_M**2,
        np.sum(raw_rotation_error**2, axis=1) / ROTATION_CAP_RAD**2,
    )
    records = []
    max_timed_difference = 0.0
    for cursor, pose in enumerate(predicted):
        positions = np.linalg.norm(path[:, :3, 3] - pose[:3, 3], axis=1)
        angles = (path_rotations * predicted_rotations[cursor].inv()).magnitude()
        scores = np.maximum((positions / POSITION_CAP_M) ** 2, (angles / ROTATION_CAP_RAD) ** 2)
        best = int(np.argmin(scores))  # Ascending full grid: the earliest exact tie wins.
        timed = (cursor + 1) * GRID_DIVISIONS
        timed_difference = max(abs(positions[timed] - raw_positions[cursor]), abs(angles[timed] - raw_angles[cursor]))
        require(timed_difference <= 1e-10, "Canonical timed errors differ from raw by more than 1e-10")
        max_timed_difference = max(max_timed_difference, float(timed_difference))
        segment_grid_min = np.asarray(
            [np.sqrt(scores[k * GRID_DIVISIONS : (k + 1) * GRID_DIVISIONS + 1]).min() for k in range(HORIZON)]
        )
        # f=max(distance/.01, angle/.05) is L-Lipschitz per unit phase.
        # Every continuous phase lies within half a grid step of a sample.
        segment_lower = np.maximum(0.0, segment_grid_min - lipschitz / (2 * GRID_DIVISIONS)) ** 2
        lower = float(segment_lower.min())
        records.append(
            dict(
                timed_phase=cursor,
                best_grid_index=best,
                best_phase=float(phases[best]),
                lag_periods=float(cursor - phases[best]),
                lag_seconds=float((cursor - phases[best]) * PERIOD_SECONDS),
                timed_position_error_m=float(raw_positions[cursor]),
                timed_rotation_error_rad=float(raw_angles[cursor]),
                timed_score=float(raw_scores[cursor]),
                timed_passed=bool(raw_scores[cursor] <= 1.0),
                canonical_timed_position_error_m=float(positions[timed]),
                canonical_timed_rotation_error_rad=float(angles[timed]),
                canonical_timed_score=float(scores[timed]),
                canonical_timed_passed=bool(scores[timed] <= 1.0),
                best_position_error_m=float(positions[best]),
                best_rotation_error_rad=float(angles[best]),
                best_score=float(scores[best]),
                best_grid_passed=bool(scores[best] <= 1.0),
                segment_grid_min_score=segment_grid_min**2,
                segment_continuous_score_lower_bound=segment_lower,
                continuous_score_lower_bound=lower,
                continuous_path_caps_excluded=bool(lower > 1.0 + BOUND_DIAGNOSTIC_TOLERANCE),
            )
        )
    best_phases = np.asarray([item["best_phase"] for item in records])
    decreases = np.flatnonzero(np.diff(best_phases) < 0)
    return dict(
        endpoints=records,
        best_phases_monotone=not bool(len(decreases)),
        phase_decreases=[
            dict(
                earlier_cursor=int(index),
                later_cursor=int(index + 1),
                earlier_phase=float(best_phases[index]),
                later_phase=float(best_phases[index + 1]),
            )
            for index in decreases
        ],
        timed_pass_count=sum(item["timed_passed"] for item in records),
        best_grid_pass_count=sum(item["best_grid_passed"] for item in records),
        continuous_path_caps_excluded_count=sum(item["continuous_path_caps_excluded"] for item in records),
        segment_lipschitz_per_phase=lipschitz,
        geometry_defects=dict(anchor=anchor_defects, targets=target_defects, predicted=predicted_defects),
        canonical_current_tcp_anchor=anchor,
        canonical_desired_tcp=targets,
        canonical_predicted_tcp=predicted,
        canonical_timed_error_max_difference=max_timed_difference,
        timed_metrics="Original raw transforms and original cap labels; canonical timed metrics are separate",
        continuous_bound_scope="Canonical SO(3) copies of the saved predicted endpoints and reference path only",
        endpoint_phase_pairs=HORIZON * len(phases),
    )


def file_reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_result(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)


def run(prefix_result, original_result, chunks, prefix_check_owner_root, output_root):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "Explicit empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH required",
    )
    output = Path(output_root).resolve()
    require(output.is_relative_to(REPO / "outputs") and not output.exists(), "New S2A outputs directory required")
    input_refs = {}

    def read(path, expected=None):
        ref = file_reference(path)
        require(expected is None or ref == expected, "Input reference SHA changed")
        require(input_refs.get(ref["path"], ref) == ref, "Previously read input changed")
        input_refs[ref["path"]] = ref
        payload = Path(ref["path"]).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == ref["sha256"], "Input changed during read")
        return json.loads(payload)

    def audit_status():
        root = Path(prefix_check_owner_root).resolve()
        registration = read(root / "registration.json")
        registration_ref = input_refs[str(root / "registration.json")]
        require(registration["stage"] == "check", "Explicit prefix independent checker owner")
        launch = read(root / "capture/launch.json")
        launch_ref = input_refs[str(root / "capture/launch.json")]
        checker_script = REPO / "scripts/check_tracking_v5_prefix_preserving_trajectory_control.py"
        require(
            launch["command"] == [str(REPO / "outputs/libero_env311/bin/python"), str(checker_script)]
            and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
            and launch["automatic_retry"] is False
            and registration["source_sha256"][str(checker_script)] == file_reference(checker_script)["sha256"],
            "Same CPU independent prefix checker",
        )
        require(root.name.count("_execution_") == 1, "Unambiguous checker owner/output directory relationship")
        checker_root = root.with_name(root.name.replace("_execution_", "_", 1))
        checker_reg = read(checker_root / "registration.json")
        checker_reg_ref = input_refs[str(checker_root / "registration.json")]
        require(checker_reg["producer_ref"] == file_reference(prefix_result), "Checker binds this exact prefix input")
        binding = dict(
            producer_ref=checker_reg["producer_ref"],
            owner_registration_ref=registration_ref,
            launch_ref=launch_ref,
            checker_registration_ref=checker_reg_ref,
        )
        if (root / "result.json").exists():
            owner = read(root / "result.json")
            require(owner["registration_ref"] == registration_ref, "Terminal owner binds observed registration")
            record = dict(status="terminal", complete=owner["complete"], passed=owner["passed"], **binding)
            if owner["complete"] is owner["passed"] is True:
                require(owner["owned_command"]["launch"] == launch_ref, "Terminal owner binds observed launch")
                exited = read(root / "capture/exit.json", owner["owned_command"]["exit"])
                released = read(root / "capture/ownership/release.json", owner["owned_command"]["release"])
                require(
                    exited["actual_wait"] is True
                    and exited["returncode"] == 0
                    and released["reaped"] is True
                    and released["returncode"] == 0
                    and released["signal_sent"] is False
                    and released["original_error"] is None
                    and released["pid"] == launch["child"]["pid"]
                    and released["start_ticks"] == launch["child"]["start_ticks"],
                    "Bound checker child actually waited/reaped0 without signal",
                )
                ref = owner["result_ref"]
                checked_result = read(ref["path"], ref)
                require(
                    checked_result["registration_ref"] == checker_reg_ref
                    and checked_result["complete"] is checked_result["passed"] is True
                    and checked_result["trace_count"] == prefix["answer"]["response_evaluations"],
                    "Terminal verified result binds this checker registration and full prefix trace",
                )
                record.update(exit=exited, release=released)
                record["verified_ref"] = ref
                record["verified_complete"] = checked_result["complete"]
                record["verified_passed"] = checked_result["passed"]
            return record
        proc = Path("/proc") / str(launch["child"]["pid"])
        try:
            stat = (proc / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            require(
                int(fields[19]) == launch["child"]["start_ticks"] and int(fields[1]) == launch["child"]["ppid"],
                "Checker child birth and parent identities match launch",
            )
            if fields[0] == "Z":
                return dict(status="awaiting_owner_result", complete=False, passed=False, **binding)
            command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            require(command == launch["command"], "Checker live command matches bound launch")
        except FileNotFoundError:
            return dict(status="awaiting_owner_result", complete=False, passed=False, **binding)
        return dict(
            status="running", complete=False, passed=False, child=launch["child"], process_state=fields[0], **binding
        )

    prefix, original, chunk_records = read(prefix_result), read(original_result), read(chunks)
    require(
        prefix["complete"] is prefix["passed"] is original["complete"] is original["passed"] is True,
        "Completed saved producer artifacts required",
    )
    prefix_reg = read(prefix["registration_ref"]["path"], prefix["registration_ref"])
    original_reg = read(original["registration_ref"]["path"], original["registration_ref"])
    require(
        prefix_reg["prior_trajectory_ref"] == file_reference(original_result)
        and prefix_reg["chunks_ref"] == original_reg["chunks_ref"] == file_reference(chunks)
        and prefix_reg["endpoint"] == original_reg["endpoint"] == 368
        and prefix_reg["horizon"] == original_reg["horizon"] == HORIZON,
        "Same original H12 producer, actual368 and generated chunk source",
    )
    prior_check_ref = prefix_reg["prior_trajectory_check_ref"]
    prior_check = read(prior_check_ref["path"], prior_check_ref)
    prior_check_reg = read(prior_check["registration_ref"]["path"], prior_check["registration_ref"])
    require(
        prior_check["complete"] is prior_check["passed"] is True
        and prior_check["trace_count"] == 4000
        and prior_check_reg["producer_ref"] == file_reference(original_result),
        "Original H12 has its own complete independent numerical audit",
    )
    for field in (
        "current_q",
        "current_velocity",
        "desired_tcp",
        "site_from_tcp",
        "warm",
        "initial_joint",
        "kp",
        "kd",
        "joint_limits",
        "torque_limits",
        "latched_rotation_goal",
    ):
        require(np.array_equal(prefix[field], original[field]), f"Same saved input: {field}")
    require(
        np.array_equal(prefix["answer"]["initial"]["commands"], original["answer"]["initial"]["commands"]),
        "Exactly the same initial warm commands",
    )
    require(
        np.array_equal(prefix["current_quantities"]["pose"], original["current_quantities"]["pose"]),
        "Same actual368 current SITE pose",
    )
    chunk = chunk_records[46]
    require(chunk["step"] == 368, "Same already generated chunk46")
    desired = validate_transform(np.asarray(chunk["effective_targets_tcp"][:HORIZON]))
    require(np.array_equal(desired, prefix["desired_tcp"]), "Exact first twelve original TCP targets")
    tool = validate_transform(prefix["site_from_tcp"])
    anchor = validate_transform(np.asarray(prefix["current_quantities"]["pose"]) @ tool)
    candidates = {
        "prefix_selected": prefix["answer"]["selected"],
        "original_h12_selected": original["answer"]["selected"],
        "same_warm": prefix["answer"]["initial"],
    }
    original_warm = original["answer"]["initial"]
    require(
        np.array_equal(candidates["same_warm"]["rollout"]["pose"], original_warm["rollout"]["pose"]),
        "Same saved warm SITE trajectory",
    )
    audit_before = audit_status()
    source_paths = [
        Path(__file__),
        REPO / "src/s2a/artifacts.py",
        REPO / "src/s2a/geometry.py",
        REPO / "tests/test_saved_trajectory_phase_diagnostic.py",
        REPO / "plans/v0.0.1/saved_trajectory_phase_diagnostic_20261028.md",
    ]
    source_before = [file_reference(path) for path in source_paths]
    inputs_before = list(input_refs.values())
    output.mkdir(parents=True)
    registration = dict(
        input_refs=inputs_before,
        source_refs=source_before,
        prefix_check_owner_root=str(Path(prefix_check_owner_root).resolve()),
        grid=dict(start=-1.0, stop=11.0, step=0.001, points=12001, integer_knots_included=True),
        interpolation="World-linear translation and shortest principal SO(3) arcs between explicitly canonicalized TCP knots",
        rotation_projection_max_abs=ROTATION_PROJECTION_TOLERANCE,
        timed_metric_policy="Preserve original raw timed errors/cap labels; canonical timed errors must differ by at most1e-10",
        score="max((position_error_m/0.01)^2,(principal_rotation_error_rad/0.05)^2)",
        tie_policy="Earliest phase among exactly equal computed minimum grid scores",
        continuous_bound="Canonical path only: for each segment max(0,min_grid_sqrt_score-L*0.0005)^2; minimum over segments",
        continuous_bound_diagnostic_tolerance=BOUND_DIAGNOSTIC_TOLERANCE,
        monotonicity_policy="Report independent best-phase decreases; never repair or constrain phases",
        qualification="Posthoc geometry only; best phase does not certify executable retiming or new control",
        cpu_environment={key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        seed=42,
        random_sampling=False,
        automatic_retry=False,
    )
    write_result(output / "registration.json", registration)
    result = dict(
        complete=False,
        passed=False,
        registration_ref=file_reference(output / "registration.json"),
        control_qualified=False,
        retiming_qualified=False,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        new_commands=0,
        new_forward_responses=0,
        original_independent_check=dict(
            verified_ref=prior_check_ref,
            complete=prior_check["complete"],
            passed=prior_check["passed"],
            trace_count=prior_check["trace_count"],
            control_qualified=prior_check["control_qualified"],
        ),
        prefix_independent_check_before=audit_before,
        current_tcp_anchor=anchor,
        desired_tcp=desired,
        candidates={},
    )
    try:
        for name, candidate in candidates.items():
            tcp = validate_transform(np.asarray(candidate["rollout"]["pose"]) @ tool)
            diagnostic = diagnose_saved_tcp(tcp, anchor, desired)
            saved_errors = np.asarray(candidate["tcp_errors"])
            timed = np.asarray(
                [[item["timed_position_error_m"], item["timed_rotation_error_rad"]] for item in diagnostic["endpoints"]]
            )
            saved_norms = np.stack(
                (np.linalg.norm(saved_errors[:, :3], axis=1), np.linalg.norm(saved_errors[:, 3:], axis=1)), axis=1
            )
            require(np.max(np.abs(timed - saved_norms)) <= 1e-10, "Original timed errors remain unchanged")
            saved_passed = np.all(np.asarray(candidate["normalized_squared_errors"]) <= 1.0, axis=1)
            require(
                np.array_equal([item["timed_passed"] for item in diagnostic["endpoints"]], saved_passed),
                "Original raw timed cap labels remain unchanged",
            )
            diagnostic["saved_timed_error_max_difference"] = float(np.max(np.abs(timed - saved_norms)))
            diagnostic["saved_timed_passed"] = saved_passed
            diagnostic["saved_predicted_tcp"] = tcp
            result["candidates"][name] = diagnostic
        result["prefix_independent_check_after"] = audit_status()
        source_after = [file_reference(path) for path in source_paths]
        require(source_after == source_before, "Diagnostic source changed during execution")
        for ref in input_refs.values():
            require(file_reference(ref["path"]) == ref, "Saved input changed during diagnosis")
        result.update(
            complete=True,
            passed=True,
            input_refs_before=inputs_before,
            input_refs_after=list(input_refs.values()),
            source_refs_before=source_before,
            source_refs_after=source_after,
            source_unchanged=True,
            inputs_unchanged=True,
            endpoint_phase_pairs=3 * HORIZON * 12001,
            interpretation="Grid minimum bounds the canonical continuous optimum from above; only the conservative lower bound excludes canonical continuous caps. Raw timed labels are preserved. Nonmonotone independent optima do not exclude another monotone assignment. Neither bound is executable retiming evidence.",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        write_result(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prefix-result", "original-result", "chunks", "prefix-check-owner-root", "output-root"):
        parser.add_argument("--" + option, type=Path, required=True)
    arguments = parser.parse_args()
    result = run(**vars(arguments))
    print(json.dumps(dict(complete=result["complete"], passed=result["passed"], control_qualified=False)), flush=True)
