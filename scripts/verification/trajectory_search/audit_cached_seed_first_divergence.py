"""Audit cached-seed selection at each original cohort's first action divergence."""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, pose_error
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.references.generated_twist import reference_world_twist
from s2a.references.oracle_timeline import OracleTimeline
from s2a.research.joint_control.preview.cached_seed_preference import solve_prefer_cached_seed
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from s2a.research.trajectory_search.margin_two_step_search import margin_prefix_satisfied
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as original
from scripts.simulation.runtime import run_reset_correct_reference_cohort as cohort

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/cached_seed_first_divergence_20261116"
QA = ROOT / "outputs/qa/cached_seed_first_divergence_20261116/result.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/cached_seed_preference_20261116.md"
COHORT_AUDIT = {
    "path": str(ROOT / "outputs/reset_correct_reference_cohort_20261116/audit.json"),
    "sha256": "dbe7d296758cc7cb9c9f66168ed140b13ed2967dbb84e5879044c3e849ec704a",
}
COHORT_DRY = {
    "path": str(ROOT / "outputs/reset_correct_reference_cohort_dry_20261116.json"),
    "sha256": "b1fa5eadf94aa475386d0fc1762da32b2df0152d4da8c5e75ee44b5099874cd5",
}
MODULE = "scripts.verification.trajectory_search.audit_cached_seed_first_divergence"


def first_divergence(baseline, candidate, *, tolerance=1e-10):
    """Require unchanged full actions before the first actual arm difference."""
    a, b = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1:] != (7,)
        or b.shape[1:] != (7,)
        or len(b) == 0
        or len(a) < len(b)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Finite recorded seven-dimensional action prefixes are required")
    difference = np.abs(a[: len(b)] - b)
    changed = np.flatnonzero(np.max(difference, axis=1) > tolerance)
    if not len(changed):
        raise ValueError("An actual action divergence is required")
    step = int(changed[0])
    if np.max(difference[step, :6]) <= tolerance or difference[step, 6] > tolerance:
        raise ValueError("The first difference must be in the arm with matching gripper commands")
    return step


def source_refs():
    return cohort.source_refs() + [
        prefix.reference(path)
        for path in (
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/preview/cached_seed_preference.py",
            ROOT / "tests/research/joint_control/preview/test_cached_seed_preference.py",
            ROOT / "tests/research/trajectory_search/test_cached_seed_first_divergence.py",
        )
    ]


def audit_case(identity, entry):
    refs, spec = identity["refs"], identity["spec"]
    rows, chunks, controller, tool_data, baseline = (
        prefix.read(refs[key]) for key in ("rows", "chunks", "controller", "tool", "result")
    )
    branch = prefix.read(entry["producer_ref"])
    periods = original.periods(branch)
    step = first_divergence([row["command"] for row in rows], [row["command"] for row in periods])
    with np.load(prefix.checked(refs["arrays"]), allow_pickle=False) as archive:
        saved_state = np.asarray(archive["sim_states"][step]).copy()
        segment = next(
            row
            for row in branch["runtime"]["continuation"]["segments"]
            if row["start_step"] <= step < row["start_step"] + row["continuation"]["steps_completed"]
        )
        start = segment["start_step"]
        source_anchor = np.asarray(archive["source_pose"][start]).copy()
    if step:
        current = np.asarray(periods[step - 1]["actual_state"], dtype=float)
        state_difference = float(np.max(np.abs(current - saved_state)))
        previous = np.asarray(periods[step - 1]["command"][:6], dtype=float)
    else:
        prefix.require(
            state_sha256(saved_state) == branch["runtime"]["reset_endpoint_comparison"]["actual_state_sha256"],
            "The saved and actual initial state hashes must agree",
        )
        state_difference = 0.0
        previous = np.zeros(6)
    prefix.require(state_difference <= 1e-10, "The first divergent decision must begin from the same saved state")
    baseline_row, candidate_row = rows[step], periods[step]
    dynamics = baseline_row["dynamics"]
    mass, jacobian, kp, kd, q, velocity = (
        np.asarray(dynamics[name], dtype=float)
        for name in ("mass", "jacobian", "kp", "kd", "robot_joint_position", "robot_joint_velocity")
    )
    initial_joint = np.asarray(branch["runtime"]["endpoint_controller"]["initial_joint"], dtype=float)
    interface = OSCInterface(
        *(controller["contract"][name] for name in ("input_min", "input_max", "output_min", "output_max"))
    )
    model = build_joint_preview(mass, jacobian, kp, kd, q, initial_joint, null_kp=10.0, dt=0.002)
    xml = ET.parse(prefix.checked(baseline["model_ref"]))
    prefix.require(xml.getroot().find("compiler").get("angle") == "radian", "Original radian joint limits required")
    limits = np.asarray(
        [
            [float(value) for value in xml.find(f'.//joint[@name="robot0_joint{k}"]').attrib["range"].split()]
            for k in range(1, 8)
        ],
        dtype=float,
    )
    lower, upper = limits[:, 0], limits[:, 1]
    tool = np.asarray(tool_data["source_from_tcp"], dtype=float)
    site_from_tcp = inverse(controller["source_from_site"]) @ tool
    site = np.asarray(baseline_row["site_pose"], dtype=float)
    timeline = OracleTimeline(chunks, periods=spec["periods"])
    poses, _openings, _metadata = timeline.window(start)
    offset = step - start
    anchor = source_anchor @ tool
    targets = poses[offset : offset + 8].copy()
    twists = np.asarray(
        [reference_world_twist(poses, index, anchor, 0.05)["twist"] for index in range(offset, offset + 8)]
    )
    seed = np.asarray(baseline_row["refinement"]["planned_commands"], dtype=float)
    selected = np.asarray(candidate_row["decision"]["refined_plan"], dtype=float)
    past = np.asarray(candidate_row["decision"]["quality"]["past_error"]["inner_position_error_rad"], dtype=float)

    def rollout(commands):
        return rollout_joint_preview(
            model,
            velocity,
            np.asarray([interface.decode(command) for command in commands]),
            site,
            lower,
            upper,
            enforce_limits=False,
        )

    def errors(value):
        return np.asarray([pose_error(targets[k], value["pose"][k] @ site_from_tcp) for k in range(8)])

    seed_rollout, selected_rollout = rollout(seed), rollout(selected)
    seed_errors, selected_errors = errors(seed_rollout), errors(selected_rollout)
    recorded_errors = np.asarray(candidate_row["decision"]["quality"]["verification"]["tcp_errors"], dtype=float)
    reconstruction_difference = float(np.max(np.abs(selected_errors - recorded_errors)))
    prefix.require(reconstruction_difference <= 1e-9, "The frozen model must reconstruct the selected candidate")
    domain = linear_domain(model, seed, previous, interface, lower, upper, past, seed_rollout)
    domain_result = domain_check(seed, domain, interface)
    seed_margin = margin_prefix_satisfied(seed_errors)
    future_slack = max(
        0.0,
        float(
            np.max(
                np.r_[
                    np.linalg.norm(seed_errors[2:, :3], axis=1) / 0.01 - 1,
                    np.linalg.norm(seed_errors[2:, 3:], axis=1) / 0.05 - 1,
                ]
            )
        ),
    )
    seed_verified = None
    if domain_result["accepted"] and seed_margin:
        witness = dict(
            witness_found=True,
            planned_commands=seed,
            rollout=seed_rollout,
            terminal_normals=domain["terminal_normals"],
            first_tcp_error=seed_errors[0],
            tcp_errors=seed_errors,
            future_quality_slack=future_slack,
            hard_pose_prefix=2,
            prefix_selection_hard=True,
        )
        seed_verified = audit_two_step_witness(
            witness,
            mass,
            jacobian,
            kp,
            kd,
            q,
            velocity,
            initial_joint,
            interface,
            previous,
            site,
            site_from_tcp,
            targets,
            lower,
            upper,
            past_inner_error_rad=past,
        )
        prefix.require(seed_verified["passed"] is True, "The historical plan needs the original independent audit")
    args = (model, velocity, seed, previous, interface, site, site_from_tcp, targets, twists, lower, upper)
    events_default, events_preferred = [], []
    default = shared.solve_margin_guarded_fallback(*args, past_inner_error_rad=past, on_response=events_default.append)
    preferred = solve_prefer_cached_seed(*args, past_inner_error_rad=past, on_response=events_preferred.append)
    prefix.require(
        default["status"] == preferred["status"] == "margin_guarded_candidate"
        and default["shared_search"]["selected_scope"]
        == preferred["shared_search"]["selected_scope"]
        == "margin_two_step"
        and default["response_evaluations"] == preferred["response_evaluations"] <= 320
        and events_default == events_preferred
        and events_default.count("attempted") == events_default.count("completed") == default["response_evaluations"],
        "Both cached rankings must keep the same complete bounded search",
    )
    default_plan, preferred_plan = (
        default["best_joint_safe"]["planned_commands"],
        preferred["best_joint_safe"]["planned_commands"],
    )
    default_difference = float(np.max(np.abs(default_plan - selected)))
    prefix.require(default_difference <= 1e-9, "The default ranking must reproduce the recorded candidate plan")
    checked = audit_two_step_witness(
        preferred["best_joint_safe"],
        mass,
        jacobian,
        kp,
        kd,
        q,
        velocity,
        initial_joint,
        interface,
        previous,
        site,
        site_from_tcp,
        targets,
        lower,
        upper,
        past_inner_error_rad=past,
    )
    prefix.require(checked["passed"] is True and margin_prefix_satisfied(checked["tcp_errors"]), "Preferred plan audit")
    preference = preferred["seed_preference"]
    if preference["selected_seed"]:
        np.testing.assert_array_equal(preferred_plan, seed)
    else:
        np.testing.assert_array_equal(preferred_plan, default_plan)
    return dict(
        case=spec,
        qualified_before=entry["qualified"],
        first_divergence_step=step,
        prestate_max_difference=state_difference,
        selected_model_error_max_difference=reconstruction_difference,
        baseline_seed_domain=domain_result,
        baseline_seed_margin_satisfied=seed_margin,
        baseline_seed_independent_witness=bool(seed_verified and seed_verified["passed"]),
        baseline_seed_first_two_position_errors_m=np.linalg.norm(seed_errors[:2, :3], axis=1),
        baseline_seed_first_two_rotation_errors_rad=np.linalg.norm(seed_errors[:2, 3:], axis=1),
        default_recorded_plan_max_difference=default_difference,
        responses_per_search=default["response_evaluations"],
        preference=preference,
        preferred_independent_witness=checked["passed"],
        historical_seed_is_near_state_proxy=True,
    )


def worker(index):
    directory = OUTPUT / f"case{index:02d}"
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    result = dict(complete=False, passed=False, case_index=index, physics_steps=0, world_model_calls=0, fresh_cases=0)
    try:
        prefix.require(registration["source_refs"] == source_refs(), "Frozen analysis sources required")
        result["metrics"] = audit_case(registration["cases"][index], registration["entries"][index])
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    if args.worker is not None:
        prefix.require(0 <= args.worker < 18, "One registered case required")
        worker(args.worker)
        return
    qa = prefix.read(prefix.reference(QA))
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and before == prefix.read(qa["source_before_ref"])["sources"],
        "Current passing CPU QA required",
    )
    audit, dry = prefix.read(COHORT_AUDIT), prefix.read(COHORT_DRY)
    producer = prefix.read(audit["producer_ref"])
    registration = prefix.read(producer["registration_ref"])
    prefix.require(
        audit["complete"] is audit["passed"] is dry["complete"] is dry["passed"] is True
        and audit["registered_cases"] == len(audit["cases"]) == len(dry["inputs"]["cases"]) == 18
        and registration["inputs"] == dry["inputs"]
        and registration["source_refs"] == cohort.source_refs(),
        "The exact completed original cohort and policy sources are required",
    )
    sources = source_refs()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        OUTPUT / "registration.json",
        dict(
            cases=dry["inputs"]["cases"],
            entries=audit["cases"],
            source_refs=sources,
            cohort_audit_ref=COHORT_AUDIT,
            cohort_dry_ref=COHORT_DRY,
            qa_ref=prefix.reference(QA),
        ),
    )
    specs = [
        dict(
            key=str(index),
            argv=[sys.executable, "-m", MODULE, "--worker", str(index)],
            log=OUTPUT / f"case{index:02d}/worker.log",
        )
        for index in range(18)
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=600)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(OUTPUT / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 18
            and all(
                row["returncode"] == 0 and row["actual_wait"] is row["reaped"] is True and not row["signal_sent"]
                for row in lifecycle["workers"]
            ),
            "All static workers must close and be reaped",
        )
        case_refs = [prefix.reference(OUTPUT / f"case{index:02d}/result.json") for index in range(18)]
        cases = [prefix.read(ref) for ref in case_refs]
        prefix.require(
            all(row["complete"] is row["passed"] is True for row in cases), "Every fixed state must be audited"
        )
        prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Static audit sources changed")
        result.update(
            complete=True,
            passed=True,
            source_unchanged=True,
            case_refs=case_refs,
            seed_eligible=sum(row["metrics"]["preference"]["seed_in_eligible_cache"] for row in cases),
            selection_changed=sum(row["metrics"]["preference"]["selected_changed"] for row in cases),
            total_responses_per_ranking=sum(row["metrics"]["responses_per_search"] for row in cases),
        )
    finally:
        print(publish_result(OUTPUT / "result.json", result), flush=True)


if __name__ == "__main__":
    main()
