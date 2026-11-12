"""Reconstruct bowl/TCP geometry from two audited saved-state trajectories."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, validate_transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.saved_geometry import reconstruct_endpoint
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as previous
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_saved_geometry_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_saved_geometry_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_saved_geometry_20261112"
QA = ROOT / "outputs/qa/reset_bowl_saved_geometry_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_saved_geometry_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.diagnose_reset_bowl_saved_geometry"
SOURCE = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
FIXED = dict(
    schema="s2a.reset-bowl-saved-geometry.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    site_name="gripper0_grip_site",
    body_name="akita_black_bowl_1_main",
    coordinate_frame="MuJoCo world",
    position_unit="m",
    anchor_absolute_tolerance_m=1e-9,
    max_models=1,
    max_data=1,
    max_forwards=97,
    physics_steps=0,
    contexts=0,
    model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=900,
)


def validate_config(config):
    prefix.require(
        previous.previous.previous._exact(config, FIXED), "The exact saved-geometry scope and budgets are required"
    )


def source_refs():
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/native_runtime/saved_geometry.py",
        ROOT / "scripts/simulation/runtime/audit_reset_bowl_fixed_h40.py",
        ROOT / "tests/research/native_runtime/test_saved_geometry.py",
        ROOT / "tests/research/native_runtime/test_diagnose_reset_bowl_saved_geometry.py",
    )
    return previous.source_refs() + [prefix.reference(path) for path in names]


def branch_periods(branch, kind):
    continuation = branch["runtime"]["continuation"]
    first = continuation["first_window"]["periods"]
    key = "fixed" if kind == "fixed" else "stream"
    live = [row for segment in continuation[key]["segments"] for row in segment["continuation"]["periods"]]
    return first + live


def load_inputs(*, verify_assets=True):
    prefix.environment()
    config_ref, plan_ref = prefix.reference(CONFIG), prefix.reference(PLAN)
    config, source = prefix.read(config_ref), prefix.read(SOURCE)
    validate_config(config)
    inputs = previous.load_inputs(verify_assets=verify_assets)
    prefix.require(
        source["complete"] is source["passed"] is True
        and source["control_qualified"] is False
        and source["new_producer_periods"] == source["new_checker_periods"] == 48
        and source["audit_model_requests"] == source["audit_physics_steps"] == 0,
        "The independently replayed fixed-H40 probe is required",
    )
    prior = prefix.read(source["prior_audit_ref"])
    refs = dict(
        config=config_ref,
        plan=plan_ref,
        source=SOURCE,
        fixed=source["producer_branch_ref"],
        fixed_checker=source["checker_branch_ref"],
        reobserved=source["old_reobservation_branch_ref"],
        reobserved_checker=prior["cases"][0]["checker_ref"],
        prior_audit=source["prior_audit_ref"],
    )
    fixed, fixed_checker, old, old_checker = (
        prefix.read(refs[key]) for key in ("fixed", "fixed_checker", "reobserved", "reobserved_checker")
    )
    prefix.require(
        all(branch["complete"] is branch["passed"] is True for branch in (fixed, fixed_checker, old, old_checker))
        and all(branch["case"]["case_id"] == config["case_id"] for branch in (fixed, fixed_checker, old, old_checker))
        and fixed_checker["candidate_modules_unloaded"] is old_checker["candidate_modules_unloaded"] is True,
        "The same completed development case and isolated checkers are required",
    )
    fixed_periods, old_periods = branch_periods(fixed, "fixed"), branch_periods(old, "reobserved")
    fixed_replayed = fixed_checker["runtime"]["continuation"]
    old_replayed = old_checker["runtime"]["continuation"]
    fixed_checked = fixed_replayed["first_window"]["periods"] + [
        row for segment in fixed_replayed["replays"] for row in segment["periods"]
    ]
    old_checked = old_replayed["first_window"]["periods"] + [
        row for segment in old_replayed["replays"] for row in segment["periods"]
    ]
    prefix.require(
        len(fixed_periods) == len(old_periods) == len(fixed_checked) == len(old_checked) == 48
        and [row["step"] for row in fixed_periods] == [row["step"] for row in old_periods] == list(range(48)),
        "Exactly two audited 48-period saved trajectories are required",
    )
    comparison.compare_periods(fixed_periods, fixed_checked)
    comparison.compare_periods(old_periods, old_checked)
    comparison.compare_periods(fixed_periods[:16], old_periods[:16])
    case = inputs["case"]
    initial = np.asarray(case["expected"]["sim_states"], dtype=np.float64)
    for branch, periods in ((fixed, fixed_periods), (old, old_periods)):
        prefix.require(
            branch["runtime"]["reset_render"]["current_endpoint"]["sim_state_sha256"] == state_sha256(initial),
            "Each trajectory must start at the same registered reset state",
        )
        for period in periods:
            prefix.require(
                state_sha256(np.asarray(period["actual_state"], dtype=np.float64))
                == period["endpoint_receipt"]["sim_state_sha256"],
                "Each saved physical state must match its endpoint digest",
            )
    xml = case["record_xml"]
    xml_sha = hashlib.sha256(xml.encode()).hexdigest()
    prefix.require(
        all(
            next(row["xml_sha256"] for row in branch["runtime"]["model_records"] if row["stage"] == "record_xml")
            == xml_sha
            for branch in (fixed, old)
        ),
        "The two trajectories must use the same asset-verified MuJoCo XML",
    )
    source_hashes = dict(inputs["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, source_hashes)
    return dict(
        config=config,
        refs=refs,
        base=inputs,
        case=case,
        initial=initial,
        fixed=fixed,
        old=old,
        fixed_periods=fixed_periods,
        old_periods=old_periods,
        xml_sha256=xml_sha,
        source_sha256=source_hashes,
    )


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        previous=previous.input_receipt(inputs["base"]),
        case=inputs["case"]["case"],
        xml_sha256=inputs["xml_sha256"],
        source_sha256=inputs["source_sha256"],
    )


def dry_run():
    inputs = load_inputs()
    prefix.require("mujoco" not in sys.modules, "Dry admission cannot import or construct MuJoCo")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                models=0,
                data=0,
                forwards=0,
                physics_steps=0,
                contexts=0,
                model_requests=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A complete 32-worker source-matched regression is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot differs")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["models"] == dry["data"] == dry["forwards"] == dry["physics_steps"] == dry["model_requests"] == 0,
        "The same sources require a zero-model, zero-forward dry",
    )
    return before


def summarize_points(points, reset_body):
    """Describe distances to a body origin, not contact or grasp feasibility."""
    prefix.require(
        len(points) == 48 and [row["endpoint_step"] for row in points] == list(range(1, 49)),
        "Exactly 48 ordered endpoints are required",
    )
    distance = np.asarray([row["body_origin_distance_m"] for row in points], dtype=float)
    horizontal = np.asarray([row["body_origin_xy_distance_m"] for row in points], dtype=float)
    height = np.asarray([row["tcp_above_body_origin_m"] for row in points], dtype=float)
    objects = np.asarray([row["body_origin_world_m"] for row in points], dtype=float)
    motion = np.linalg.norm(objects - np.asarray(reset_body, dtype=float), axis=1)
    prefix.require(
        all(np.isfinite(value).all() for value in (distance, horizontal, height, objects, motion)),
        "Finite reconstructed geometry required",
    )
    closest = int(np.argmin(distance))
    closest_xy = int(np.argmin(horizontal))
    return dict(
        first_distance_m=float(distance[0]),
        last_distance_m=float(distance[-1]),
        minimum_distance_m=float(distance[closest]),
        minimum_distance_endpoint_step=closest + 1,
        first_xy_distance_m=float(horizontal[0]),
        last_xy_distance_m=float(horizontal[-1]),
        minimum_xy_distance_m=float(horizontal[closest_xy]),
        minimum_xy_distance_endpoint_step=closest_xy + 1,
        minimum_tcp_above_body_origin_m=float(height.min()),
        maximum_tcp_above_body_origin_m=float(height.max()),
        maximum_body_origin_displacement_m=float(motion.max()),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in points),
        task_success_periods=sum(bool(row["task_success"]) for row in points),
    )


def run_geometry(inputs, counters, *, mujoco):
    """Build one model and query only the 97 previously recorded states."""
    config, case = inputs["config"], inputs["case"]
    prefix.require("mujoco" in sys.modules, "The worker must explicitly admit the CPU MuJoCo module")
    counters["model_attempts"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    counters["models"] += 1
    counters["data_attempts"] += 1
    data = mujoco.MjData(model)
    counters["data"] += 1
    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config["site_name"]))
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["body_name"]))
    prefix.require(
        site_id >= 0 and body_id >= 0 and model.na == 0, "The registered TCP site and target body must exist"
    )
    expected_site = validate_transform(np.asarray(case["expected"]["controller_site_pose"], dtype=float))
    expected_source = validate_transform(np.asarray(case["expected"]["source_pose"], dtype=float))
    tool = validate_transform(np.asarray(case["tool"], dtype=float))
    site_from_tcp = validate_transform(inverse(expected_site) @ expected_source @ tool)
    expected_tcp = (expected_source @ tool)[:3, 3]
    original_forward = mujoco.mj_forward

    def counted_forward(actual_model, actual_data):
        prefix.require(
            counters["forwards_attempted"] < config["max_forwards"], "The geometry-query budget is exhausted"
        )
        counters["forwards_attempted"] += 1
        original_forward(actual_model, actual_data)
        counters["forwards_completed"] += 1

    def query(state):
        return reconstruct_endpoint(
            model,
            data,
            state,
            site_id=site_id,
            body_id=body_id,
            site_from_tcp=site_from_tcp,
            forward=counted_forward,
        )

    originals = {}
    for name in ("mj_step", "mj_step1", "mj_step2"):
        if hasattr(mujoco, name):
            originals[name] = getattr(mujoco, name)

            def reject(*_args, **_kwargs):
                raise ValueError("Saved-state geometry cannot advance physics")

            setattr(mujoco, name, reject)
    try:
        reset = query(inputs["initial"])
        np.testing.assert_allclose(
            reset["tcp_world_m"], expected_tcp, atol=config["anchor_absolute_tolerance_m"], rtol=0
        )
        points = {}
        for key, periods in (("fixed", inputs["fixed_periods"]), ("reobserved", inputs["old_periods"])):
            rows = []
            for period in periods:
                point = query(period["actual_state"])
                point.update(
                    endpoint_step=period["step"] + 1,
                    state_sha256=period["endpoint_receipt"]["sim_state_sha256"],
                    measured_opening=period["completed_feedback"]["score"]["opening"],
                    target_grasp=period["target_grasp"],
                    task_success=period["task_success"],
                )
                rows.append(point)
            points[key] = rows
        checked = []
        for key, branch, captures in (
            ("fixed", inputs["fixed"], [inputs["fixed"]["runtime"]["fixed_reference_capture"]]),
            ("reobserved", inputs["old"], inputs["old"]["runtime"]["streaming_captures"]),
        ):
            captures = [branch["runtime"]["reset_render"], *captures]
            for capture in captures:
                step = capture["frame_index"]
                actual = reset if step == 0 else points[key][step - 1]
                observed = (np.asarray(capture["current_source_pose"], dtype=float) @ tool)[:3, 3]
                difference = float(np.max(np.abs(actual["tcp_world_m"] - observed)))
                prefix.require(
                    difference <= config["anchor_absolute_tolerance_m"],
                    "Reconstructed TCP differs from its actual captured endpoint",
                )
                checked.append(dict(branch=key, endpoint_step=step, max_absolute_difference_m=difference))
        prefix.require(
            counters["models"] == counters["data"] == 1
            and counters["forwards_attempted"] == counters["forwards_completed"] == 97
            and counters["physics_steps"] == counters["contexts"] == counters["model_requests"] == 0,
            "The actual geometry work must match the fixed query budget",
        )
        summaries = {key: summarize_points(rows, reset["body_origin_world_m"]) for key, rows in points.items()}
        return dict(
            model_shape=dict(nq=model.nq, nv=model.nv, na=model.na),
            site_id=site_id,
            body_id=body_id,
            site_from_tcp=site_from_tcp,
            reset=reset,
            capture_checks=checked,
            points=points,
            summaries=summaries,
        )
    finally:
        for name, original in originals.items():
            setattr(mujoco, name, original)


def worker():
    directory = OUTPUT / "worker"
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        counters=dict(
            model_attempts=0,
            models=0,
            data_attempts=0,
            data=0,
            forwards_attempted=0,
            forwards_completed=0,
            physics_steps=0,
            contexts=0,
            model_requests=0,
        ),
        automatic_retry=False,
        fresh_cases=0,
    )
    try:
        prefix.environment()
        registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
        inputs = load_inputs(verify_assets=False)
        prefix.require(
            registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
            "The geometry worker must use exactly the registered saved inputs",
        )
        prefix.require("mujoco" not in sys.modules, "MuJoCo cannot load before source and budget admission")
        import mujoco

        result["geometry"] = run_geometry(inputs, result["counters"], mujoco=mujoco)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.environment()
    inputs = load_inputs()
    before = check_qa_and_dry(inputs)
    sources = source_refs()
    prefix.require(not OUTPUT.exists(), "The geometry output must be fresh")
    OUTPUT.mkdir()
    directory = OUTPUT / "worker"
    directory.mkdir()
    registration = publish(OUTPUT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=sources))
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration,
        automatic_retry=False,
        fresh_cases=0,
    )
    try:
        environment = physical._cpu_environment(directory)
        spec = dict(key="geometry", argv=[sys.executable, "-m", MODULE, "--worker"], log=OUTPUT / "worker.log")
        with physical._temporary_environment(environment):
            lifecycle = run_owned_workers([spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        registered_feedback_phase.require_closed_workers(lifecycle, ["geometry"])
        worker_ref = prefix.reference(directory / "result.json")
        completed = prefix.read(worker_ref)
        counters = completed["counters"]
        prefix.require(
            completed["complete"] is completed["passed"] is True
            and counters["models"] == counters["data"] == 1
            and counters["forwards_attempted"] == counters["forwards_completed"] == 97
            and counters["physics_steps"] == counters["contexts"] == counters["model_requests"] == 0,
            "The closed worker must report its exact one-model, 97-query budget",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "The saved geometry sources changed"
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            worker_ref=worker_ref,
            counters=counters,
            summaries=completed["geometry"]["summaries"],
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        path = directory / "result.json"
        if path.is_file():
            result["available_worker_ref"] = prefix.reference(path)
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker()


if __name__ == "__main__":
    main()
