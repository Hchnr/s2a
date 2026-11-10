"""Bound first-step position error at the four prior development states."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.tracking.first_step_position_bound import first_step_position_bound
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_multitask_feasibility_first as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as previous
from scripts.tracking.trajectory_search import compare_multitask_feasibility_first as comparison
from scripts.verification.reference_ports.check_first_step_position_bound import check_corners

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/first_step_position_bound_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/first_step_position_bound_20261111.md"
QA = REPO / "outputs/qa/first_step_position_bound_20261111/result.json"
DRY = REPO / "outputs/first_step_position_bound_dry_20261111/result.json"
OUTPUT = REPO / "outputs/first_step_position_bound_20261111"
PRIOR = {
    "path": str(REPO / "outputs/multitask_feasibility_first_20261111/result.json"),
    "sha256": "4aa9e97a3384d6e88c7f6932a5d4aa7c4120a681a99728bcbb8dc4c831d70d09",
}
INDICES = [2, 6, 10, 11]


def source_refs():
    return comparison.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/joint_control/tracking/first_step_position_bound.py",
            REPO / "scripts/verification/reference_ports/check_first_step_position_bound.py",
            REPO / "tests/research/joint_control/tracking/test_first_step_position_bound.py",
            REPO / "tests/research/native_runtime/test_first_step_position_bound_check.py",
            REPO / "tests/research/native_runtime/test_first_step_position_bound_inputs.py",
        )
    ]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == dict(
            schema="s2a.first-step-position-bound.v1",
            case_indices=INDICES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            position_cap_m=0.01,
            arm_rate_limit=0.25,
            numerical_allowance_m=1e-10,
            directions=["positive_axes", "negative_axes", "seed_error", "previous_closest_error"],
            corners_per_case=64,
            max_models=4,
            max_static_forward_queries=4,
            max_seed_rollouts=4,
            max_corner_responses=256,
            max_corner_transition_steps=6400,
            max_workers=4,
            worker_timeout_seconds=600,
            optimizer_calls=0,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact first-step outer-bound scope and budget",
    )
    inputs = binding.load_inputs()
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed static solver comparison",
    )
    prior_cases = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(index not in prior_cases and item["complete"] is item["passed"] is True, "Unique prior result")
        prior_cases[index] = item["data"]
    prefix.require(sorted(prior_cases) == sorted(INDICES), "All four prior comparison states")
    for case in inputs["cases"]:
        index = case["comparison_spec"]["case_index"]
        saved = prior_cases[index]
        prefix.require(
            saved["case"] == case["comparison_spec"] and saved["seed_scope"] == case["seed_scope"],
            "Same state and seed scope",
        )
        closest = saved["comparison"]["solvers"]["feasibility_first"]["solver_result"][
            "closest_linear_feasible_attempt"
        ]
        prefix.require(closest is not None, "Prior closest linear-feasible attempt exists")
        case["previous_data"] = saved
    return dict(
        config_ref=config_ref, config=config, cases=inputs["cases"], asset_refs=inputs["asset_refs"], source_sha256=refs
    )


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        prior_ref=PRIOR,
        cases=[dict(**case["comparison_spec"], seed_scope=case["seed_scope"]) for case in inputs["cases"]],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )


def analyze(case, inputs, stats):
    previous.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static position bound forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    spec = case["comparison_spec"]
    offset = spec["completed_feedback_steps"]
    periods = case["feedback_periods"]
    if spec["kind"] == "stopped_quality":
        stop = case["stop_inputs"]
        periods = periods + [{"step": stop["stop_step"], "decision": {"refined_plan": stop["seed"]}}]
    current = previous.restored_context(model, data, case, periods, offset, stats)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    cursor = case["intervention_spec"]["cursor"] + offset
    targets = case["targets"][cursor : cursor + 8]
    twists = np.asarray(
        [
            reference_world_twist(case["targets"], index, case["anchor"], 0.05)["twist"]
            for index in range(cursor, cursor + 8)
        ]
    )
    prefix.require(targets.shape == (8, 4, 4) and twists.shape == (8, 6), "Previously generated reference window")
    args = (
        current["model"],
        current["velocity"],
        current["plan"],
        current["previous"][:6],
        current["interface"],
        current["site"],
        site_from_tcp,
        targets,
        twists,
        current["limits"][:, 0],
        current["limits"][:, 1],
    )
    prior = case["previous_data"]
    prior_profile = np.asarray(prior["past_inner_error_rad"])
    fingerprint = comparison.input_fingerprint(args, prior_profile)
    prefix.require(fingerprint == prior["comparison"]["input_sha256"], "Identical frozen state and solver inputs")
    candidate = prior["comparison"]["solvers"]["feasibility_first"]["solver_result"]["closest_linear_feasible_attempt"]
    previous_error = np.asarray(candidate["first_tcp_error"][:3])
    seed_position = (current["rollout"]["pose"][0] @ site_from_tcp)[:3, 3]
    directions = np.vstack((np.eye(3), -np.eye(3), targets[0, :3, 3] - seed_position, previous_error))
    scale = (current["interface"].output_max - current["interface"].output_min) / (
        current["interface"].input_max - current["interface"].input_min
    )
    seed_q = current["rollout"]["joint_position"][0]
    joint_jacobian = current["rollout"]["joint_position_jacobian"][0, :, :6] * scale
    bound = first_step_position_bound(
        current["site"],
        site_from_tcp,
        current["jacobian"],
        seed_q - current["q"],
        joint_jacobian,
        current["plan"][0],
        current["previous"][:6],
        current["interface"].input_min,
        current["interface"].input_max,
        targets[0, :3, 3],
        directions,
        rate_limit=inputs["config"]["arm_rate_limit"],
        numerical_allowance_m=inputs["config"]["numerical_allowance_m"],
    )
    np.testing.assert_allclose(bound["seed_tcp_position"], seed_position, atol=1e-10, rtol=0)
    stats["corner_responses_charged"] += 64
    stats["corner_transition_steps_charged"] += 1600
    audit = check_corners(
        current["model"],
        current["velocity"],
        current["interface"],
        current["site"],
        site_from_tcp,
        targets[0, :3, 3],
        current["plan"][0],
        seed_q,
        joint_jacobian,
        bound["command_lower"],
        bound["command_upper"],
        current["limits"][:, 0],
        current["limits"][:, 1],
        bound["lower_bound_m"],
    )
    stats["corner_responses_completed"] += audit["corner_responses"]
    stats["corner_transition_steps_completed"] += audit["transition_steps"]
    return dict(
        case=spec,
        seed_scope=case["seed_scope"],
        input_sha256=fingerprint,
        first_target_cursor=cursor,
        current_joint_margin_rad=prior["current_joint_margin_rad"],
        bound=bound,
        corner_check=audit,
        exceeds_original_position_cap=bool(bound["lower_bound_m"] > inputs["config"]["position_cap_m"]),
        scope="frozen-model relaxed first-command position bound; not a physical feasibility claim",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen bound inputs",
    )
    case = next(item for item in inputs["cases"] if item["comparison_spec"]["case_index"] == index)
    stats = dict(
        model_initializations_attempted=0,
        model_initializations_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        corner_responses_charged=0,
        corner_responses_completed=0,
        corner_transition_steps_charged=0,
        corner_transition_steps_completed=0,
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        case=case["comparison_spec"],
        stats=stats,
        automatic_retry=False,
    )
    try:
        data = analyze(case, inputs, stats)
        prefix.require(
            stats["model_initializations_completed"]
            == stats["forward_queries_completed"]
            == stats["baseline_rollouts"]
            == 1
            and stats["corner_responses_charged"] == stats["corner_responses_completed"] == 64
            and stats["corner_transition_steps_charged"] == stats["corner_transition_steps_completed"] == 1600,
            "Per-case static and independent corner budgets",
        )
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Closed CPU QA",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry identity",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(OUTPUT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=sources))
    specs = [
        dict(
            key=f"case_{index:02d}",
            argv=[
                sys.executable,
                "-m",
                "scripts.tracking.trajectory_search.audit_first_step_position_bound",
                "--worker",
                str(index),
            ],
            log=OUTPUT / f"case_{index:02d}" / "worker.log",
        )
        for index in INDICES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(OUTPUT / "workers.json", lifecycle)
    available = [
        prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json")
        for index in INDICES
        if (OUTPUT / f"case_{index:02d}" / "result.json").is_file()
    ]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration,
        workers_ref=workers,
        available_case_refs=available,
        optimizer_calls=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 4
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "Four workers exited and were reaped",
        )
        case_refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index in INDICES]
        cases = [prefix.read(ref) for ref in case_refs]
        prefix.require(all(item["complete"] is item["passed"] is True for item in cases), "All corner audits closed")
        keys = (
            "model_initializations_completed",
            "forward_queries_completed",
            "baseline_rollouts",
            "corner_responses_charged",
            "corner_responses_completed",
            "corner_transition_steps_charged",
            "corner_transition_steps_completed",
        )
        counts = {key: sum(item["stats"][key] for item in cases) for key in keys}
        prefix.require(
            counts["model_initializations_completed"]
            == counts["forward_queries_completed"]
            == counts["baseline_rollouts"]
            == 4
            and counts["corner_responses_charged"] == counts["corner_responses_completed"] == 256
            and counts["corner_transition_steps_charged"] == counts["corner_transition_steps_completed"] == 6400,
            "Registered full numerical costs",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Bound source identity unchanged"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=case_refs, counts=counts)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=INDICES)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        inputs = load_inputs()
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=input_receipt(inputs),
                    source_refs=source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.run:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
