"""Bind three-task solver comparisons to independently replayed causal states."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_task2_warning_feedback as task2
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_quality_stop_feasibility as stopped

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/multitask_feasibility_first_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/multitask_feasibility_first_20261111.md"
TASK2_PRODUCER = {
    "path": str(REPO / "outputs/task2_warning_feedback_20261111/producer/result.json"),
    "sha256": "b3530db3cd9babc7f0c8431f59c14cee4809ca60217d17f92741b6361ee5210c",
}
TASK2_CHECKER = {
    "path": str(REPO / "outputs/task2_warning_feedback_20261111/checker/result.json"),
    "sha256": "d86848a947998f07b6763f6a41c1da7ce416a8d6074b3bd6075a3a442943689a",
}
CASES = [
    dict(case_index=2, case_id="task0_init0_s43_raw", kind="stopped_quality", step=48, completed_feedback_steps=5),
    dict(case_index=6, case_id="task1_init0_s43_raw", kind="stopped_quality", step=57, completed_feedback_steps=5),
    dict(case_index=10, case_id="task2_init0_s43_raw", kind="warning_start", step=34, completed_feedback_steps=0),
    dict(
        case_index=11, case_id="task2_init0_s43_terminal058", kind="warning_start", step=82, completed_feedback_steps=0
    ),
]


def source_refs():
    return (
        stopped.source_refs()
        + task2.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                CONFIG,
                PLAN,
                REPO / "src/s2a/research/trajectory_search/feasibility_first_joint_plan.py",
                REPO / "tests/research/trajectory_search/test_feasibility_first_joint_plan.py",
                REPO / "tests/research/native_runtime/test_multitask_feasibility_first_inputs.py",
            )
        ]
    )


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.multitask-feasibility-first.v1",
            "cases": CASES,
            "seed": 43,
            "solvers": ["original", "feasibility_first"],
            "plan_horizon": 8,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "braking_time_s": 0.1,
            "max_response_evaluations_per_solver": 160,
            "max_optimizer_calls": 8,
            "max_optimizer_responses": 1280,
            "max_models": 4,
            "max_static_forward_queries": 14,
            "max_independent_endpoint_queries": 1600,
            "max_independent_transition_steps": 20800,
            "max_workers": 4,
            "worker_timeout_seconds": 1800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact three-task cases, original constraints and finite solver budgets",
    )
    return ref, config


def merge_refs(target, source):
    for path, digest in source.items():
        prefix.require(path not in target or target[path] == digest, "Historical input identity conflict")
        target[path] = digest


def index_branches(owner, refs):
    branches = {}
    for ref in owner["branch_refs"]:
        item = prefix.read(ref, refs)
        key = (item["case"]["case_index"], item["branch"])
        prefix.require(key not in branches and item["complete"] is item["passed"] is True, "Unique closed task2 branch")
        branches[key] = item
    return branches


def load_inputs():
    config_ref, config = read_config()
    first, third = stopped.load_inputs(), task2.load_inputs()
    prefix.require(
        first["asset_refs"] == third["asset_refs"] and first["cv2_admission"] == third["cv2_admission"],
        "All three tasks share the same independently migrated runtime assets",
    )
    refs = {config_ref["path"]: config_ref["sha256"]}
    merge_refs(refs, first["source_sha256"])
    merge_refs(refs, third["source_sha256"])
    producer, checker = (prefix.read(ref, refs) for ref in (TASK2_PRODUCER, TASK2_CHECKER))
    prefix.require(
        producer["complete"]
        is producer["passed"]
        is producer["sources_unchanged"]
        is checker["complete"]
        is checker["passed"]
        is checker["sources_unchanged"]
        is True,
        "Both task2 physical feedback and independent replay are closed",
    )
    produced, verified = index_branches(producer, refs), index_branches(checker, refs)
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        if spec["kind"] == "stopped_quality":
            case = next(item for item in first["cases"] if item["spec"]["case_index"] == index)
            stop = case["stop_inputs"]
            prefix.require(
                case["spec"]["case_id"] == spec["case_id"]
                and stop["stop_step"] == spec["step"]
                and len(case["feedback_periods"]) == spec["completed_feedback_steps"]
                and case["spec"]["prefix_steps"] + len(case["feedback_periods"]) == spec["step"],
                "Actual five-period quality stop and causal history",
            )
            seed_scope = "shifted_last_completed_plan_not_original_stop_seed"
        else:
            case = dict(next(item for item in third["cases"] if item["spec"]["case_index"] == index))
            left, right = produced[(index, "original")], verified[(index, "original")]
            p, q = left["runtime"]["continuation"]["periods"], right["runtime"]["continuation"]["periods"]
            prefix.require(
                case["spec"]["case_id"] == spec["case_id"]
                and case["spec"]["prefix_steps"] == spec["step"]
                and len(p) >= 1
                and len(p) == len(q)
                and p[0]["step"] == q[0]["step"] == spec["step"],
                "Replayed task2 original refiner at the shared warning state",
            )
            np.testing.assert_array_equal(p[0]["command"], q[0]["command"])
            np.testing.assert_array_equal(p[0]["actual_state"], q[0]["actual_state"])
            case["feedback_periods"] = [p[0]]
            case["initial_joint"] = np.asarray(left["runtime"]["runtime_initial_joint"], dtype=float)
            prefix.require(
                case["initial_joint"].shape == (7,) and np.isfinite(case["initial_joint"]).all(),
                "Actual initialized joint anchor",
            )
            seed_scope = "matched_original_refiner_plan_at_warning"
        prefix.require(case["config"]["seed"] == config["seed"], "Original sampling seed")
        case["comparison_spec"] = spec
        case["seed_scope"] = seed_scope
        cases.append(case)
    return dict(config_ref=config_ref, config=config, cases=cases, asset_refs=first["asset_refs"], source_sha256=refs)


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        cases=[dict(**case["comparison_spec"], seed_scope=case["seed_scope"]) for case in inputs["cases"]],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
