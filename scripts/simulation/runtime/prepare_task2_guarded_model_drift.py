"""Bind completed guarded-feedback periods to their independent replay."""

import copy
from pathlib import Path

import numpy as np

from s2a.research.joint_control.tracking.plan_model_drift import shifted_completed_plan
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_guarded_two_step_feedback as previous
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as context

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_guarded_model_drift_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_guarded_model_drift_20261111.md"
PRODUCER = {
    "path": str(REPO / "outputs/task2_guarded_two_step_feedback_20261111/producer/result.json"),
    "sha256": "f83d77c74a4230419cb41eb624d8c8fdd07b8f27ca8646e8838121dbaad3a4b7",
}
CHECKER = {
    "path": str(REPO / "outputs/task2_guarded_two_step_feedback_20261111/checker/result.json"),
    "sha256": "2e4ce651b8d7a0d62cf88a1f0097555bbe21c8d65f1b615b42cc919c89718f5c",
}
CASES = [
    dict(case_index=10, start_step=32, completed_periods=1),
    dict(case_index=11, start_step=80, completed_periods=4),
]


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/joint_control/tracking/plan_model_drift.py",
            REPO / "tests/research/joint_control/tracking/test_plan_model_drift.py",
            REPO / "tests/research/native_runtime/test_task2_guarded_model_drift_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == dict(
            schema="s2a.task2-guarded-model-drift.v1",
            cases=CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            braking_time_s=0.1,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            max_models=2,
            max_static_forward_queries=10,
            max_eight_step_rollouts=10,
            max_counterfactual_model_builds=10,
            max_counterfactual_first_rollouts=10,
            max_independent_first_responses=10,
            max_independent_transition_steps=250,
            max_workers=2,
            worker_timeout_seconds=1200,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact registered completed-period model-drift scope and budgets",
    )
    return ref, value


def completed_periods(produced, checked, spec):
    """Admit only commands and endpoints exactly reproduced by the checker."""
    p, q = produced["runtime"]["continuation"], checked["runtime"]["continuation"]
    n, start = spec["completed_periods"], spec["start_step"]
    prefix.require(
        produced["complete"] is produced["passed"] is checked["complete"] is checked["passed"] is True
        and produced["control_qualified"] is checked["control_qualified"] is False
        and produced["phase"] == "producer"
        and checked["phase"] == "checker"
        and produced["case"] == checked["case"]
        and produced["case"]["case_index"] == spec["case_index"]
        and produced["case"]["prefix_steps"] == start
        and produced["branch"] == checked["branch"] == "guarded_two_step"
        and checked["candidate_modules_unloaded"] is True
        and p["stop"] == q["stop"]
        and p["stop"]["reason"] == "quality_no_witness"
        and p["stop"]["at_step"] == start + n
        and p["stop"]["next_action_dispatched"] is False
        and q["decision_stop_recorded_not_recomputed"] is True
        and len(p["periods"]) == len(q["periods"]) == n,
        "Exact closed producer and independent replay for the registered branch",
    )
    np.testing.assert_array_equal(
        produced["runtime"]["runtime_initial_joint"], checked["runtime"]["runtime_initial_joint"]
    )
    for offset, (left, right) in enumerate(zip(p["periods"], q["periods"], strict=True)):
        prefix.require(left["step"] == right["step"] == start + offset, "Consecutive completed periods")
        for key in ("command", "actual_state", "tcp_error"):
            np.testing.assert_array_equal(left[key], right[key])
        prefix.require(
            left["actual_inner"] == right["actual_inner"]
            and len(left["actual_inner"]) == 25
            and all(left[key] == right[key] for key in ("pose_caps_satisfied", "task_success", "target_grasp")),
            "All inner samples and outcome labels must match the independent replay",
        )
        shifted_completed_plan(left)
        past = left["completed_feedback"]["past_error_after_completion"]
        profile = np.asarray(past["inner_position_error_rad"], dtype=float)
        prefix.require(
            past["available"] is True
            and past["future_guarantee"] is False
            and past["completed_periods"] == offset + 1
            and past["last_completed_step"] == start + offset
            and profile.shape == (25, 7)
            and np.isfinite(profile).all()
            and np.all(profile >= 0)
            and np.all(np.diff(profile, axis=0) >= 0),
            "Past error must come only from completed local periods",
        )
    return p["periods"]


def load_inputs():
    config_ref, config = read_config()
    old = previous.load_inputs()
    previous_receipt = copy.deepcopy(previous.input_receipt(old))
    refs = dict(old["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    producer, checker = (prefix.read(ref, refs) for ref in (PRODUCER, CHECKER))
    branches = {}
    for phase, owner in (("producer", producer), ("checker", checker)):
        prefix.require(
            owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
            and owner["control_qualified"] is False
            and owner["phase"] == phase
            and owner["world_model_calls"] == owner["fresh_cases"] == 0,
            "Closed earlier feedback and independent replay without new WAM or fresh cases",
        )
        registration = prefix.read(owner["registration_ref"], refs)
        prefix.require(
            registration["inputs"] == previous_receipt and registration["source_refs"] == previous.source_refs(),
            "Feedback inputs and frozen source identity",
        )
        for ref in owner["branch_refs"]:
            item = prefix.read(ref, refs)
            key = (phase, item["case"]["case_index"])
            prefix.require(key not in branches, "Each feedback branch is unique")
            branches[key] = item
    prefix.require(
        set(branches) == {(phase, spec["case_index"]) for phase in ("producer", "checker") for spec in CASES},
        "Registered feedback branches",
    )
    prefix.checked(context.OSMESA, refs)
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        produced, checked = branches[("producer", index)], branches[("checker", index)]
        case = dict(next(item for item in old["cases"] if item["spec"]["case_index"] == index))
        prefix.require(
            case["spec"]["prefix_steps"] == spec["start_step"]
            and case["config"]["osmesa_library"] == context.OSMESA["path"],
            "Registered causal start and software GL identity",
        )
        case.update(
            drift_spec=spec,
            feedback_periods=completed_periods(produced, checked, spec),
            initial_joint=np.asarray(produced["runtime"]["runtime_initial_joint"], dtype=float),
        )
        cases.append(case)
    return dict(
        config_ref=config_ref,
        config=config,
        cases=cases,
        asset_refs=old["asset_refs"],
        previous_receipt=previous_receipt,
        source_sha256=refs,
    )


def input_receipt(inputs):
    return dict(
        previous=inputs["previous_receipt"],
        config_ref=inputs["config_ref"],
        producer_ref=PRODUCER,
        checker_ref=CHECKER,
        cases=CASES,
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
