"""Bind only the completed history at independently replayed shared-search stops."""

import copy
from pathlib import Path

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.plan_model_drift import shifted_completed_plan
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_margin_guarded_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_shared_shift_qualification_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_shared_shift_qualification_20261111.md"
PRODUCER = {
    "path": str(REPO / "outputs/task2_margin_guarded_feedback_20261111/producer/result.json"),
    "sha256": "dd064000117f33e8d5342327d1b0751fb0f713708249e2f31bd845ea5669f687",
}
CHECKER = {
    "path": str(REPO / "outputs/task2_margin_guarded_feedback_20261111/checker/result.json"),
    "sha256": "6d264b2c938fc396ef624a3e27073a6cb496a50653732f64234e099cd830ff1e",
}
CASES = [dict(case_index=10, stop_step=33), dict(case_index=11, stop_step=84)]


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "tests/research/native_runtime/test_task2_shared_shift_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-shared-shift-qualification.v1",
            cases=CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            plan_horizon=8,
            joint_buffer_rad=0.05,
            braking_time_s=0.1,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            max_models=2,
            max_static_forward_queries=2,
            max_eight_step_rollouts=2,
            max_independent_audits=2,
            max_independent_endpoint_queries=400,
            max_independent_transition_steps=5200,
            max_workers=2,
            worker_timeout_seconds=1200,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact shared-search stops and shifted-plan audit budgets",
    )
    return ref, config


def completed_stop(produced, checked, spec):
    """Require the actual branch-local history and independent replay."""
    p, q = produced["runtime"]["continuation"], checked["runtime"]["continuation"]
    start = produced["case"]["prefix_steps"]
    n = spec["stop_step"] - start
    prefix.require(
        produced["complete"] is produced["passed"] is checked["complete"] is checked["passed"] is True
        and produced["control_qualified"] is checked["control_qualified"] is False
        and produced["phase"] == "producer"
        and checked["phase"] == "checker"
        and produced["case"] == checked["case"]
        and produced["case"]["case_index"] == spec["case_index"]
        and produced["branch"] == checked["branch"] == "margin_guarded_fallback"
        and checked["candidate_modules_unloaded"] is True
        and p["stop"] == q["stop"]
        and p["stop"]["reason"] == "quality_no_witness"
        and p["stop"]["solver_status"] == "margin_guarded_no_witness"
        and p["stop"]["at_step"] == spec["stop_step"]
        and p["stop"]["next_action_dispatched"] is False
        and q["decision_stop_recorded_not_recomputed"] is True
        and len(p["periods"]) == len(q["periods"]) == n
        and n > 0,
        "Exact replayed feedback, completed history and recorded stop",
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
            "All completed inner samples and outcome labels must match replay",
        )
        shifted_completed_plan(left)
    last = p["periods"][-1]
    past = last["completed_feedback"]["past_error_after_completion"]
    profile = finite(past["inner_position_error_rad"], (25, 7)).copy()
    prefix.require(
        past["available"] is True
        and past["future_guarantee"] is False
        and past["completed_periods"] == n
        and past["last_completed_step"] == spec["stop_step"] - 1
        and np.all(profile >= 0)
        and np.all(np.diff(profile, axis=0) >= 0),
        "Only the completed branch-local causal error profile is allowed",
    )
    trace = produced["runtime"]["margin_guarded_traces"][-1]
    prefix.require(
        trace["status"] == "margin_guarded_no_witness"
        and trace["selected_scope"] is None
        and trace["audit_status"] == "not_attempted"
        and trace["primary_responses"] + trace["fallback_responses"] == trace["response_evaluations"] <= 320,
        "The recorded shared-budget search stopped without dispatch",
    )
    return dict(
        periods=p["periods"],
        profile=profile,
        shifted=shifted_completed_plan(last),
        prior_primary_responses=trace["primary_responses"],
        prior_fallback_responses=trace["fallback_responses"],
    )


def load_inputs():
    config_ref, config = read_config()
    old = previous.load_inputs()
    previous_receipt = copy.deepcopy(previous.input_receipt(old))
    refs = dict(old["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    producer, checker = (prefix.read(ref, refs) for ref in (PRODUCER, CHECKER))
    branches = {}
    for phase, result in (("producer", producer), ("checker", checker)):
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["control_qualified"] is False
            and result["phase"] == phase
            and result["world_model_calls"] == result["fresh_cases"] == 0,
            "Closed shared-budget feedback and independent replay",
        )
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            registration["inputs"] == previous_receipt and registration["source_refs"] == previous.source_refs(),
            "Exact prior feedback inputs and source identities",
        )
        for ref in result["branch_refs"]:
            branch = prefix.read(ref, refs)
            key = (phase, branch["case"]["case_index"])
            prefix.require(key not in branches, "Unique feedback branch")
            branches[key] = branch
    prefix.require(
        set(branches) == {(phase, spec["case_index"]) for phase in ("producer", "checker") for spec in CASES},
        "Both prior feedback branches are required",
    )
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        produced, checked = branches[("producer", index)], branches[("checker", index)]
        case = dict(next(item for item in old["cases"] if item["spec"]["case_index"] == index))
        completed = completed_stop(produced, checked, spec)
        case.update(
            stop_spec=spec,
            feedback_periods=completed["periods"],
            stop_inputs=completed,
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
