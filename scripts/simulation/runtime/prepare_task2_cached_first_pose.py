"""Bind cache inspection to the independently replayed fallback stops."""

import copy
from pathlib import Path

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.plan_model_drift import shifted_completed_plan
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_causal_shift_fallback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_cached_first_pose_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_cached_first_pose_20261111.md"
PRODUCER = {
    "path": str(REPO / "outputs/task2_causal_shift_fallback_20261111/producer/result.json"),
    "sha256": "54e42a23f0d150d11ee3a896df8e28a04a7786edd538784dd91b8a76d74f4ad2",
}
CHECKER = {
    "path": str(REPO / "outputs/task2_causal_shift_fallback_20261111/checker/result.json"),
    "sha256": "4848e256de842e96cb66c9d07991a5da223e6015b1a07a0d1bd73b4e1b8c66c7",
}
CASES = [dict(case_index=10, stop_step=33), dict(case_index=11, stop_step=85)]
SEEDS = ("recorded_lp_start", "shifted_completed_plan")


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/cached_first_pose.py",
            REPO / "tests/research/trajectory_search/test_cached_first_pose.py",
            REPO / "tests/research/native_runtime/test_task2_cached_first_pose_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-cached-first-pose.v1",
            cases=CASES,
            seeds=list(SEEDS),
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            braking_time_s=0.1,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            max_responses_per_solve=320,
            max_models=2,
            max_static_forward_queries=2,
            max_preparation_rollouts=4,
            max_solver_calls=4,
            max_solver_responses=1280,
            max_independent_audits=4,
            max_independent_endpoint_queries=800,
            max_independent_transition_steps=10400,
            max_workers=2,
            worker_timeout_seconds=1800,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact stopped-state cached-first-pose scope and budgets",
    )
    return ref, config


def completed_stop(produced, checked, spec):
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
        and produced["branch"] == checked["branch"] == "causal_shift_fallback"
        and checked["candidate_modules_unloaded"] is True
        and p["stop"] == q["stop"]
        and p["stop"]["reason"] == "quality_no_witness"
        and p["stop"]["at_step"] == spec["stop_step"]
        and p["stop"]["next_action_dispatched"] is False
        and q["decision_stop_recorded_not_recomputed"] is True
        and len(p["periods"]) == len(q["periods"]) == n
        and n > 0,
        "Exact replayed feedback and recorded stop",
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
            "Every completed inner sample and outcome label must match replay",
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
        "Completed branch-local causal error profile",
    )
    trace = produced["runtime"]["causal_shift_fallback_traces"][-1]
    info = trace["primary_initialization"]
    prefix.require(
        trace["primary_status"] == "two_step_no_witness"
        and trace["selected_scope"] is None
        and trace["fallback_status"] == "model_safety_or_first_pose_rejected"
        and trace["fallback_safety"]["accepted"] is True
        and trace["fallback_first_pose_caps_satisfied"] is False
        and info["warm_start_used"] is True
        and info["check"]["accepted"] is True,
        "Recorded failed primary search and rejected shifted first pose",
    )
    return dict(
        periods=p["periods"],
        profile=profile,
        shifted=shifted_completed_plan(last),
        lp_start=finite(info["warm_start_commands"], (8, 6)).copy(),
        recorded_shifted_errors=finite(trace["fallback_first_two_tcp_errors"], (2, 6)).copy(),
        prior_primary_responses=trace["primary_response_evaluations"],
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
            "Closed fallback feedback and independent replay",
        )
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            registration["inputs"] == previous_receipt and registration["source_refs"] == previous.source_refs(),
            "Exact original fallback inputs and source identity",
        )
        for ref in result["branch_refs"]:
            branch = prefix.read(ref, refs)
            key = (phase, branch["case"]["case_index"])
            prefix.require(key not in branches, "Unique fallback branch")
            branches[key] = branch
    prefix.require(
        set(branches) == {(phase, spec["case_index"]) for phase in ("producer", "checker") for spec in CASES},
        "Both prior branches",
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
        seeds=list(SEEDS),
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
