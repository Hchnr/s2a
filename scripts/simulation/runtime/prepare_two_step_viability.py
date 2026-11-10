"""Bind pre-stop model inputs without exposing the next physical endpoint."""

import copy
from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_stopped_pose_safety_bound as stopped
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/two_step_viability_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/two_step_viability_20261111.md"
STOPPED = {
    "path": str(REPO / "outputs/stopped_pose_safety_bound_20261111/result.json"),
    "sha256": "c91b00b8e3ecf9de6081d195e3715d63a58b690fae3550ac25037d081a721968",
}
CASES = [
    dict(case_index=2, step=47),
    dict(case_index=6, step=56),
    dict(case_index=10, step=32),
    dict(case_index=11, step=80),
]


def source_refs():
    return stopped.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/two_step_viability.py",
            REPO / "src/s2a/research/joint_control/tracking/two_step_quality_audit.py",
            REPO / "scripts/tracking/trajectory_search/audit_two_step_viability.py",
            REPO / "scripts/simulation/runtime/registered_static_phase.py",
            REPO / "tests/research/trajectory_search/test_two_step_viability.py",
            REPO / "tests/research/native_runtime/test_two_step_viability_inputs.py",
            REPO / "tests/research/native_runtime/test_two_step_viability_runtime.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == dict(
            schema="s2a.two-step-viability.v1",
            cases=CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            plan_horizon=8,
            hard_pose_prefix=2,
            max_responses_per_solve=320,
            max_feasibility_iterations=60,
            max_total_iterations=120,
            max_start_lp_per_solve=2,
            max_start_lp_calls=8,
            max_models=4,
            max_static_forward_queries=4,
            max_preparation_seed_rollouts=4,
            max_solver_calls=4,
            max_solver_responses=1280,
            max_audit_endpoint_queries=800,
            max_audit_transition_steps=10400,
            max_workers=4,
            worker_timeout_seconds=1800,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact four pre-stop decisions and two-point feasibility budget",
    )
    return ref, value


def causal_decision(case, spec):
    """Retain the current saved plan and earlier completed periods only."""
    periods = case["feedback_periods"]
    prefix.require(periods and periods[-1]["step"] == spec["step"], "Last completed recorded action is this decision")
    index = len(periods) - 1
    decision = periods[-1]["decision"]
    plan = np.asarray(decision["refined_plan"], dtype=float)
    command = np.asarray(periods[-1]["command"], dtype=float)
    prefix.require(plan.shape == (8, 6) and command.shape == (7,), "An eight-step dispatched decision is required")
    np.testing.assert_array_equal(plan[0], command[:6])
    if index:
        past = periods[index - 1]["completed_feedback"]["past_error_after_completion"]
        profile = np.asarray(past["inner_position_error_rad"], dtype=float)
        prefix.require(
            past["available"] is True
            and past["future_guarantee"] is False
            and past["completed_periods"] == index
            and past["last_completed_step"] == spec["step"] - 1,
            "Only the profile from periods completed before this decision",
        )
        profile_scope = "completed_before_decision"
    else:
        profile = np.zeros((25, 7))
        profile_scope = "no_paired_inner_error_available"
    prefix.require(
        profile.shape == (25, 7)
        and np.isfinite(profile).all()
        and np.all(profile >= 0)
        and np.all(np.diff(profile, axis=0) >= 0),
        "Finite nondecreasing completed-error profile",
    )
    # The current period's actual endpoint/inner samples are deliberately absent.
    current = dict(step=spec["step"], decision=dict(refined_plan=plan.copy()))
    return dict(
        periods=periods[:index] + [current],
        offset=index,
        profile=profile,
        profile_scope=profile_scope,
        recorded_plan=plan.copy(),
        recorded_model_first_error=np.asarray(decision["quality"]["first_tcp_error"], dtype=float),
        observed_first_error=np.asarray(periods[-1]["tcp_error"], dtype=float),
    )


def load_inputs():
    config_ref, config = read_config()
    old = stopped.load_inputs()
    previous_receipt = copy.deepcopy(stopped.input_receipt(old))
    refs = dict(old["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(STOPPED, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Closed stopped-pose outer-bound study",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == previous_receipt and registration["source_refs"] == stopped.source_refs(),
        "The same replayed stop inputs and source identities",
    )
    prior_cases = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        prefix.require(item["complete"] is item["passed"] is True, "Closed independent support audit")
        prior_cases[item["case"]["case_index"]] = item["data"]
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        case = dict(next(item for item in old["cases"] if item["bound_spec"]["case_index"] == index))
        prior_case = prior_cases[index]
        prefix.require(
            prior_case["case"]["step"] == spec["step"] + 1
            and prior_case["excludes_first_pose_caps"]["box"] is True
            and case["config"]["seed"] == config["seed"],
            "The next recorded stop has a box-only frozen-model exclusion",
        )
        current = causal_decision(case, spec)
        prefix.require(
            case["spec"]["prefix_steps"] + current["offset"] == spec["step"]
            and current["recorded_model_first_error"].shape == current["observed_first_error"].shape == (6,),
            "Exact pre-stop decision and separate offline scoring",
        )
        case.update(decision_spec=spec, decision_inputs=current)
        cases.append(case)
    return dict(config_ref=config_ref, config=config, cases=cases, asset_refs=old["asset_refs"], source_sha256=refs)


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        stopped_ref=STOPPED,
        cases=[
            dict(**case["decision_spec"], profile_scope=case["decision_inputs"]["profile_scope"])
            for case in inputs["cases"]
        ],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
