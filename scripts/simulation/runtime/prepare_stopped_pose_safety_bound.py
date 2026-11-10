"""Bind three-task stop states to completed causal histories and replays."""

import copy
from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_multitask_feasibility_first as multitask
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_staged_first_feasibility as staged

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/stopped_pose_safety_bound_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/stopped_pose_safety_bound_20261111.md"
MULTITASK = {
    "path": str(REPO / "outputs/multitask_feasibility_first_20261111/result.json"),
    "sha256": "4aa9e97a3384d6e88c7f6932a5d4aa7c4120a681a99728bcbb8dc4c831d70d09",
}
STAGED_PRODUCER = {
    "path": str(REPO / "outputs/task2_staged_first_feasibility_20261111/producer/result.json"),
    "sha256": "6010b4acbac2f489cf5720318408e8729085697a6e333d77b1d2a05cb0cec3d6",
}
STAGED_CHECKER = {
    "path": str(REPO / "outputs/task2_staged_first_feasibility_20261111/checker/result.json"),
    "sha256": "427010338d1f617b4cb8d6a479aefd0d314c91b42ba900f5a97428e219d4302d",
}
CASES = [
    dict(case_index=2, step=48, source="multitask_quality_stop"),
    dict(case_index=6, step=57, source="multitask_quality_stop"),
    dict(case_index=10, step=33, source="staged_first_stop"),
    dict(case_index=11, step=81, source="staged_first_stop"),
]


def source_refs():
    return (
        multitask.source_refs()
        + staged.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                CONFIG,
                PLAN,
                REPO / "src/s2a/research/joint_control/tracking/stopped_pose_safety_bound.py",
                REPO / "scripts/tracking/trajectory_search/audit_stopped_pose_safety_bound.py",
                REPO / "scripts/verification/reference_ports/check_stopped_pose_safety_bound.py",
                REPO / "tests/research/joint_control/tracking/test_stopped_pose_safety_bound.py",
                REPO / "tests/research/native_runtime/test_stopped_pose_safety_bound_inputs.py",
                REPO / "tests/research/native_runtime/test_stopped_pose_safety_bound_check.py",
            )
        ]
    )


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == dict(
            schema="s2a.stopped-pose-safety-bound.v1",
            cases=CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            scopes=["box", "first_interval_safety", "eight_interval_safety"],
            position_weights=[0.0, 0.25, 0.5, 0.75, 1.0],
            normalized_numerical_allowance=1e-7,
            terminal_direction_included=False,
            max_models=4,
            max_static_forward_queries=4,
            max_seed_rollouts=4,
            max_support_lp_calls=40,
            max_independent_first_responses=4,
            max_independent_transition_steps=100,
            max_workers=4,
            worker_timeout_seconds=600,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact registered four-case pose and safety bound",
    )
    return ref, value


def completed_staged_stop(produced, checked, spec):
    """Use only the last actually dispatched action, its endpoint and history."""
    p, q = produced["runtime"]["continuation"], checked["runtime"]["continuation"]
    prefix.require(
        produced["complete"] is produced["passed"] is checked["complete"] is checked["passed"] is True
        and checked["candidate_modules_unloaded"] is True
        and p["stop"] == q["stop"]
        and p["stop"]["reason"] == "quality_no_witness"
        and p["stop"]["at_step"] == spec["step"]
        and p["stop"]["next_action_dispatched"] is False
        and q["decision_stop_recorded_not_recomputed"] is True
        and len(p["periods"]) == len(q["periods"]) == 1,
        "Closed one-period feedback and independent replay at the registered stop",
    )
    left, right = p["periods"][0], q["periods"][0]
    prefix.require(left["step"] == right["step"] == spec["step"] - 1, "Consecutive completed period")
    for key in ("command", "actual_state", "tcp_error"):
        np.testing.assert_array_equal(left[key], right[key])
    prefix.require(left["actual_inner"] == right["actual_inner"], "All completed inner samples were replayed")
    plan = np.asarray(left["decision"]["refined_plan"], dtype=float)
    previous = np.asarray(left["command"], dtype=float)
    prefix.require(plan.shape == (8, 6) and previous.shape == (7,), "Completed eight-command plan")
    np.testing.assert_array_equal(plan[0], previous[:6])
    shifted = np.vstack((plan[1:], plan[-1]))
    prefix.require(
        np.isfinite(shifted).all()
        and np.max(abs(shifted)) <= 1 + 1e-12
        and np.max(abs(np.diff(np.vstack((previous[:6], shifted)), axis=0))) <= 0.25 + 1e-12,
        "Shifted completed plan retains original command box and rate",
    )
    past = left["completed_feedback"]["past_error_after_completion"]
    profile = np.asarray(past["inner_position_error_rad"], dtype=float)
    prefix.require(
        past["available"] is True
        and past["future_guarantee"] is False
        and past["completed_periods"] == 1
        and past["last_completed_step"] == spec["step"] - 1
        and profile.shape == (25, 7)
        and np.isfinite(profile).all()
        and np.all(profile >= 0)
        and np.all(np.diff(profile, axis=0) >= 0),
        "Only the completed past inner-error profile is available",
    )
    trace = produced["runtime"]["first_hard_restoration_traces"][-1]
    closest = trace["closest_first_rejected"]
    prefix.require(
        trace["status"] == "staged_first_no_witness" and closest is not None,
        "Recorded stopped search has a closest linear-safe attempt",
    )
    return dict(
        periods=p["periods"],
        seed=shifted,
        profile=profile,
        closest_error=np.asarray(closest["first_tcp_error"], dtype=float),
        closest_commands=np.asarray(closest["planned_commands"], dtype=float),
    )


def load_inputs():
    config_ref, config = read_config()
    old, current = multitask.load_inputs(), staged.load_inputs()
    old_stage_receipt = copy.deepcopy(staged.input_receipt(current))
    prefix.require(old["asset_refs"] == current["asset_refs"], "The same frozen runtime assets are required")
    refs = {config_ref["path"]: config_ref["sha256"]}
    multitask.merge_refs(refs, old["source_sha256"])
    multitask.merge_refs(refs, current["source_sha256"])
    prior, producer, checker = (prefix.read(ref, refs) for ref in (MULTITASK, STAGED_PRODUCER, STAGED_CHECKER))
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            for item in (prior, producer, checker)
        )
        and all(item["control_qualified"] is False for item in (prior, producer, checker)),
        "All prior comparisons and replays closed without control qualification",
    )
    prior_cases = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        prefix.require(item["complete"] is item["passed"] is True, "Closed earlier static result")
        prior_cases[item["case"]["case_index"]] = item["data"]
    branches = {}
    for phase, result in (("producer", producer), ("checker", checker)):
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            result["phase"] == phase
            and registration["inputs"] == old_stage_receipt
            and registration["source_refs"] == staged.source_refs(),
            "Staged feedback retains its exact causal inputs and source identity",
        )
        for ref in result["branch_refs"]:
            item = prefix.read(ref, refs)
            key = (phase, item["case"]["case_index"])
            prefix.require(key not in branches and item["branch"] == "staged_first_feasibility", "Unique staged branch")
            branches[key] = item
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        if spec["source"] == "multitask_quality_stop":
            case = dict(next(item for item in old["cases"] if item["spec"]["case_index"] == index))
            prior_data = prior_cases[index]
            prefix.require(
                prior_data["case"] == case["comparison_spec"]
                and case["stop_inputs"]["stop_step"] == spec["step"]
                and len(case["feedback_periods"]) == 5,
                "Same previously replayed five-period stop",
            )
            closest = prior_data["comparison"]["solvers"]["feasibility_first"]["solver_result"][
                "closest_linear_feasible_attempt"
            ]
            case.update(
                bound_seed=np.asarray(case["stop_inputs"]["seed"]),
                bound_profile=np.asarray(case["stop_inputs"]["past_inner_error_rad"]),
                closest_error=np.asarray(closest["first_tcp_error"]),
                closest_commands=np.asarray(closest["commands"]),
                previous_fingerprint=prior_data["comparison"]["input_sha256"],
            )
            np.testing.assert_allclose(case["bound_profile"], prior_data["past_inner_error_rad"], atol=1e-10, rtol=0)
        else:
            case = dict(next(item for item in current["cases"] if item["spec"]["case_index"] == index))
            produced, checked = branches[("producer", index)], branches[("checker", index)]
            stop = completed_staged_stop(produced, checked, spec)
            case.update(
                feedback_periods=stop["periods"],
                initial_joint=np.asarray(produced["runtime"]["runtime_initial_joint"], dtype=float),
                bound_seed=stop["seed"],
                bound_profile=stop["profile"],
                closest_error=stop["closest_error"],
                closest_commands=stop["closest_commands"],
                previous_fingerprint=None,
                seed_scope="shifted_last_completed_staged_plan_not_original_stop_seed",
            )
        prefix.require(
            case["config"]["seed"] == config["seed"]
            and case["bound_seed"].shape == case["closest_commands"].shape == (8, 6)
            and case["bound_profile"].shape == (25, 7)
            and case["closest_error"].shape == (6,)
            and case["spec"]["prefix_steps"] + len(case["feedback_periods"]) == spec["step"],
            "Correct stopped state, bounded plan and completed history",
        )
        case["bound_spec"] = spec
        cases.append(case)
    return dict(config_ref=config_ref, config=config, cases=cases, asset_refs=old["asset_refs"], source_sha256=refs)


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        prior_refs=[MULTITASK, STAGED_PRODUCER, STAGED_CHECKER],
        cases=[dict(**case["bound_spec"], seed_scope=case["seed_scope"]) for case in inputs["cases"]],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
