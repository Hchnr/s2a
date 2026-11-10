"""Audit the shifted completed plans at the two guarded task-2 stops."""

import argparse
import copy
from pathlib import Path

import numpy as np

from s2a.geometry import finite, inverse, pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.joint_control.tracking.plan_model_drift import shifted_completed_plan
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from scripts.simulation.runtime import prepare_task2_guarded_model_drift as previous_binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as context
from scripts.tracking.trajectory_search import audit_task2_guarded_model_drift as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_shifted_seed_qualification_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_shifted_seed_qualification_20261111.md"
QA = REPO / "outputs/qa/task2_shifted_seed_qualification_20261111/result.json"
DRY = REPO / "outputs/task2_shifted_seed_qualification_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_shifted_seed_qualification_20261111"
PRIOR = {
    "path": str(REPO / "outputs/task2_guarded_model_drift_20261111/result.json"),
    "sha256": "4e9bcddb23ce4f7cf7e39afece182e6d312046d515ba142b9be027a0f15ff3ff",
}
CASES = [dict(case_index=10, stop_step=33), dict(case_index=11, stop_step=84)]
INDICES = (10, 11)


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (Path(__file__), CONFIG, PLAN, REPO / "tests/research/native_runtime/test_task2_shifted_seed.py")
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-shifted-seed-qualification.v1",
            cases=CASES,
            seed=43,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            plan_horizon=8,
            hard_pose_prefix=2,
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
        "Exact stopped-state shifted-seed qualification and budgets",
    )
    return ref, config


def qualify(errors, safety_passed):
    """Score the original two-point caps separately from the six-point tail."""
    errors = finite(errors, (8, 6))
    if type(safety_passed) is not bool:
        raise ValueError("Explicit independent safety result required")
    position, rotation = np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)
    point_passed = (position <= 0.01 + 1e-10) & (rotation <= 0.05 + 1e-10)
    future = max(0.0, float(np.max(np.r_[position[2:] / 0.01 - 1, rotation[2:] / 0.05 - 1])))
    return dict(
        position_error_m=position,
        rotation_error_rad=rotation,
        point_pose_caps_satisfied=point_passed,
        prefix_pose_caps_satisfied=bool(np.all(point_passed[:2])),
        future_quality_slack=future,
        all_pose_caps_satisfied=bool(np.all(point_passed)),
        model_prefix_candidate=bool(safety_passed and np.all(point_passed[:2])),
    )


def load_inputs():
    config_ref, config = read_config()
    old = previous_binding.load_inputs()
    previous_receipt = copy.deepcopy(previous_binding.input_receipt(old))
    refs = dict(old["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = prefix.read(PRIOR, refs)
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_qualified"] is False
        and prior["physics_steps"] == prior["world_model_calls"] == prior["fresh_cases"] == 0,
        "Completed prior static drift analysis without control qualification",
    )
    registration = prefix.read(prior["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == previous_receipt and registration["source_refs"] == previous.source_refs(),
        "Prior drift inputs and frozen source identity",
    )
    results = {}
    for ref in prior["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(item["complete"] is item["passed"] is True and index not in results, "Unique closed drift case")
        results[index] = item
    prefix.require(sorted(results) == list(INDICES), "Both guarded feedback branches")
    cases = []
    for spec in CASES:
        index = spec["case_index"]
        case = dict(next(item for item in old["cases"] if item["drift_spec"]["case_index"] == index))
        result = results[index]
        prefix.require(
            result["case"] == case["drift_spec"]
            and len(result["data"]["transitions"]) == len(case["feedback_periods"])
            and case["feedback_periods"][-1]["step"] == spec["stop_step"] - 1,
            "The last completed period precedes the registered stop",
        )
        transition = result["data"]["transitions"][-1]
        prefix.require(
            transition["step"] == spec["stop_step"] - 1
            and transition["shifted_plan_safety"]["check"]["accepted"] is True,
            "Same completed period and original shifted-plan safety diagnostic",
        )
        case.update(stop_spec=spec, prior_transition=transition)
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
        prior_ref=PRIOR,
        cases=CASES,
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )


def new_stats():
    return dict(
        model_initializations_attempted=0,
        model_initializations_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        independent_audits_attempted=0,
        independent_audits_completed=0,
        independent_endpoint_queries_charged=0,
        independent_transition_steps_charged=0,
    )


def analyze(case, inputs, stats):
    context.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static shifted-seed qualification forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    periods = case["feedback_periods"]
    last = periods[-1]
    shifted = shifted_completed_plan(last)
    decision = dict(step=case["stop_spec"]["stop_step"], decision=dict(refined_plan=shifted))
    current = context.restored_context(model, data, case, periods + [decision], len(periods), stats)
    np.testing.assert_array_equal(current["state"], last["actual_state"])
    np.testing.assert_array_equal(current["previous"], last["command"])
    np.testing.assert_array_equal(current["plan"], shifted)
    np.testing.assert_allclose(current["q"], last["actual_inner"][-1]["q"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(current["velocity"], last["actual_inner"][-1]["qvel"], atol=1e-10, rtol=0)
    cursor = case["intervention_spec"]["cursor"] + len(periods)
    targets = np.asarray(case["targets"][cursor : cursor + 8])
    prefix.require(targets.shape == (8, 4, 4), "Eight already-generated reference points")
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    errors = np.asarray(
        [pose_error(target, pose @ tool) for target, pose in zip(targets, current["rollout"]["pose"], strict=True)]
    )
    np.testing.assert_allclose(
        errors[0], case["prior_transition"]["next_forecast"]["dynamics_refreshed"], atol=1e-10, rtol=0
    )
    profile = np.asarray(last["completed_feedback"]["past_error_after_completion"]["inner_position_error_rad"])
    domain = linear_domain(
        current["model"],
        shifted,
        current["previous"][:6],
        current["interface"],
        current["limits"][:, 0],
        current["limits"][:, 1],
        profile,
        current["rollout"],
    )
    check = domain_check(shifted, domain, current["interface"])
    prefix.require(
        check == case["prior_transition"]["shifted_plan_safety"]["check"], "Exact prior shifted-plan safety check"
    )
    # The API's witness flag admits a supplied plan for audit; it does not claim pose quality.
    candidate = dict(
        witness_found=True,
        planned_commands=shifted,
        rollout=current["rollout"],
        first_tcp_error=errors[0],
        tcp_errors=errors,
        terminal_normals=domain["terminal_normals"],
    )
    stats["independent_audits_attempted"] += 1
    stats["independent_endpoint_queries_charged"] += 200
    stats["independent_transition_steps_charged"] += 2600
    try:
        audited = audit_full_horizon_witness(
            candidate,
            current["mass"],
            current["jacobian"],
            current["contract"]["kp"],
            current["contract"]["kd"],
            current["q"],
            current["velocity"],
            case["initial_joint"],
            current["interface"],
            current["previous"][:6],
            current["site"],
            tool,
            targets,
            current["limits"][:, 0],
            current["limits"][:, 1],
            past_inner_error_rad=profile,
            enforce_pose_caps=False,
            enforce_braking=True,
        )
        prefix.require(
            audited["independent_endpoint_queries"] == 200 and audited["independent_transition_steps"] == 2600,
            "Full independent audit cost",
        )
        stats["independent_audits_completed"] += 1
        audit = dict(status="passed", verification=audited)
    except (AssertionError, ValueError) as error:
        audit = dict(status="rejected", error=dict(type=type(error).__name__, message=str(error)))
    quality = qualify(errors, check["accepted"] and audit["status"] == "passed")
    return dict(
        first_target_cursor=cursor,
        shifted_commands=shifted,
        tcp_errors=errors,
        linear_safety=check,
        terminal_normals=domain["terminal_normals"],
        terminal_scope="derived from the shifted plan, not the original stopped solver seed",
        independent_audit=audit,
        quality=quality,
        scope="static frozen-model seed assessment; no new action or physical feasibility claim",
    )


def require_case_counts(stats):
    prefix.require(
        stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
        and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == stats["baseline_rollouts"] == 1
        and stats["independent_audits_attempted"] == 1
        and stats["independent_audits_completed"] in (0, 1)
        and stats["independent_endpoint_queries_charged"] == 200
        and stats["independent_transition_steps_charged"] == 2600,
        "Exact per-case static and conservatively charged audit costs",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen stop inputs and source identity",
    )
    case = next(item for item in inputs["cases"] if item["stop_spec"]["case_index"] == index)
    stats = new_stats()
    result = dict(complete=False, passed=False, control_qualified=False, case=case["stop_spec"], stats=stats)
    try:
        data = analyze(case, inputs, stats)
        require_case_counts(stats)
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def summarize(cases, config):
    prefix.require([item["case"] for item in cases] == CASES, "Exact registered stops")
    for item in cases:
        require_case_counts(item["stats"])
    keys = (
        "model_initializations_completed",
        "forward_queries_completed",
        "baseline_rollouts",
        "independent_audits_attempted",
        "independent_audits_completed",
        "independent_endpoint_queries_charged",
        "independent_transition_steps_charged",
    )
    counts = {key: sum(item["stats"][key] for item in cases) for key in keys}
    prefix.require(
        counts["model_initializations_completed"] == config["max_models"] == 2
        and counts["forward_queries_completed"] == config["max_static_forward_queries"] == 2
        and counts["baseline_rollouts"] == config["max_eight_step_rollouts"] == 2
        and counts["independent_audits_attempted"] == config["max_independent_audits"] == 2
        and counts["independent_endpoint_queries_charged"] == config["max_independent_endpoint_queries"] == 400
        and counts["independent_transition_steps_charged"] == config["max_independent_transition_steps"] == 5200,
        "Registered aggregate model and independent audit budgets",
    )
    return counts


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
        registered_static_phase.execute_static_phase(
            root=OUTPUT,
            qa_path=QA,
            dry_path=DRY,
            indices=INDICES,
            worker_module="scripts.tracking.trajectory_search.audit_task2_shifted_seed",
            load_inputs=load_inputs,
            input_receipt=input_receipt,
            source_refs=source_refs,
            summarize=summarize,
        )
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
