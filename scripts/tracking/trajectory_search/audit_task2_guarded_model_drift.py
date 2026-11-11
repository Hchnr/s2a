"""Decompose model forecasts over five already-replayed feedback periods."""

import argparse
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.joint_control.tracking.plan_model_drift import (
    compare_joint_endpoint,
    compare_next_pose_forecasts,
    shifted_completed_plan,
)
from s2a.research.response_models.forward.joint_limit_response import build_joint_response, predict_joint_response
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from scripts.simulation.runtime import prepare_task2_guarded_model_drift as binding
from scripts.simulation.runtime import registered_static_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as previous

REPO = prefix.REPO
QA = REPO / "outputs/qa/task2_guarded_model_drift_20261111/result.json"
DRY = REPO / "outputs/task2_guarded_model_drift_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_guarded_model_drift_20261111"
INDICES = (10, 11)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "tests/research/native_runtime/test_task2_guarded_model_drift_runtime.py",
        )
    ]


def new_stats():
    return dict(
        model_initializations_attempted=0,
        model_initializations_completed=0,
        forward_queries_attempted=0,
        forward_queries_completed=0,
        baseline_rollouts=0,
        max_qpos_restore_difference=0.0,
        counterfactual_builds_attempted=0,
        counterfactual_builds_completed=0,
        counterfactual_rollouts_attempted=0,
        counterfactual_rollouts_completed=0,
        independent_first_responses_charged=0,
        independent_first_responses_completed=0,
        independent_transition_steps_charged=0,
        independent_transition_steps_completed=0,
        independent_scalar_models_built=0,
    )


def decompose_next_forecast(old, current, initial_joint, target, site_from_tcp, stats):
    """Reanchor one original next command without changing frozen old M/J."""
    stats["counterfactual_builds_attempted"] += 1
    model = build_joint_preview(
        old["mass"],
        old["jacobian"],
        old["contract"]["kp"],
        old["contract"]["kd"],
        current["q"],
        initial_joint,
        null_kp=10.0,
        dt=0.002,
    )
    stats["counterfactual_builds_completed"] += 1
    delta = np.asarray([current["interface"].decode(current["plan"][0])])
    scores = []
    for velocity in (old["rollout"]["joint_velocity"][0], current["velocity"]):
        stats["counterfactual_rollouts_attempted"] += 1
        rollout = rollout_joint_preview(
            model,
            velocity,
            delta,
            current["site"],
            current["limits"][:, 0],
            current["limits"][:, 1],
            enforce_limits=False,
        )
        stats["counterfactual_rollouts_completed"] += 1
        scores.append(pose_error(target, rollout["pose"][0] @ site_from_tcp))
    original = pose_error(target, old["rollout"]["pose"][1] @ site_from_tcp)
    refreshed = pose_error(target, current["rollout"]["pose"][0] @ site_from_tcp)
    return compare_next_pose_forecasts(original, scores[0], scores[1], refreshed)


def independent_first_response(context, initial_joint, command, stats):
    """Rebuild a scalar 25-step transition without reusing preview internals."""
    model = build_joint_response(
        context["mass"],
        context["jacobian"],
        context["contract"]["kp"],
        context["contract"]["kd"],
        context["q"],
        initial_joint,
        null_kp=10.0,
        dt=0.002,
    )
    stats["independent_scalar_models_built"] += 1
    stats["independent_first_responses_charged"] += 1
    stats["independent_transition_steps_charged"] += 25
    result = predict_joint_response(
        model,
        context["velocity"],
        context["interface"].decode(command),
        context["limits"][:, 0],
        context["limits"][:, 1],
        substeps=25,
        enforce_limits=False,
    )
    stats["independent_first_responses_completed"] += 1
    stats["independent_transition_steps_completed"] += 25
    difference = {}
    for key in ("joint_position", "joint_velocity"):
        difference[key] = float(np.max(abs(result[key] - context["rollout"][key][0])))
        np.testing.assert_allclose(result[key], context["rollout"][key][0], atol=1e-10, rtol=0)
    return difference


def analyze(case, inputs, stats):
    previous.preload_software_gl(case, inputs["source_sha256"])
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static model-drift diagnosis forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    periods = case["feedback_periods"]
    start = case["drift_spec"]["start_step"]
    tool = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    transitions = []
    for offset, period in enumerate(periods):
        saved_plan = np.asarray(period["decision"]["refined_plan"], dtype=float)
        current_decision = dict(step=start + offset, decision=dict(refined_plan=saved_plan))
        old = previous.restored_context(model, data, case, periods[:offset] + [current_decision], offset, stats)
        shifted = shifted_completed_plan(period)
        next_decision = dict(step=start + offset + 1, decision=dict(refined_plan=shifted))
        current = previous.restored_context(
            model, data, case, periods[: offset + 1] + [next_decision], offset + 1, stats
        )
        np.testing.assert_array_equal(old["plan"], saved_plan)
        np.testing.assert_array_equal(current["plan"], shifted)
        np.testing.assert_array_equal(current["state"], period["actual_state"])
        np.testing.assert_array_equal(current["previous"], period["command"])
        actual_q = np.asarray(period["actual_inner"][-1]["q"], dtype=float)
        actual_velocity = np.asarray(period["actual_inner"][-1]["qvel"], dtype=float)
        np.testing.assert_allclose(current["q"], actual_q, atol=1e-10, rtol=0)
        np.testing.assert_allclose(current["velocity"], actual_velocity, atol=1e-10, rtol=0)
        cursor = case["intervention_spec"]["cursor"] + offset
        targets = np.asarray(case["targets"][cursor : cursor + 2])
        prefix.require(targets.shape == (2, 4, 4), "Two already-generated reference points")
        saved_errors = np.asarray(period["decision"]["quality"]["verification"]["tcp_errors"][:2])
        rebuilt_errors = np.asarray([pose_error(targets[i], old["rollout"]["pose"][i] @ tool) for i in range(2)])
        np.testing.assert_allclose(rebuilt_errors, saved_errors, atol=1e-10, rtol=0)
        profile = np.asarray(period["completed_feedback"]["past_error_after_completion"]["inner_position_error_rad"])
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
        residual = np.minimum(
            domain["joint_a"] @ shifted.ravel() - domain["joint_lo"],
            domain["joint_hi"] - domain["joint_a"] @ shifted.ravel(),
        )
        next_forecast = decompose_next_forecast(old, current, case["initial_joint"], targets[1], tool, stats)
        independent = dict(
            original=independent_first_response(old, case["initial_joint"], period["command"][:6], stats),
            reanchored=independent_first_response(current, case["initial_joint"], shifted[0], stats),
        )
        transitions.append(
            dict(
                step=period["step"],
                first_target_cursor=cursor,
                actual_first_tcp_error=np.asarray(period["tcp_error"]),
                model_first_tcp_error=rebuilt_errors[0],
                recorded_model_tcp_error_max_difference=float(np.max(abs(rebuilt_errors - saved_errors))),
                endpoint=compare_joint_endpoint(
                    old["rollout"]["joint_position"][0], old["rollout"]["joint_velocity"][0], actual_q, actual_velocity
                ),
                next_forecast=next_forecast,
                shifted_plan_safety=dict(
                    check=domain_check(shifted, domain, current["interface"]),
                    minimum_tightened_inner_residual_rad=float(np.min(residual[: 200 * 7])),
                    minimum_braking_residual_rad=float(np.min(residual[200 * 7 :])),
                    terminal_normals_count=len(domain["terminal_normals"]),
                    scope="terminal normals derive from the shifted plan, not the actual next solver seed",
                ),
                independent_first_response_max_difference=independent,
            )
        )
    return dict(
        transitions=transitions,
        scope="offline frozen-model counterfactuals of replayed periods; no action was dispatched",
    )


def require_case_counts(stats, periods):
    prefix.require(
        stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
        and stats["forward_queries_attempted"]
        == stats["forward_queries_completed"]
        == stats["baseline_rollouts"]
        == 2 * periods
        and stats["counterfactual_builds_attempted"] == stats["counterfactual_builds_completed"] == periods
        and stats["counterfactual_rollouts_attempted"] == stats["counterfactual_rollouts_completed"] == 2 * periods
        and stats["independent_first_responses_charged"]
        == stats["independent_first_responses_completed"]
        == stats["independent_scalar_models_built"]
        == 2 * periods
        and stats["independent_transition_steps_charged"]
        == stats["independent_transition_steps_completed"]
        == 50 * periods,
        "Exact per-case static and independent response costs",
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen feedback inputs and source identity",
    )
    case = next(item for item in inputs["cases"] if item["drift_spec"]["case_index"] == index)
    stats = new_stats()
    result = dict(complete=False, passed=False, control_qualified=False, case=case["drift_spec"], stats=stats)
    try:
        data = analyze(case, inputs, stats)
        require_case_counts(stats, case["drift_spec"]["completed_periods"])
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def summarize(cases, config):
    keys = (
        "model_initializations_completed",
        "forward_queries_completed",
        "baseline_rollouts",
        "counterfactual_builds_completed",
        "counterfactual_rollouts_completed",
        "independent_first_responses_completed",
        "independent_transition_steps_completed",
        "independent_scalar_models_built",
    )
    prefix.require([item["case"] for item in cases] == binding.CASES, "Exact registered development periods")
    for item in cases:
        require_case_counts(item["stats"], item["case"]["completed_periods"])
        prefix.require(
            len(item["data"]["transitions"]) == item["case"]["completed_periods"], "Every completed period analyzed"
        )
    counts = {key: sum(item["stats"][key] for item in cases) for key in keys}
    prefix.require(
        counts["model_initializations_completed"] == config["max_models"] == 2
        and counts["forward_queries_completed"] == config["max_static_forward_queries"] == 10
        and counts["baseline_rollouts"] == config["max_eight_step_rollouts"] == 10
        and counts["counterfactual_builds_completed"] == 5 <= config["max_counterfactual_model_builds"]
        and counts["counterfactual_rollouts_completed"] == config["max_counterfactual_first_rollouts"] == 10
        and counts["independent_first_responses_completed"] == config["max_independent_first_responses"] == 10
        and counts["independent_transition_steps_completed"] == config["max_independent_transition_steps"] == 250,
        "Registered aggregate static and independent response budgets",
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
        inputs = binding.load_inputs()
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=binding.input_receipt(inputs),
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
            worker_module="scripts.tracking.trajectory_search.audit_task2_guarded_model_drift",
            load_inputs=binding.load_inputs,
            input_receipt=binding.input_receipt,
            source_refs=source_refs,
            summarize=summarize,
        )
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
