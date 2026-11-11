"""Run and independently replay causal close deferral on one studied release."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.simulation.runtime import prepare_defer_close_task1 as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as base
from scripts.simulation.runtime import run_causal_opening_feedback as earlier
from scripts.verification.reference_ports.check_defer_close_release import independent_proposal, verify_proposal

REPO = prefix.REPO
ROOT = REPO / "outputs/defer_close_task1_feedback_20261111"
DRY = REPO / "outputs/defer_close_task1_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/defer_close_task1_feedback_20261111/result.json"
BRANCHES = ((7, "deferred"),)
FORBIDDEN = base.FORBIDDEN_CHECKER_MODULES + (
    "s2a.research.gripper.defer_close_release",
    "s2a.research.gripper.defer_close_transaction",
    "s2a.research.gripper.loaded_gripper_transaction",
)


def source_refs():
    return (
        binding.source_refs()
        + earlier.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                REPO / "src/s2a/research/gripper/defer_close_release.py",
                REPO / "src/s2a/research/gripper/defer_close_transaction.py",
                REPO / "src/s2a/research/native_runtime/guarded_correct_prefix.py",
                REPO / "src/s2a/research/native_runtime/guarded_task1_release_prefix.py",
                REPO / "scripts/verification/reference_ports/check_defer_close_release.py",
                REPO / "scripts/wam/tracking/preview/shadow_defer_close_release.py",
                REPO / "tests/research/native_runtime/test_defer_close_task1_feedback.py",
                REPO / "tests/research/gripper/test_defer_close_transaction.py",
            )
        ]
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered task1 deferral branch required")
    return f"case_{index:02d}_{mode}"


def checker_isolation():
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Checker cannot import candidate policies")


def reconstruct_gripper(case, result):
    from s2a.research.gripper.causal_loaded_history import reconstruct_loaded_history
    from s2a.research.gripper.defer_close_transaction import DeferringLoadedGripperTransaction
    from scripts.simulation.runtime.reconstruct_causal_gripper_history import read_chunk_prefix

    n = case["spec"]["prefix_steps"]
    refs = case["gripper_inputs"]
    old_chunks = read_chunk_prefix(prefix.checked(refs["chunks_ref"]), n // 8)
    controller, history = reconstruct_loaded_history(
        case["rows"],
        old_chunks,
        case["tool"],
        prefix.read(refs["parameters_ref"]),
        velocity_envelope=case["controller"]["gripper_velocity_envelope"],
        gripper_config=case["controller"]["gripper_config"],
        progress=lambda progress: result.update(gripper_reconstruction_progress=progress),
    )
    prefix.require(
        history["steps"] == n
        and history["last_completed_command"] == float(case["commands"][-1, 6])
        and history["candidate_sequences"] == history["planner_calls"] * 3**8,
        "Complete original causal gripper history required",
    )
    result["gripper_history"] = history
    return DeferringLoadedGripperTransaction(controller)


def check_recorded_proposal(env, case, selected, offset, period):
    """Recompute the predicate using current measurements and the recorded baseline proposal."""
    checker_isolation()
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    gripper = base.measured_gripper(env)
    detail = period["decision"]["gripper_decision"]
    original, proposed = detail["baseline"], detail["deferral"]
    checked = independent_proposal(
        selected["targets"],
        selected["openings"],
        selected["cursor"] + offset,
        np.asarray(site) @ site_from_tcp,
        gripper["opening"],
        gripper["motor_target"],
        original["command"],
        original["latch"]["closing_episode"],
    )
    difference = verify_proposal(proposed, checked)
    prefix.require(
        checked["command"] == detail["command"] == float(period["command"][6]),
        "Independent predicate must match the recorded dispatched command",
    )
    return difference


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        registration["inputs"] == binding.input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Frozen task1 inputs and source identity required",
    )
    case, selected = inputs["case"], inputs["selected"]
    baseline = inputs["baseline"]["runtime"]["continuation"]["periods"]
    expected, original = None, None
    if phase == "checker":
        checker_isolation()
        original = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(original["complete"] is original["passed"] is True, "Closed candidate branch required")
        expected = original["runtime"]["continuation"]["periods"]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case=case["spec"],
        branch=mode,
        automatic_retry=False,
    )
    try:
        gripper = None if phase == "checker" else reconstruct_gripper(case, result)
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.guarded_task1_release_prefix import GuardedTask1ReleasePrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            policy = None
            if phase == "checker":
                checker_isolation()
                actual_prefix, recorded_prefix = (
                    receipt["prefix_inner_periods"],
                    original["runtime"]["prefix_inner_periods"],
                )
                prefix.require(len(actual_prefix) == len(recorded_prefix) == 232, "Complete replayed prefix required")
                for actual_period, recorded_period in zip(actual_prefix, recorded_prefix, strict=True):
                    base.check_prefix_inner(actual_period, recorded_period)
                base.check_prefix_inner(receipt["final_prefix_inner"], original["runtime"]["final_prefix_inner"])
                receipt["independent_deferral_max_difference"] = 0.0

                def action(_q, _limits, _jacobian, _previous, offset):
                    difference = check_recorded_proposal(env, case, selected, offset, expected[offset])
                    receipt["independent_deferral_max_difference"] = max(
                        receipt["independent_deferral_max_difference"], difference
                    )
                    return dict(command=np.asarray(expected[offset]["command"]), newly_selected=False)

            else:
                from s2a.research.gripper.causal_loaded_history import compare_recorded

                policy = base.make_feedback(env, case, selected, gripper, receipt)

                def action(q, limits, jacobian, previous, offset):
                    decision = policy.select(q, limits, jacobian, previous, offset)
                    if "stop" in decision:
                        return decision
                    detail = decision["gripper_decision"]
                    if offset < 2:
                        np.testing.assert_allclose(decision["command"], baseline[offset]["command"], atol=1e-10, rtol=0)
                        compare_recorded(detail["baseline"], baseline[offset]["decision"]["gripper_decision"])
                        prefix.require(detail["deferral"]["deferred"] is False, "No earlier divergence is admitted")
                    if offset == 2:
                        prefix.require(
                            detail["baseline"]["command"] == 1.0
                            and detail["deferral"]["deferred"] is True
                            and float(decision["command"][6]) == 0.0,
                            "Original step234 trigger must be reproduced before dispatch",
                        )
                    return decision

            def observed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [item["q"] for item in period["actual_inner"]]
                    )
                expected_score = None if expected is None else expected[offset]["completed_feedback"]["scores"]
                if phase == "producer" and offset < 2:
                    np.testing.assert_allclose(
                        period["actual_state"], baseline[offset]["actual_state"], atol=1e-10, rtol=0
                    )
                    prefix.require(
                        period["target_grasp"] == baseline[offset]["target_grasp"]
                        and period["task_success"] == baseline[offset]["task_success"],
                        "Pre-intervention labels must match the original baseline",
                    )
                    expected_score = baseline[offset]["completed_feedback"]["scores"]
                scores = base.score_completed(
                    actual_env, case, selected, "latest", offset, period["tcp_error"], expected=expected_score
                )
                return dict(scores=scores, policy_history=history, policy_history_recomputed=policy is not None)

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=232,
                previous_command=case["commands"][-1],
                targets=selected["fixed_targets"],
                tool=case["tool"],
                target_object=case["spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=observed,
                progress=progress,
            )
            if phase == "checker":
                q, limits, _jac = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                return base.reconcile_stop(
                    outcome,
                    original["runtime"]["continuation"]["stop"],
                    start_step=232,
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
            return outcome

        try:
            runtime = GuardedTask1ReleasePrefixRuntime().run(
                case["config"],
                case["record"],
                case["record_xml"],
                case["commands"],
                case["saved"],
                case["receipts"],
                directory / "runtime",
                inputs["asset_refs"],
                case["controller"],
                continuation=continuation,
                continuation_budget=8,
                registered_seed=43,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            checker_isolation()
        result.update(
            complete=True,
            passed=True,
            runtime=runtime,
            candidate_modules_unloaded=all(name not in sys.modules for name in FORBIDDEN),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_phase(branches, phase, config):
    budget = dict(
        control_steps=config["max_control_steps_per_phase"],
        native_steps=config["max_native_steps_per_phase"],
        model_initializations=3,
        contexts=3,
        gripper_planner_calls=config["max_gripper_planner_calls"],
        gripper_candidate_sequences=config["max_gripper_candidate_sequences"],
        refiner_calls=config["max_refiner_calls"],
        refiner_response_evaluations=config["max_refiner_responses"],
        quality_calls=config["max_quality_calls"],
        quality_response_evaluations=config["max_quality_responses"],
    )
    prefix.require(len(branches) == 1, "One registered branch per phase")
    counts = earlier.summarize_phase(branches, phase, budget)
    runtime = branches[0]["runtime"]
    periods = runtime["prefix_inner_periods"]
    prefix.require(
        len(periods) == 232
        and all(len(period["samples"]) == 25 for period in periods)
        and runtime["checked_endpoints"] == 233
        and (phase != "checker" or branches[0]["candidate_modules_unloaded"]),
        "All prefix samples, endpoints and independent checker isolation required",
    )
    counts.update(
        prefix_inner_samples=sum(len(period["samples"]) for period in periods),
        prefix_minimum_margin_rad=min(period["minimum_margin_rad"] for period in periods),
        prefix_below_buffer_samples=sum(period["below_buffer_samples"] for period in periods),
        continuation_steps=len(runtime["continuation"]["periods"]),
    )
    return counts


def dry_run():
    from scripts.wam.tracking.preview.shadow_defer_close_release import analyze_one

    inputs = binding.load_inputs()
    selected, baseline = inputs["selected"], inputs["baseline"]["runtime"]["continuation"]["periods"]
    prior = baseline[1]["completed_feedback"]["scores"]
    result, difference = analyze_one(
        selected["targets"],
        selected["openings"],
        2,
        np.asarray(prior["actual_tcp"]),
        prior["gripper"]["opening"],
        prior["gripper"]["motor_target"],
        baseline[2]["command"][6],
        baseline[2]["decision"]["gripper_decision"],
    )
    prefix.require(result["deferred"] is True and result["command"] == 0.0, "Original release trigger must be present")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=binding.input_receipt(inputs),
                source_refs=source_refs(),
                independent_max_abs=difference,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        )
    )


def execute(phase):
    return registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_defer_close_task1_feedback",
        branch_key=branch_key,
        load_inputs=binding.load_inputs,
        input_receipt=binding.input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="config",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "Known feedback phase required")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
