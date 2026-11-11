"""Compare a model-safe bounded tracking fallback on four development starts."""

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.simulation.runtime import build_model_safe_tracking_feedback as builder
from scripts.simulation.runtime import prepare_model_safe_tracking_fallback as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_causal_event_long_feedback as long
from scripts.simulation.runtime import run_early_quality_feedback as quality
from scripts.simulation.runtime import run_replan_quality_feedback as replan_quality
from scripts.simulation.runtime import run_replan_reference_switch as replan_switch
from scripts.simulation.runtime import run_replan_shared_shift_feedback as previous

REPO = prefix.REPO
ROOT = REPO / "outputs/model_safe_tracking_fallback_20261111"
QA = REPO / "outputs/qa/model_safe_tracking_fallback_20261111/result.json"
DRY = REPO / "outputs/model_safe_tracking_fallback_dry_20261111/result.json"
MODE = "model_safe_fallback"
DEGRADED = "model_safe_tracking_degraded"
BRANCHES = tuple((index, MODE) for index in (2, 6, 10, 11))
CHECKER_FORBIDDEN = tuple(
    dict.fromkeys(previous.CHECKER_FORBIDDEN + ("s2a.research.joint_control.preview.model_safe_tracking_fallback",))
)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "scripts/simulation/runtime/build_model_safe_tracking_feedback.py",
            REPO / "src/s2a/research/joint_control/preview/model_safe_tracking_fallback.py",
            REPO / "tests/research/native_runtime/test_model_safe_tracking_builder.py",
            REPO / "tests/research/native_runtime/test_model_safe_tracking_fallback_runtime.py",
            REPO / "tests/research/joint_control/preview/test_model_safe_tracking_fallback.py",
        )
    ]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered cross-task fallback branch")
    return f"case_{index:02d}_{mode}"


def require_checker_independence():
    prefix.require(all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot load candidate policies")


def verify_strict_period(period, prior_periods, offset, route, state):
    """Check prior physical outcomes after each period until the first fallback."""
    if state["fallback_seen"]:
        return
    if route == DEGRADED:
        prefix.require(offset == len(prior_periods), "Fallback must begin exactly at the earlier strict stop")
        state["fallback_seen"] = True
        return
    prefix.require(offset < len(prior_periods), "A changed strict continuation cannot be called a replay")
    long.check_previous_period(period, prior_periods[offset])
    state["strict_periods_verified"] += 1


def verify_strict_prefix_closed(outcome, prior, state):
    """A no-fallback branch must reproduce every earlier dispatched action."""
    old_periods = prior["runtime"]["continuation"]["periods"]
    prefix.require(state["strict_periods_verified"] == len(old_periods), "The complete strict prefix must match")
    if not state["fallback_seen"]:
        old_stop = prior["runtime"]["continuation"]["stop"]
        new_stop = outcome["stop"]
        old_reason = None if old_stop is None else old_stop.get("reason")
        new_reason = None if new_stop is None else new_stop.get("reason")
        compatible_reason = old_reason == new_reason or (
            old_reason == "quality_no_witness" and new_reason == "quality_witness_rejected"
        )
        prefix.require(
            len(outcome["periods"]) == len(old_periods)
            and (None if old_stop is None else old_stop.get("at_step"))
            == (None if new_stop is None else new_stop.get("at_step"))
            and compatible_reason,
            "A no-fallback run must retain the earlier stop or completed window",
        )


def feedback_callbacks(env, group, case, policy_case, receipt):
    from s2a.research.joint_control.preview.completed_buffer_gate import CompletedBufferGate

    gate = CompletedBufferGate(initial_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0)
    policy, history = builder.build_refiner(env, policy_case, receipt)
    select, observed = quality.quality_callbacks(policy, history, receipt, case["spec"]["prefix_steps"])
    if group == "replan":
        observed = replan_quality.after_quality_completed(observed, receipt, case)

    def action(q, limits, jacobian, previous_command, offset):
        return gate.select(q, limits, jacobian, previous_command, offset, select)

    def completed(actual_env, command, offset):
        feedback = observed(actual_env, command, offset)
        period = receipt["continuation_partial"]["periods"][-1]
        feedback["completed_buffer_gate"] = gate.complete(offset, [sample["q"] for sample in period["actual_inner"]])
        return feedback

    return action, completed


def summarize_phase(branches, phase, config):
    prefix.require(
        phase in ("producer", "checker")
        and [(item["case"]["case_index"], item["branch"]) for item in branches] == list(BRANCHES),
        "The registered cross-task branch order is required",
    )
    prefix.require(
        all(
            item["complete"] is item["passed"] is True and item["runtime"]["cleanup_complete"] is True
            for item in branches
        ),
        "All bounded physical workers and cleanups must close",
    )
    runtimes = [item["runtime"] for item in branches]
    counts = {
        "control_steps": sum(item["control_steps_completed"] for item in runtimes),
        "native_steps": sum(item["counters"]["mj_step"]["completed"] for item in runtimes),
        "model_initializations": sum(item["counters"]["model_construct"]["completed"] for item in runtimes),
        "contexts": sum(item["counters"]["context"]["completed"] for item in runtimes),
        "newly_selected_commands": sum(item["newly_selected_commands"] for item in runtimes),
    }
    prefix.require(
        counts["native_steps"] == 25 * counts["control_steps"]
        and counts["model_initializations"] == counts["contexts"] == 12,
        "Every native control period and model lifecycle must be accounted for",
    )
    budgets = {
        "control_steps": "max_control_steps_per_phase",
        "native_steps": "max_native_steps_per_phase",
        "model_initializations": "max_models_per_phase",
        "contexts": "max_contexts_per_phase",
        "refiner_calls": "max_refiner_calls_per_phase",
        "refiner_response_evaluations": "max_refiner_response_evaluations_per_phase",
        "quality_calls": "max_quality_calls_per_phase",
        "quality_response_evaluations": "max_quality_response_evaluations_per_phase",
        "audit_endpoint_queries": "max_audit_endpoint_queries_per_phase",
        "audit_transition_steps": "max_audit_transition_steps_per_phase",
        "start_lp_calls": "max_start_lp_per_phase",
        "shifted_rollouts": "max_shifted_rollouts_per_phase",
    }
    if phase == "checker":
        prefix.require(
            counts["newly_selected_commands"] == 0
            and all(item["candidate_modules_unloaded"] is True for item in branches),
            "The checker must only replay recorded commands",
        )
    else:
        counts.update(
            refiner_calls=sum(item["refiner_calls"]["attempted"] for item in runtimes),
            refiner_response_evaluations=sum(
                item["quality_refiner"]["baseline_response_evaluations"] for item in runtimes
            ),
            quality_calls=sum(item["quality_refiner"]["quality_calls_attempted"] for item in runtimes),
            quality_response_evaluations=sum(
                item["quality_refiner"]["quality_response_evaluations"] for item in runtimes
            ),
            audit_endpoint_queries=sum(item["quality_refiner"]["endpoint_queries_charged"] for item in runtimes),
            audit_transition_steps=sum(item["quality_refiner"]["transition_steps_charged"] for item in runtimes),
            start_lp_calls=0,
            shifted_rollouts=0,
            strict_selected=0,
            degraded_selected=0,
        )
        for runtime in runtimes:
            records = runtime["model_safe_tracking_fallback_traces"]
            strict = runtime["causal_shift_fallback_traces"]
            shared = runtime["shared_shift_traces"]
            stats = runtime["quality_refiner"]
            prefix.require(
                len(records) == len(strict) == len(shared) == stats["quality_calls_completed"],
                "One accounted strict and fallback trace per completed quality call",
            )
            for record, prior, combined in zip(records, strict, shared, strict=True):
                scope, audit = record["selected_scope"], record["audit_status"]
                prefix.require(
                    record["strict_responses"] == prior["response_evaluations"] == combined["response_evaluations"]
                    and record["extra_rollouts"] in (0, 1)
                    and record["response_evaluations"] == record["strict_responses"] + record["extra_rollouts"]
                    and record["response_evaluations"] <= config["quality_response_budget"]
                    and prior["fallback_rollouts"] in (0, 1)
                    and scope in (None, "strict", DEGRADED)
                    and audit in ("not_attempted", "passed", "rejected"),
                    "Shared response budget and candidate scope must be exact",
                )
                if scope == "strict":
                    prefix.require(
                        record["extra_rollouts"] == 0 and audit == prior["audit_status"], "Strict priority and audit"
                    )
                elif scope == DEGRADED:
                    prefix.require(
                        record["extra_rollouts"] == 1
                        and prior["fallback_rollouts"] == 1
                        and prior["fallback_safety"]["accepted"] is True
                        and prior["fallback_first_pose_caps_satisfied"] is False
                        and prior["audit_status"] == "not_attempted",
                        "The degraded candidate is distinct from a strict pose witness",
                    )
                else:
                    prefix.require(audit == "not_attempted", "A missing candidate cannot have a completed audit")
                counts["start_lp_calls"] += prior["primary_initialization"]["start_lp_attempts"]
                counts["shifted_rollouts"] += prior["fallback_rollouts"] + record["extra_rollouts"]
                counts["strict_selected"] += int(scope == "strict" and audit == "passed")
                counts["degraded_selected"] += int(scope == DEGRADED and audit == "passed")
            prefix.require(
                sum(record["response_evaluations"] for record in records) == stats["quality_response_evaluations"]
                and sum(record["audit_status"] == "passed" for record in records) == stats["witness_audits_passed"],
                "Quality responses and independent audits must match their receipts",
            )
    for key, budget in budgets.items():
        if key in counts:
            prefix.require(counts[key] <= config[budget], f"Registered {key} budget")
    return counts


def worker(phase, index, mode):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered phase required")
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen cross-task sources and inputs",
    )
    entry = inputs["cases"][index]
    group, case = entry["group"], entry["case"]
    prior = prefix.read(inputs["prior_branch_refs"][index]["producer"])
    prior_periods = prior["runtime"]["continuation"]["periods"]
    selected = replan_switch.selected_case(case, "new") if group == "replan" else case
    expected, expected_stop, producer = None, None, None
    if phase == "checker":
        require_checker_independence()
        producer = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(producer["complete"] is producer["passed"] is True, "A closed producer branch is required")
        expected = producer["runtime"]["continuation"]["periods"]
        expected_stop = producer["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "case": {"case_index": index, "group": group},
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime
        from s2a.research.native_runtime.task2_prefix_binding import IsolatedTask2PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            common.check_prefix_inner(receipt["final_prefix_inner"], prior["runtime"]["final_prefix_inner"])
            state = {"strict_periods_verified": 0, "fallback_seen": False}
            receipt["strict_prefix_comparison"] = state
            if phase == "checker":
                require_checker_independence()
                common.check_prefix_inner(receipt["final_prefix_inner"], producer["runtime"]["final_prefix_inner"])
                action = common.checker_action(expected)

                def no_history(_env, _command, _offset):
                    return {}

                completed = (
                    replan_quality.after_quality_completed(no_history, receipt, case, expected)
                    if group == "replan"
                    else no_history
                )
            else:
                policy_case = binding.policy_case(group, case)
                action, completed = feedback_callbacks(env, group, case, policy_case, receipt)

            def on_completed(actual_env, command, offset):
                feedback = completed(actual_env, command, offset)
                period = receipt["continuation_partial"]["periods"][-1]
                route = (
                    period["decision"].get("route") if expected is None else expected[offset]["decision"].get("route")
                )
                verify_strict_period(period, prior_periods, offset, route, state)
                gripper = common.measured_gripper(actual_env)
                if expected is not None:
                    checked = expected[offset]["completed_feedback"]["measured_gripper"]
                    for field in ("opening", "qpos", "qvel", "motor_target"):
                        np.testing.assert_allclose(gripper[field], checked[field], atol=1e-10, rtol=0)
                feedback["measured_gripper"] = gripper
                return feedback

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=case["spec"]["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=selected["fixed_targets"],
                tool=case["tool"],
                target_object=selected["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=on_completed,
                progress=progress,
            )
            if phase == "checker":
                q, limits, _ = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                outcome = common.reconcile_stop(
                    outcome,
                    expected_stop,
                    start_step=case["spec"]["prefix_steps"],
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
            verify_strict_prefix_closed(outcome, prior, state)
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 16 == 0 or completed >= case["spec"]["prefix_steps"]:
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "case_index": index,
                            "control_completed": completed,
                            "new_commands": receipt.get("newly_selected_commands", 0),
                            "native_steps": receipt["counters"]["mj_step"],
                        }
                    ),
                    flush=True,
                )

        run = run_prefix_runtime if group == "replan" else IsolatedTask2PrefixRuntime().run
        try:
            runtime = run(
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
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in CHECKER_FORBIDDEN)
        if phase == "checker":
            prefix.require(
                unloaded and runtime["newly_selected_commands"] == 0, "Only recorded commands may be replayed"
            )
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=unloaded)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        inputs = binding.load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "inputs": binding.input_receipt(inputs),
                    "source_refs": source_refs(),
                    "model_initializations": 0,
                    "physics_steps": 0,
                },
            )
        )
    elif args.producer or args.checker:
        phase = "producer" if args.producer else "checker"
        if phase == "checker":
            require_checker_independence()
        registered_feedback_phase.execute_phase(
            phase,
            root=ROOT,
            qa_path=QA,
            dry_path=DRY,
            branches=BRANCHES,
            worker_module="scripts.simulation.runtime.run_model_safe_tracking_fallback",
            branch_key=branch_key,
            load_inputs=binding.load_inputs,
            input_receipt=binding.input_receipt,
            source_refs=source_refs,
            summarize_phase=summarize_phase,
            config_key="config",
        )
    else:
        phase, index, mode = args.worker
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
