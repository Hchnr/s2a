"""Compare shared-budget feedback on two independently replayed replan states."""

import argparse
import json
import signal
import sys
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.simulation.runtime import build_causal_quality_feedback as builder
from scripts.simulation.runtime import prepare_replan_shared_shift as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_early_quality_feedback as quality
from scripts.simulation.runtime import run_replan_quality_feedback as previous
from scripts.simulation.runtime import run_replan_reference_switch as switch
from scripts.simulation.runtime import run_task2_shared_shift_feedback as task2

REPO = prefix.REPO
ROOT = REPO / "outputs/replan_shared_shift_feedback_20261111"
DRY = REPO / "outputs/replan_shared_shift_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/replan_shared_shift_feedback_20261111/result.json"
BRANCHES = ((2, "shared_shift_quality_new"), (6, "shared_shift_quality_new"))
CHECKER_FORBIDDEN = tuple(dict.fromkeys(task2.CHECKER_FORBIDDEN + (previous.RESTORE_MODULE,)))


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "tests/research/native_runtime/test_replan_shared_shift_feedback.py",
        )
    ]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered cross-task shared-shift branch")
    return f"case_{index:02d}_{mode}"


def make_shared_refiner(env, case, receipt):
    """Attach the same audited candidate to this branch's completed history."""
    from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit

    adapter = SharedShiftFallbackAudit(receipt)
    policy, history = builder.build_refiner(env, binding.policy_case(case), receipt, lambda _receipt: adapter)
    adapter.attach_history(policy.tracking.history)
    return policy, history


def feedback_callbacks(env, case, receipt):
    """Gate dispatch on actual inner samples and score only completed periods."""
    from s2a.research.joint_control.preview.completed_buffer_gate import CompletedBufferGate

    gate = CompletedBufferGate(initial_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0)
    policy, history = make_shared_refiner(env, case, receipt)
    select, quality_done = quality.quality_callbacks(policy, history, receipt, 16)
    score_completed = previous.after_quality_completed(quality_done, receipt, case)

    def action(q, limits, jacobian, previous_command, offset):
        return gate.select(q, limits, jacobian, previous_command, offset, select)

    def completed(actual_env, command, offset):
        score = score_completed(actual_env, command, offset)
        period = receipt["continuation_partial"]["periods"][-1]
        score["completed_buffer_gate"] = gate.complete(offset, [sample["q"] for sample in period["actual_inner"]])
        return score

    return action, completed


def replay_action(expected):
    prefix.require(all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot import candidates")
    return common.checker_action(expected)


def summarize_phase(branches, phase, config):
    """Apply the established shared-budget accounting to these two task IDs."""
    namespace = task2.summarize_phase.__globals__.copy()
    namespace["BRANCHES"] = BRANCHES
    old = task2.summarize_phase
    summarize = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    summarize.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return summarize(branches, phase, config)


def worker(phase, index, mode):
    """Replay one original prefix and execute or independently replay its branch."""

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
        "Frozen cross-task feedback inputs and sources",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    selected = switch.selected_case(case, "new")
    expected, expected_stop, producer = None, None, None
    if phase == "checker":
        producer = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(producer["complete"] is producer["passed"] is True, "Closed producer branch required")
        expected = producer["runtime"]["continuation"]["periods"]
        expected_stop = producer["runtime"]["continuation"]["stop"]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case={"case_index": index},
        branch=mode,
        automatic_retry=False,
    )
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            receipt["preview_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_response_evaluations"] = 0
            if phase == "checker":
                common.check_prefix_inner(receipt["final_prefix_inner"], producer["runtime"]["final_prefix_inner"])
                action = replay_action(expected)

                def no_candidate_history(_env, _command, _offset):
                    return None

                completed = previous.after_quality_completed(no_candidate_history, receipt, case, expected)
            else:
                action, completed = feedback_callbacks(env, case, receipt)
            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=16,
                previous_command=case["commands"][-1],
                targets=selected["fixed_targets"],
                tool=case["tool"],
                target_object=selected["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
            )
            if phase == "checker":
                q, limits, _ = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                return common.reconcile_stop(
                    outcome,
                    expected_stop,
                    start_step=16,
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 16 == 0 or completed >= 16:
                print(
                    json.dumps(
                        dict(
                            phase=phase,
                            case_index=index,
                            branch=mode,
                            control_completed=completed,
                            new_commands=receipt.get("newly_selected_commands", 0),
                            native_steps=receipt["counters"]["mj_step"],
                        ),
                        allow_nan=False,
                    ),
                    flush=True,
                )

        try:
            runtime = run_prefix_runtime(
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
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in CHECKER_FORBIDDEN)
        if phase == "checker":
            prefix.require(
                unloaded
                and runtime["newly_selected_commands"] == 0
                and runtime["preview_calls"]["attempted"] == runtime["refiner_calls"]["attempted"] == 0,
                "Checker only replays recorded actions and scores completed states",
            )
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=unloaded)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_replan_shared_shift_feedback",
        branch_key=branch_key,
        load_inputs=binding.load_inputs,
        input_receipt=binding.input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="replan_shared_config",
    )


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
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    else:
        phase, index, mode = args.worker
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
