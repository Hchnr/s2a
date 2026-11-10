"""Compare and independently replay earlier task-6 preview feedback."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_task2_earlier_preview as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_early_quality_feedback as quality
from scripts.simulation.runtime import run_early_refiner_feedback as baseline
from scripts.simulation.runtime import run_replan_quality_feedback as full

REPO = prefix.REPO
ROOT = REPO / "outputs/task2_earlier_preview_feedback_20261111"
DRY = REPO / "outputs/task2_earlier_preview_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_earlier_preview_feedback_20261111/result.json"
BRANCHES = binding.BRANCHES
CHECKER_FORBIDDEN = common.FORBIDDEN_CHECKER_MODULES + (full.FULL_MODULE,)


def source_refs():
    return (
        binding.source_refs()
        + quality.source_refs()
        + common.source_refs()
        + full.source_refs(full=True)
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                REPO / "src/s2a/research/joint_control/preview/completed_buffer_gate.py",
                REPO / "src/s2a/research/trajectory_search/full_horizon_quality.py",
                REPO / "src/s2a/research/joint_control/tracking/full_horizon_quality_audit.py",
                REPO / "tests/research/native_runtime/test_task2_earlier_preview_runtime.py",
            )
        ]
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered earlier feedback branch")
    return f"case_{index:02d}_{mode}"


def replay_action(expected):
    """Check the checker process has not loaded any candidate feedback policy."""
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot import candidate policies"
    )
    return common.checker_action(expected)


def summarize_phase(branches, phase, config):
    if phase not in ("producer", "checker") or len(branches) != len(BRANCHES):
        raise ValueError("Exactly six registered branches in one phase required")
    identities = [(item["case"]["case_index"], item["branch"]) for item in branches]
    if identities != list(BRANCHES):
        raise ValueError("Branch identities and ordering must match the registered matrix")
    if not all(
        item["complete"] is item["passed"] is True and item["runtime"]["cleanup_complete"] is True for item in branches
    ):
        raise ValueError("Every registered runtime and cleanup must close")
    runtime = [item["runtime"] for item in branches]
    control = sum(item["control_steps_completed"] for item in runtime)
    native = sum(item["counters"]["mj_step"]["completed"] for item in runtime)
    models = sum(item["counters"]["model_construct"]["completed"] for item in runtime)
    contexts = sum(item["counters"]["context"]["completed"] for item in runtime)
    selected = sum(item["newly_selected_commands"] for item in runtime)
    if (
        control > config["max_control_steps_per_phase"]
        or native != 25 * control
        or native > config["max_native_steps_per_phase"]
        or models != config["max_models_per_phase"]
        or contexts != config["max_contexts_per_phase"]
        or (phase == "checker" and selected != 0)
    ):
        raise ValueError("Physical budget, runtime count, or replay-only checker contract exceeded")
    counts = dict(
        control_steps=control,
        native_steps=native,
        model_initializations=models,
        contexts=contexts,
        newly_selected_commands=selected,
    )
    if phase == "producer":
        values = {
            "refiner_calls": sum(item.get("refiner_calls", {}).get("attempted", 0) for item in runtime),
            "refiner_response_evaluations": sum(
                item["runtime"].get("quality_refiner", {}).get("baseline_response_evaluations", 0)
                if item["branch"] != "original"
                else item["runtime"].get("refiner_response_evaluations", 0)
                for item in branches
            ),
            "quality_calls": sum(item.get("quality_refiner", {}).get("quality_calls_attempted", 0) for item in runtime),
            "quality_response_evaluations": sum(
                item.get("quality_refiner", {}).get("quality_response_evaluations", 0) for item in runtime
            ),
            "audit_endpoint_queries": sum(
                item.get("quality_refiner", {}).get("endpoint_queries_charged", 0) for item in runtime
            ),
            "audit_transition_steps": sum(
                item.get("quality_refiner", {}).get("transition_steps_charged", 0) for item in runtime
            ),
        }
        budgets = {
            "refiner_calls": "max_refiner_calls_per_phase",
            "refiner_response_evaluations": "max_refiner_response_evaluations_per_phase",
            "quality_calls": "max_quality_calls_per_phase",
            "quality_response_evaluations": "max_quality_response_evaluations_per_phase",
            "audit_endpoint_queries": "max_audit_endpoint_queries_per_phase",
            "audit_transition_steps": "max_audit_transition_steps_per_phase",
        }
        if any(value > config[budgets[key]] for key, value in values.items()):
            raise ValueError("Registered feedback solver or audit budget exceeded")
        counts.update(values)
    return counts


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    phase_dir, directory = ROOT / phase, ROOT / phase / key
    registration = prefix.read(prefix.reference(phase_dir / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen earlier feedback inputs",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    policy_case = binding.policy_case(case) if phase == "producer" else None
    expected, expected_stop, producer_branch = None, None, None
    if phase == "checker":
        producer_branch = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(
            producer_branch["complete"] is producer_branch["passed"] is True, "Closed corresponding producer branch"
        )
        expected = producer_branch["runtime"]["continuation"]["periods"]
        expected_stop = producer_branch["runtime"]["continuation"]["stop"]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case=case["intervention_spec"],
        branch=mode,
        automatic_retry=False,
    )
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.task2_prefix_binding import IsolatedTask2PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if phase == "checker":
                common.check_prefix_inner(
                    receipt["final_prefix_inner"], producer_branch["runtime"]["final_prefix_inner"]
                )
                action, completed = replay_action(expected), None
            else:
                from s2a.research.joint_control.preview.completed_buffer_gate import CompletedBufferGate

                gate = CompletedBufferGate(initial_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0)
                if mode == "original":
                    policy = baseline.make_refiner(env, policy_case, receipt)
                    choose, observe = policy.select, policy.observe
                else:
                    if mode == "first_quality":
                        policy, history = quality.make_quality_refiner(env, policy_case, receipt)
                    else:
                        policy, history = full.make_full_refiner(env, policy_case, receipt)
                    choose, observe = quality.quality_callbacks(policy, history, receipt, case["spec"]["prefix_steps"])

                def action(q, limits, jacobian, previous, offset):
                    return gate.select(q, limits, jacobian, previous, offset, choose)

                def completed(actual_env, command, offset):
                    observed = observe(actual_env, command, offset)
                    period = receipt["continuation_partial"]["periods"][-1]
                    observed["completed_buffer_gate"] = gate.complete(
                        offset, [sample["q"] for sample in period["actual_inner"]]
                    )
                    return observed

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=case["spec"]["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=case["fixed_targets"],
                tool=case["tool"],
                target_object=case["intervention_spec"]["target"],
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
                    start_step=case["spec"]["prefix_steps"],
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 16 == 0 or completed >= case["spec"]["prefix_steps"]:
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
            runtime = IsolatedTask2PrefixRuntime().run(
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
        if phase == "checker":
            result["candidate_modules_unloaded"] = all(name not in sys.modules for name in CHECKER_FORBIDDEN)
            prefix.require(result["candidate_modules_unloaded"], "Replay never loaded candidate feedback")
        result.update(complete=True, passed=True, runtime=runtime)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute(phase):
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU QA",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == binding.input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry identity",
    )
    producer_refs = None
    if phase == "checker":
        producer = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 6,
            "All producer branches closed",
        )
        producer_refs = {
            branch_key(*branch): ref for branch, ref in zip(BRANCHES, producer["branch_refs"], strict=True)
        }
    phase_dir = ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    registration = publish(
        phase_dir / "registration.json",
        dict(
            phase=phase, inputs=binding.input_receipt(inputs), source_refs=sources, producer_branch_refs=producer_refs
        ),
    )
    specs = [
        dict(
            key=branch_key(index, mode),
            argv=[
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_task2_earlier_preview_feedback",
                "--worker",
                phase,
                str(index),
                mode,
            ],
            log=phase_dir / branch_key(index, mode) / "worker.log",
        )
        for index, mode in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["feedback_config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(phase_dir / "workers.json", lifecycle)
    available = [
        prefix.reference(phase_dir / branch_key(*branch) / "result.json")
        for branch in BRANCHES
        if (phase_dir / branch_key(*branch) / "result.json").is_file()
    ]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        registration_ref=registration,
        workers_ref=workers,
        available_branch_refs=available,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 6
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "All feedback workers exited and were reaped",
        )
        branch_refs = [prefix.reference(phase_dir / branch_key(*branch) / "result.json") for branch in BRANCHES]
        branches = [prefix.read(ref) for ref in branch_refs]
        counts = summarize_phase(branches, phase, inputs["feedback_config"])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before,
            "Earlier feedback source identity unchanged",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_refs=branch_refs, counts=counts)
    finally:
        print(publish(phase_dir / "result.json", result), flush=True)


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
        prefix.require(phase in ("producer", "checker"), "One known phase required")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
