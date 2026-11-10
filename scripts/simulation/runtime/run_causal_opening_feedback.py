"""Execute and independently replay screened opening-event feedback branches."""

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
from scripts.simulation.runtime import prepare_causal_opening_feedback as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as base

REPO = prefix.REPO
ROOT = REPO / "outputs/causal_opening_event_feedback_20261111"
DRY = REPO / "outputs/causal_opening_event_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/causal_opening_event_feedback_20261111/result.json"


def source_refs():
    return (
        binding.source_refs()
        + base.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                REPO / "tests/research/native_runtime/test_causal_opening_feedback_runtime.py",
            )
        ]
    )


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "cases": [case["spec"] for case in inputs["cases"]],
        "branches": [list(item) for item in inputs["branches"]],
        "admitted_cases": inputs["admitted_cases"],
        "rejected_cases": inputs["rejected_cases"],
        "phase_budget": inputs["phase_budget"],
        "asset_refs": inputs["asset_refs"],
        "source_sha256": inputs["source_sha256"],
    }


def branch_key(index, mode):
    if type(index) is not int or mode not in ("latest", "remembered"):
        raise ValueError("One explicit recorded case and paired-reference branch required")
    return f"case_{index:02d}_{mode}"


def reconstruct_gripper(case, result):
    """Replay completed actual history, checking every original gripper decision."""
    from s2a.research.gripper.causal_loaded_history import reconstruct_loaded_history
    from s2a.research.gripper.loaded_gripper_transaction import LoadedGripperTransaction
    from scripts.simulation.runtime.reconstruct_causal_gripper_history import read_chunk_prefix

    n = case["spec"]["prefix_steps"]
    refs = case["gripper_inputs"]
    old_chunks = read_chunk_prefix(prefix.checked(refs["chunks_ref"]), n // 8)
    parameters = prefix.read(refs["parameters_ref"])
    controller, history = reconstruct_loaded_history(
        case["rows"],
        old_chunks,
        case["tool"],
        parameters,
        velocity_envelope=case["controller"]["gripper_velocity_envelope"],
        gripper_config=case["controller"]["gripper_config"],
        progress=lambda progress: result.update(gripper_reconstruction_progress=progress),
    )
    prefix.require(
        history["steps"] == n
        and history["last_completed_command"] == float(case["commands"][-1, 6])
        and history["candidate_sequences"] == history["planner_calls"] * 3**8,
        "Every historical gripper command and planner cost must match",
    )
    result["gripper_history"] = history
    return LoadedGripperTransaction(controller)


def summarize_phase(branches, phase, budget):
    if phase not in ("producer", "checker") or not branches:
        raise ValueError("One nonempty producer or checker phase required")
    if not all(
        item["complete"] is item["passed"] is True and item["runtime"]["cleanup_complete"] is True for item in branches
    ):
        raise ValueError("Every branch must complete its bounded runtime and cleanup")
    runtime = [item["runtime"] for item in branches]
    control = sum(item["control_steps_completed"] for item in runtime)
    native = sum(item["counters"]["mj_step"]["completed"] for item in runtime)
    models = sum(item["counters"]["model_construct"]["completed"] for item in runtime)
    contexts = sum(item["counters"]["context"]["completed"] for item in runtime)
    selected = sum(item["newly_selected_commands"] for item in runtime)
    if (
        control > budget["control_steps"]
        or native != 25 * control
        or native > budget["native_steps"]
        or models != budget["model_initializations"]
        or contexts != budget["contexts"]
        or (phase == "checker" and selected != 0)
    ):
        raise ValueError("Registered physical cost or replay-only checker contract exceeded")
    counts = dict(
        control_steps=control,
        native_steps=native,
        model_initializations=models,
        contexts=contexts,
        newly_selected_commands=selected,
    )
    if phase == "producer":
        histories = [item["gripper_history"] for item in branches]
        proposals = [item.get("gripper_proposals", {}) for item in runtime]
        values = {
            "gripper_planner_calls": sum(item["planner_calls"] for item in histories)
            + sum(item.get("planner_calls", 0) for item in proposals),
            "gripper_candidate_sequences": sum(item["candidate_sequences"] for item in histories)
            + sum(item.get("candidate_sequences", 0) for item in proposals),
            "refiner_calls": sum(item.get("refiner_calls", {}).get("attempted", 0) for item in runtime),
            "refiner_response_evaluations": sum(
                item.get("quality_refiner", {}).get("baseline_response_evaluations", 0) for item in runtime
            ),
            "quality_calls": sum(item.get("quality_refiner", {}).get("quality_calls_attempted", 0) for item in runtime),
            "quality_response_evaluations": sum(
                item.get("quality_refiner", {}).get("quality_response_evaluations", 0) for item in runtime
            ),
        }
        if any(value > budget[key] for key, value in values.items()):
            raise ValueError("Registered planner or solver budget exceeded")
        counts.update(values)
    return counts


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        (index, mode) in inputs["branches"]
        and input_receipt(inputs) == registration["inputs"]
        and source_refs() == registration["source_refs"],
        "Frozen screened branch inputs and source identity",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    selected = binding.selected_case(case, mode)
    original, expected, expected_stop = None, None, None
    if phase == "checker":
        original = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
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
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            policy = None
            if phase == "checker":
                base.check_prefix_inner(receipt["final_prefix_inner"], original["runtime"]["final_prefix_inner"])
                action = base.checker_action(expected)
            else:
                policy = base.make_feedback(env, case, selected, gripper, receipt)
                action = policy.select

            def observed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == case["spec"]["prefix_steps"] + offset, "Completed physical endpoint")
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [item["q"] for item in period["actual_inner"]]
                    )
                expected_score = None if expected is None else expected[offset]["completed_feedback"]["scores"]
                scores = base.score_completed(
                    actual_env, case, selected, mode, offset, period["tcp_error"], expected=expected_score
                )
                return dict(scores=scores, policy_history=history, policy_history_recomputed=policy is not None)

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=case["spec"]["prefix_steps"],
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
                    expected_stop,
                    start_step=case["spec"]["prefix_steps"],
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 32 == 0 or completed >= case["spec"]["prefix_steps"]:
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
                registered_seed=43,
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
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
        "Frozen CPU feedback QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA and feedback source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves screened cases, budgets and sources",
    )
    producer_refs = None
    if phase == "checker":
        producer = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == len(inputs["branches"]),
            "Every screened producer branch closed",
        )
        producer_refs = {
            branch_key(*branch): ref for branch, ref in zip(inputs["branches"], producer["branch_refs"], strict=True)
        }
    phase_dir = ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    registration = publish(
        phase_dir / "registration.json",
        dict(phase=phase, inputs=input_receipt(inputs), source_refs=sources, producer_branch_refs=producer_refs),
    )
    specs = [
        dict(
            key=branch_key(index, mode),
            argv=[
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_causal_opening_feedback",
                "--worker",
                phase,
                str(index),
                mode,
            ],
            log=phase_dir / branch_key(index, mode) / "worker.log",
        )
        for index, mode in inputs["branches"]
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(phase_dir / "workers.json", lifecycle)
    available = [
        prefix.reference(phase_dir / branch_key(*branch) / "result.json")
        for branch in inputs["branches"]
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
        admitted_cases=inputs["admitted_cases"],
        rejected_cases=inputs["rejected_cases"],
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == len(inputs["branches"])
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "All screened phase workers exited and were reaped",
        )
        branch_refs = [
            prefix.reference(phase_dir / branch_key(*branch) / "result.json") for branch in inputs["branches"]
        ]
        branches = [prefix.read(ref) for ref in branch_refs]
        counts = summarize_phase(branches, phase, inputs["phase_budget"])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Physical phase sources unchanged"
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
                    inputs=input_receipt(inputs),
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
        prefix.require(phase in ("producer", "checker"), "Known phase required")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
