"""Own a bounded old/new reference comparison and its isolated replays."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import bounded_reference_choice as runtime
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import replay_multitask_wrench as admission
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/bounded_reference_choice_recovery_20261115"
PLAN = ROOT / "plans/v0.0.1/tracking/references/bounded_reference_choice_20261113.md"
RECOVERY_PLAN = ROOT / "plans/v0.0.1/tracking/references/bounded_reference_choice_recovery_20261115.md"
QA = ROOT / "outputs/qa/bounded_reference_choice_recovery_20261115/result.json"
DRY = OUTPUT / "dry.json"
MODULE = "scripts.simulation.runtime.run_bounded_reference_choice"
MODES = ("old", "new")


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            Path(runtime.__file__).resolve(),
            PLAN,
            RECOVERY_PLAN,
            ROOT / "src/s2a/research/native_runtime/reference_choice_comparison.py",
            ROOT / "src/s2a/research/native_runtime/finite_reference_tail.py",
            ROOT / "src/s2a/research/native_runtime/reset_bounded_runtime.py",
            ROOT / "tests/research/native_runtime/test_reference_choice_comparison.py",
            ROOT / "tests/research/native_runtime/test_bounded_reference_choice.py",
        )
    ]


def progress_event(phase, mode, receipt):
    """Record only counters already exposed by the running physical runtime."""
    if phase not in ("producer", "checker") or mode not in MODES or type(receipt) is not dict:
        raise ValueError("A registered phase, mode and runtime receipt are required")
    counters = receipt.get("counters", {})
    return dict(
        event="physical_progress",
        pid=os.getpid(),
        phase=phase,
        mode=mode,
        control_steps_attempted=receipt.get("control_steps_attempted"),
        control_steps_completed=receipt.get("control_steps_completed"),
        newly_selected_commands=receipt.get("newly_selected_commands"),
        native_steps=copy.deepcopy(counters.get("mj_step")),
        models=copy.deepcopy(counters.get("model_construct")),
        contexts=copy.deepcopy(counters.get("context")),
    )


def prepare():
    """Admit the original two references and expose only the completed prefix."""
    inputs, producer, model, _bundle, row, registration, replacements = admission.admit(2)
    sequence = producer["runtime"]["continuation"]
    segments = sequence["adaptive"]["segments"]
    first, second = sequence["first_window"]["periods"], segments[0]["continuation"]["periods"]
    captures = producer["runtime"]["adaptive_captures"][:2]
    exchanges = producer["runtime"]["adaptive_model_exchanges"][:2]
    responses = model["responses"][:2]
    prefix.require(
        len(first) == len(second) == 8
        and segments[0]["start_step"] == 8
        and segments[0]["boundary"] == "window_end"
        and segments[1]["start_step"] == 16
        and len(captures) == len(exchanges) == len(responses) == 2,
        "The selected original case must contain the full 16-period prefix and both observations",
    )
    history = copy.deepcopy(first + second)
    prefix.require([row["step"] for row in history] == list(range(16)), "The exact completed history is required")
    decoded = [
        runtime.saved_reference(capture, exchange, response, inputs["case"]["tool"], step)
        for capture, exchange, response, step in zip(captures, exchanges, responses, (8, 16), strict=True)
    ]
    old_openings = decoded[0][1]
    closes = np.flatnonzero(old_openings[8:] <= 0.2)
    prefix.require(len(closes) and int(closes[0] + 1) == 28, "The registered old strong-close horizon differs")
    refs = [admission.SOURCE, row["producer_ref"], row["model_ref"], row["original_owner_ref"]]
    for exchange, response in zip(exchanges, responses, strict=True):
        refs.extend((exchange["request_ref"], exchange["response_ref"], response["actual_response_ref"]))
    for ref in refs:
        prefix.checked(ref)
    identity = dict(
        case_index=2,
        seed=43,
        inputs=admission.admission.input_receipt(inputs),
        recorded_refs=refs,
        original_registration_refs=registration["source_refs"],
        documented_source_replacements=replacements,
    )
    return dict(
        inputs=inputs,
        history=history,
        captures=copy.deepcopy(captures),
        exchanges=copy.deepcopy(exchanges),
        responses=copy.deepcopy(responses),
        identity=identity,
    )


def dry_run():
    from s2a.research.joint_control.preview.causal_reference_replan import completed_quality_margin
    from s2a.research.native_runtime.reference_choice_comparison import quality_stop

    prefix.environment()
    data = prepare()
    prefix.require(
        quality_stop(data["history"][-1], 16, completed_quality_margin) is None,
        "Initial quality must allow continuation",
    )
    result = dict(
        complete=True,
        passed=True,
        source_refs=source_refs(),
        identity=data["identity"],
        native_models=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish_result(DRY, result), flush=True)


def worker(phase, mode):
    prefix.require(phase in ("producer", "checker") and mode in MODES, "A fixed phase and branch are required")
    print(json.dumps(dict(event="worker_started", pid=os.getpid(), phase=phase, mode=mode)), flush=True)
    directory = OUTPUT / phase / mode / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        data = prepare()
        prefix.require(
            registration["phase"] == phase
            and registration["identity"] == data["identity"]
            and registration["source_refs"] == source_refs(),
            "The frozen worker inputs and source identity are required",
        )
        saved = None
        if phase == "checker":
            isolation.checker_isolation()
            saved = prefix.read(registration["producer_branch_refs"][mode])
            prefix.require(saved["complete"] is saved["passed"] is True, "A completed producer branch is required")
            saved = saved["runtime"]["continuation"]
        cv2 = data["inputs"]["base"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime

        inputs, case = data["inputs"], data["inputs"]["case"]
        adapter = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            return runtime.run_comparison(
                env,
                terminal,
                observers,
                receipt,
                phase=phase,
                mode=mode,
                case=case,
                history=data["history"],
                original_captures=data["captures"],
                exchanges=data["exchanges"],
                responses=data["responses"],
                runtime_output=directory / "runtime",
                producer=saved,
                progress=progress,
            )

        def progress(receipt):
            print(json.dumps(progress_event(phase, mode, receipt)), flush=True)

        actual = adapter.run_feedback(
            case["config"],
            case["record"],
            case["record_xml"],
            directory / "runtime",
            inputs["base"]["asset_refs"],
            registered_seed=43,
            continuation=continuation,
            max_control_steps=48,
            progress=progress,
        )
        if phase == "checker":
            isolation.checker_isolation()
            prefix.require(actual["newly_selected_commands"] == 0, "The checker may only replay recorded actions")
        prefix.require(
            actual["control_steps_completed"] <= 48 and actual["world_model_calls"] == 0, "Bounded physical work"
        )
        return dict(
            complete=True, passed=True, runtime=actual, case=case["case"], cv2_admission=cv2_receipt, fresh_cases=0
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def tail_periods(branch):
    tail = branch["runtime"]["continuation"]["tail"]
    return [period for segment in tail["segments"] for period in segment["continuation"]["periods"]]


def validate_branch(branch, phase):
    prefix.require(
        branch["complete"] is branch["passed"] is True and branch["phase"] == phase, "Completed branch required"
    )
    actual = branch["runtime"]
    continuation = actual["continuation"]
    history = continuation["first_window"]["periods"] + continuation["second_window"]["periods"]
    tail = tail_periods(branch)
    prefix.require(
        len(history) == 16
        and 0 <= len(tail) <= 32
        and actual["control_steps_attempted"] == actual["control_steps_completed"] == 16 + len(tail)
        and actual["counters"]["mj_step"]["completed"] == 25 * (16 + len(tail))
        and actual["counters"]["model_construct"]["completed"] == actual["counters"]["context"]["completed"] == 3
        and actual["world_model_calls"] == 0
        and branch["fresh_cases"] == 0,
        "The physical and native resource bounds must match the actual periods",
    )
    prefix.require(
        actual["newly_selected_commands"] == (16 + len(tail) if phase == "producer" else 0),
        "The checker cannot select actions and the producer must account for every decision",
    )
    return dict(control_steps=16 + len(tail), native_steps=25 * (16 + len(tail)), tail_steps=len(tail))


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One phase required")
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "The CPU regression and zero-physics dry run must pass",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot must match")
    data, sources = prepare(), source_refs()
    prefix.require(dry["source_refs"] == sources and dry["identity"] == data["identity"], "Dry identities must match")
    directory = OUTPUT / phase
    prefix.require(not directory.exists(), "Each physical phase must use a new output directory")
    producer_refs = {}
    if phase == "checker":
        previous = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(previous["complete"] is previous["passed"] is True, "The producer group must finish first")
        producer_refs = previous["branch_refs"]
        for ref in producer_refs.values():
            prefix.checked(ref)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            owner_pid=os.getpid(),
            identity=data["identity"],
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            producer_branch_refs=producer_refs,
            max_control_steps=96,
            max_native_steps=2400,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(
            key=mode, argv=[sys.executable, "-m", MODULE, "--worker", phase, mode], log=directory / mode / "worker.log"
        )
        for mode in MODES
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=3600)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        registration_ref=registration_ref,
        lifecycle=lifecycle,
        branch_refs={},
        counts={},
        automatic_retry=False,
    )
    try:
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 2
            and all(row["returncode"] == 0 and row["actual_wait"] and row["reaped"] for row in lifecycle["workers"]),
            "Both owned workers must exit and be reaped successfully",
        )
        for mode in MODES:
            ref = prefix.reference(directory / mode / "worker/result.json")
            branch = prefix.read(ref)
            result["branch_refs"][mode], result["counts"][mode] = ref, validate_branch(branch, phase)
            if phase == "checker":
                producer = prefix.read(producer_refs[mode])
                result["counts"][mode]["replay_max_difference"] = comparison.compare_periods(
                    tail_periods(producer), tail_periods(branch)
                )
                prefix.require(
                    branch["runtime"]["continuation"]["tail"]["stop"]
                    == producer["runtime"]["continuation"]["tail"]["stop"],
                    "The checker must retain the producer's reported stop",
                )
        prefix.require(
            snapshot(ROOT)["sources"] == before and prepare()["identity"] == data["identity"],
            "Inputs and sources must remain unchanged",
        )
        prefix.require(sum(row["control_steps"] for row in result["counts"].values()) <= 96, "Group physical budget")
        result.update(complete=True, passed=True, sources_unchanged=True, world_model_calls=0, fresh_cases=0)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--phase", choices=("producer", "checker"))
    parser.add_argument("--worker", nargs=2, metavar=("PHASE", "MODE"))
    args = parser.parse_args()
    if sum((args.dry, args.phase is not None, args.worker is not None)) != 1:
        parser.error("Choose exactly one dry run, phase or worker")
    if args.dry:
        dry_run()
    elif args.phase:
        phase_run(args.phase)
    else:
        worker(*args.worker)


if __name__ == "__main__":
    main()
