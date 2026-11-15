"""Diagnose the saved wine/seed42 capture boundary without advancing beyond it."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.recorded_capture_diagnostic import (
    capture_differences,
    complete_windows_before,
    replay_until_capture,
)
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import replay_multitask_wrench as admission
from scripts.simulation.runtime import reset_live_two_window_replay as captures
from scripts.simulation.runtime import run_multitask_margin_recheck as previous
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/wine_capture_identity_diagnostic_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/wine_capture_identity_diagnostic_20261116.md"
OUTPUT = ROOT / "outputs/wine_capture_identity_diagnostic_20261116"
DRY = OUTPUT / "dry.json"
QA = ROOT / "outputs/qa/wine_capture_identity_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.diagnose_wine_capture_identity"


def validate_config(config):
    fixed = dict(
        schema="s2a.wine-capture-identity-diagnostic.v1",
        output="outputs/wine_capture_identity_diagnostic_20261116",
        failed_worker=dict(
            path="outputs/multitask_margin_recheck_20261116/producer/case04/worker/result.json",
            sha256="66c786de10e3105988ae545e09ffff0c2e71ff07594bc387106600c50dc5676d",
        ),
        case_index=4,
        sampling_seed=42,
        capture_step=104,
        phases=["producer", "checker"],
        max_control_steps_per_phase=104,
        max_native_steps_per_phase=2600,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
        max_workers=2,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if (
        not isinstance(config, dict)
        or set(config) != set(fixed)
        or any(config[key] != value or type(config[key]) is not type(value) for key, value in fixed.items())
    ):
        raise ValueError("The fixed capture, two phases and explicit physical budgets are required")


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/recorded_capture_diagnostic.py",
            ROOT / "tests/research/native_runtime/test_recorded_capture_diagnostic.py",
            ROOT / "tests/research/native_runtime/test_wine_capture_identity_diagnostic.py",
        )
    ]


def prepare():
    """Admit only the references consumed before the registered capture."""
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    failed_ref = dict(path=str(ROOT / config["failed_worker"]["path"]), sha256=config["failed_worker"]["sha256"])
    failed = prefix.read(failed_ref)
    partial = failed["runtime"]
    prefix.require(
        failed["complete"] is failed["passed"] is False
        and failed["error"]["type"] == "PrefixReplayFailure"
        and failed["error"]["message"] == "The independent current image or endpoint differs"
        and partial["control_steps_attempted"] == partial["control_steps_completed"] == config["capture_step"]
        and partial["counters"]["mj_step"]["completed"] == config["max_native_steps_per_phase"]
        and partial["cleanup_complete"] is True,
        "The original failed attempt and its physical cost must remain intact",
    )
    inputs, producer, model, _bundle, row, registration, changes = admission.admit(config["case_index"])
    sequence, runtime = producer["runtime"]["continuation"], producer["runtime"]
    initial = copy.deepcopy(sequence["first_window"]["periods"])
    windows = complete_windows_before(sequence["adaptive"]["segments"], capture_step=config["capture_step"])
    count = len(windows)
    current_captures = copy.deepcopy(runtime["adaptive_captures"][: count + 1])
    exchanges = copy.deepcopy(runtime["adaptive_model_exchanges"][:count])
    responses = copy.deepcopy(model["responses"][:count])
    expected_steps = list(range(8, config["capture_step"], 8))
    prefix.require(
        row["case_index"] == config["case_index"]
        and inputs["case"]["case"]["sampling_seed"] == config["sampling_seed"]
        and len(initial) == 8
        and len(current_captures) == count + 1
        and len(exchanges) == len(responses) == count
        and [item["start_step"] for item in windows] == expected_steps
        and [item["frame_index"] for item in current_captures] == expected_steps + [config["capture_step"]]
        and [item["step"] for item in exchanges] == [item["step"] for item in responses] == expected_steps,
        "Only the pre-capture references and the original target image may be admitted",
    )
    comparison.compare_periods(initial, inputs["baseline_periods"])
    refs = [admission.SOURCE, row["producer_ref"], row["model_ref"], row["checker_ref"], row["original_owner_ref"]]
    for exchange, response in zip(exchanges, responses, strict=True):
        refs.extend((exchange["request_ref"], exchange["response_ref"], response["actual_response_ref"]))
    for ref in refs:
        prefix.checked(ref)
    identity = dict(
        config_ref=config_ref,
        failed_worker_ref=failed_ref,
        inputs=admission.admission.input_receipt(inputs),
        source_refs=refs,
        original_registration_refs=registration["source_refs"],
        documented_source_replacements=changes,
        consumed_reference_steps=expected_steps,
        capture_step=config["capture_step"],
    )
    return dict(
        config=config,
        inputs=inputs,
        initial=initial,
        windows=windows,
        captures=current_captures,
        exchanges=exchanges,
        responses=responses,
        identity=identity,
    )


def dry_run():
    prefix.environment()
    original.require_no_simulator_import()
    data = prepare()
    previous.checker_isolation()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                identity=data["identity"],
                source_refs=source_refs(),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def persist_target_capture(path, expected, actual):
    """Persist both captures before invoking any comparison that may fail."""
    capture_ref = publish_result(path, dict(expected=expected, actual=actual))
    fields = capture_differences(expected, actual)
    error = None
    try:
        captures.compare_capture(expected, actual)
    except (ValueError, AssertionError) as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    prefix.require(
        fields["passed"] == (error is None), "Field diagnostics must retain the original capture guard's outcome"
    )
    return dict(capture_ref=capture_ref, fields=fields, original_guard_passed=error is None, original_guard_error=error)


def worker(phase):
    prefix.require(phase in ("producer", "checker"), "One registered phase is required")
    directory = OUTPUT / phase / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    print(json.dumps(dict(event="worker_started", phase=phase, pid=os.getpid())), flush=True)

    def execute():
        data = prepare()
        prefix.require(
            phase in registration["phases"]
            and registration["identity"] == data["identity"]
            and registration["source_refs"] == source_refs(),
            "The registered case and source identities must match",
        )
        if phase == "checker":
            previous.checker_isolation()
        inputs, case, config = data["inputs"], data["inputs"]["case"], data["config"]
        cv2 = inputs["base"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime

        adapter = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        observed = []

        def render(actual_env, current, step):
            sample, endpoint = current["sample"], current["endpoint_receipt"]
            identity = dict(
                endpoint_step=step, sim_time=endpoint["sim_time"], sim_state_sha256=endpoint["sim_state_sha256"]
            )
            return render_with_two_torch_threads(
                observers.render_anchor, {}, actual_env, step, sample, identity, directory / "runtime"
            )

        observers = None

        def continuation(env, terminal, active_observers, receipt, *, progress=None):
            nonlocal observers
            observers = active_observers

            def load_window(actual_env, current, position, step):
                actual = render(actual_env, current, step)
                captures.compare_capture(data["captures"][position], actual)
                observed.append(actual)
                return previous.saved_reference(
                    actual,
                    data["exchanges"][position],
                    data["responses"][position],
                    case["tool"],
                    step=step,
                    seed=config["sampling_seed"],
                )

            def capture_final(actual_env, current, step):
                prefix.require(step == config["capture_step"], "Only the fixed target capture is allowed")
                actual = render(actual_env, current, step)
                return persist_target_capture(directory / "capture104.json", data["captures"][-1], actual)

            return replay_until_capture(
                env,
                terminal,
                active_observers,
                receipt,
                phase=phase,
                case=case,
                initial_periods=data["initial"],
                windows=data["windows"],
                capture_step=config["capture_step"],
                load_window=load_window,
                capture_final=capture_final,
                score_gripper=original.measured_gripper,
                compare_score=original.compare_score,
                progress=progress,
            )

        last_reported = -1

        def progress(receipt):
            nonlocal last_reported
            steps = receipt.get("control_steps_completed", 0)
            if steps != last_reported and steps % 8 == 0:
                print(json.dumps(dict(event="physical_progress", phase=phase, control_steps=steps)), flush=True)
                last_reported = steps

        actual = adapter.run_feedback(
            case["config"],
            case["record"],
            case["record_xml"],
            directory / "runtime",
            inputs["base"]["asset_refs"],
            registered_seed=config["sampling_seed"],
            continuation=continuation,
            max_control_steps=config["max_control_steps_per_phase"],
            progress=progress,
        )
        if phase == "checker":
            previous.checker_isolation()
        prefix.require(len(observed) == len(data["windows"]), "All consumed reference images must match")
        return dict(
            complete=True,
            passed=True,
            case=case["case"],
            runtime=actual,
            reference_captures=observed,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=phase == "checker",
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def periods(branch):
    continuation = branch["runtime"]["continuation"]
    return continuation["initial"]["periods"] + [
        item for window in continuation["windows"] for item in window["continuation"]["periods"]
    ]


def validate_branch(branch, phase, config):
    runtime = branch["runtime"]
    trace = runtime["continuation"]
    rows = periods(branch)
    target = trace["capture"]
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and branch["phase"] == phase
        and branch["case"]["case_index"] == config["case_index"]
        and branch["case"]["sampling_seed"] == config["sampling_seed"]
        and runtime["reset_observation_verified"] is True
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == len(rows) == 104
        and runtime["counters"]["mj_step"]["attempted"]
        == runtime["counters"]["mj_step"]["completed"]
        == config["max_native_steps_per_phase"]
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["cleanup_complete"] is True
        and runtime["world_model_calls"] == branch["fresh_cases"] == 0
        and trace["capture_step"] == 104
        and trace["stop"] == "captured_before_next_action"
        and trace["next_action_dispatched"] is False
        and len(branch["reference_captures"]) == 12
        and target["original_guard_passed"] == target["fields"]["passed"]
        and runtime["newly_selected_commands"] == (104 if phase == "producer" else 0)
        and branch["candidate_modules_unloaded"] == (phase == "checker"),
        "A complete bounded diagnostic trace and capture are required",
    )
    pair = prefix.read(target["capture_ref"])
    prefix.require(
        capture_differences(pair["expected"], pair["actual"]) == target["fields"],
        "Saved field diagnostics must reproduce from the persisted captures",
    )
    return dict(
        control_steps=len(rows),
        native_steps=runtime["counters"]["mj_step"]["completed"],
        capture_ref=target["capture_ref"],
        original_guard_passed=target["original_guard_passed"],
        differing_paths=target["fields"]["differing_paths"],
    )


def run():
    prefix.environment()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "Passing source-bound CPU QA and zero-physics preflight are required",
    )
    data = prepare()
    config, sources = data["config"], source_refs()
    prefix.require(dry["identity"] == data["identity"] and dry["source_refs"] == sources, "Dry inputs must match")
    registration_ref = publish_result(
        OUTPUT / "registration.json",
        dict(
            phases=config["phases"],
            identity=data["identity"],
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(
            key=phase,
            argv=[sys.executable, "-m", MODULE, "--worker", phase],
            log=OUTPUT / phase / "worker.log",
        )
        for phase in config["phases"]
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(OUTPUT / "workers.json", lifecycle)
    workers = {item["key"]: item for item in lifecycle["workers"]}
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        owner_closed=False,
        sources_unchanged=False,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        phases={},
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        closed = {}
        for phase in config["phases"]:
            entry = result["phases"][phase] = dict(status="not_run")
            info = workers.get(phase)
            if info is None:
                continue
            path = OUTPUT / phase / "worker/result.json"
            if path.is_file():
                entry["available_branch_ref"] = prefix.reference(path)
            try:
                prefix.require(
                    info["returncode"] == 0
                    and info["actual_wait"] is info["reaped"] is True
                    and info["signal_sent"] is False
                    and path.is_file(),
                    "The diagnostic worker must close normally",
                )
                branch_ref = prefix.reference(path)
                branch = prefix.read(branch_ref)
                counts = validate_branch(branch, phase, config)
                recorded = data["initial"] + [item for window in data["windows"] for item in window["periods"]]
                counts["original_prefix_max_difference"] = comparison.compare_periods(recorded, periods(branch))
                entry.update(status="closed", branch_ref=branch_ref, counts=counts)
                closed[phase] = branch
            except Exception as error:
                entry.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        if set(closed) == set(config["phases"]):
            result["replay_max_difference"] = comparison.compare_periods(
                periods(closed["producer"]), periods(closed["checker"])
            )
        unchanged = (
            snapshot(ROOT)["sources"] == before
            and source_refs() == sources
            and prepare()["identity"] == data["identity"]
        )
        owner_closed = len(workers) == len(config["phases"]) and all(
            item["actual_wait"] is item["reaped"] is True for item in workers.values()
        )
        result.update(
            owner_closed=owner_closed,
            sources_unchanged=unchanged,
            complete=owner_closed and unchanged and lifecycle["launch_error"] is None,
            passed=owner_closed
            and unchanged
            and lifecycle["launch_error"] is None
            and len(closed) == len(config["phases"]),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(OUTPUT / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("One or more diagnostic paths did not close")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", choices=("producer", "checker"))
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.run:
        run()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
