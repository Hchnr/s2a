"""Supervise one registered six-window live WAM development trial."""

import argparse
import json
import os
import sys
from pathlib import Path

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_48 as simulation
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import causal_stream_live_48_admission as admission
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission, live_stream_model_48
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import run_causal_stream_live_wam as previous

ROOT, OUTPUT, QA, DRY, GPU = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY, admission.GPU
MODULE = "scripts.wam.tracking.preview.run_causal_stream_live_48"


def source_refs():
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/wam/tracking/preview/causal_stream_live_48_admission.py",
        ROOT / "scripts/wam/tracking/preview/live_stream_model_48.py",
        ROOT / "scripts/wam/tracking/preview/live_stream_requests_48.py",
        ROOT / "scripts/simulation/runtime/run_causal_stream_live_48.py",
        ROOT / "scripts/simulation/runtime/causal_stream_live_48_replay.py",
        ROOT / "src/s2a/research/native_runtime/development_prefix_runtime_48.py",
        ROOT / "src/s2a/research/native_runtime/guarded_stream_prefix.py",
        ROOT / "src/s2a/research/native_runtime/streaming_replay_48.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/causal_stream_control.py",
        ROOT / "tests/research/native_runtime/test_guarded_stream_prefix.py",
        ROOT / "tests/research/native_runtime/test_causal_stream_live_48_runtime.py",
        ROOT / "tests/research/wam/test_causal_stream_live_48_admission.py",
        ROOT / "tests/research/wam/test_causal_stream_live_48_replay.py",
        ROOT / "tests/research/wam/test_causal_stream_live_48_owner.py",
        ROOT / "tests/research/wam/test_live_stream_requests_48.py",
    )
    return previous.source_refs() + [prefix.reference(path) for path in names]


def dry_run():
    inputs = admission.load_inputs()
    result = {
        "complete": True,
        "passed": True,
        "inputs": admission.input_receipt(inputs),
        "source_refs": source_refs(),
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    print(publish(DRY, result), flush=True)


def check_cpu_admission(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and qa["configured_parallelism"] == 64,
        "Complete 64-worker CPU QA without skips or collection errors is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Long-stream QA source identity changed")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["new_physics_steps"] == dry["new_model_requests"] == dry["new_denoise_calls"] == 0,
        "Matching zero-model and zero-physics dry admission required",
    )
    return before


def preflight():
    prefix.environment()
    inputs = admission.load_inputs()
    before = check_cpu_admission(inputs)
    prefix.require(not OUTPUT.exists(), "Preflight and physical outputs must be fresh")
    OUTPUT.mkdir()
    started = {"complete": False, "pid": os.getpid(), "config_ref": inputs["long_config_ref"]}
    publish(OUTPUT / "preflight_started.json", started)
    try:
        sources = source_refs()
        gpu = engine.gpu_observation()
        prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "Registered GPU must be idle")
        manifests = previous.verify_model_manifests(inputs)
        probe_dir = OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
        probe_ref = publish(probe_dir / "result.json", probe)
        prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Preflight sources changed")
        result = {
            "complete": True,
            "passed": True,
            "inputs": admission.input_receipt(inputs),
            "source_refs": sources,
            "qa_ref": prefix.reference(QA),
            "dry_ref": prefix.reference(DRY),
            "gpu_uuid": GPU,
            "gpu_before": gpu,
            "cv2_probe_ref": probe_ref,
            **manifests,
            "model_python_sha256": repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
            "model_initializations": 0,
            "model_requests": 0,
            "denoise_calls": 0,
            "physics_steps": 0,
            "automatic_retry": False,
        }
        print(publish(OUTPUT / "preflight.json", result), flush=True)
    except BaseException as error:
        publish(
            OUTPUT / "preflight_failed.json",
            {**started, "error": {"type": type(error).__name__, "message": str(error)}},
        )
        raise


def checked_preflight(inputs):
    result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    prefix.require(
        result["complete"] is result["passed"] is True
        and result["inputs"] == admission.input_receipt(inputs)
        and result["source_refs"] == source_refs()
        and result["qa_ref"] == prefix.reference(QA)
        and result["dry_ref"] == prefix.reference(DRY)
        and result["gpu_uuid"] == GPU
        and result["source_files_verified"] >= 100
        and result["weight_files_verified"] == 163
        and result["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"])
        and previous.file_stats(inputs["weight_hashes"]) == result["weight_stats"],
        "The long-stream preflight and checkpoint identity must remain unchanged",
    )
    probe = prefix.read(result["cv2_probe_ref"])
    prefix.require(
        probe["actual_wait"] is probe["reaped"] is True and probe["returncode"] == 0,
        "A completed CPU OpenCV probe is required",
    )
    return result


def worker_inputs(phase):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
    inputs = admission.load_inputs()
    checked_preflight(inputs)
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Worker inputs and sources must match the owner registration",
    )
    return inputs, registration


def model_worker_inputs():
    prefix.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and all(os.environ.get(key) == "1" for key in prefix.THREAD_VARS)
        and Path.cwd().resolve() == ROOT,
        "The model worker must expose exactly its registered GPU and repository environment",
    )
    registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
    bundle = prefix.read(registration["model_bundle_ref"])
    preflight_result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    admission.validate_config(bundle["live_config"])
    expected_weights = {path: detail["sha256"] for path, detail in bundle["prepared"]["weight_files"].items()}
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == bundle["input_receipt"] == preflight_result["inputs"]
        and registration["source_refs"] == preflight_result["source_refs"] == source_refs()
        and preflight_result["complete"] is preflight_result["passed"] is True
        and preflight_result["gpu_uuid"] == GPU
        and preflight_result["source_files_verified"] >= 100
        and preflight_result["weight_files_verified"] == 163
        and preflight_result["qa_ref"] == prefix.reference(QA)
        and preflight_result["dry_ref"] == prefix.reference(DRY)
        and bundle["weight_hashes"] == expected_weights
        and bundle["model_sha256"] == physical.admission.FIXED["historical_weight_manifest_sha256"]
        and preflight_result["model_python_sha256"] == repro.sha256(bundle["prepared"]["model_config"]["model_python"])
        and previous.file_stats(bundle["weight_hashes"]) == preflight_result["weight_stats"],
        "The model worker requires its frozen bundle and unchanged preflight",
    )
    probe = prefix.read(preflight_result["cv2_probe_ref"])
    prefix.require(probe["actual_wait"] is probe["reaped"] is True and probe["returncode"] == 0, "Completed CPU probe")
    return bundle


def model_bundle(inputs):
    bundle = previous.model_bundle(inputs)
    bundle["input_receipt"] = admission.input_receipt(inputs)
    return bundle


def summarize_branch(branch, *, phase, producer=None):
    runtime = branch["runtime"]
    stream = runtime["continuation"]
    periods = [period for row in stream["segments"] for period in row["continuation"]["periods"]]
    count = len(periods)
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is True
        and runtime["prefix_control_steps"] == 80
        and runtime["control_steps_completed"] == 80 + count
        and runtime["counters"]["mj_step"]["completed"] == 25 * (80 + count)
        and 0 <= count <= 48
        and len(runtime["prefix_inner_periods"]) == 80
        and sum(len(row["samples"]) for row in runtime["prefix_inner_periods"]) == 2000
        and [row["step"] for row in periods] == list(range(80, 80 + count))
        and all(len(row["actual_inner"]) == 25 for row in periods),
        "Every physically completed long-stream period must be accounted for",
    )
    if phase == "producer":
        stop = stream["stop"]
        prefix.require(
            stream["control_steps_attempted"]
            == stream["control_steps_completed"]
            == stream["feedback_periods_committed"]
            == count
            and stream["requests_issued"] == stream["responses_accepted"] == len(stream["segments"])
            and all(row["inflight"] is None for row in stream["segments"])
            and runtime["newly_selected_commands"] == count
            and (
                (stream["complete"] is True and stop is None and count == 48)
                or (
                    stream["complete"] is False
                    and type(stop) is dict
                    and stop.get("at_step") == 80 + count
                    and stop.get("next_action_dispatched") is False
                )
            ),
            "The producer must finish or explicitly stop before its next dispatch",
        )
        loss = [period["step"] for period in periods if period["inner_below_buffer_samples"]]
        prefix.require(
            not loss or (loss == [79 + count] and stop is not None), "No action may follow an observed buffer loss"
        )
    else:
        prefix.require(
            producer is not None
            and stream["complete"] is True
            and stream["control_steps_attempted"] == stream["control_steps_completed"] == count
            and stream["reported_producer_stop"] == producer["runtime"]["continuation"]["stop"]
            and branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0,
            "The isolated checker must replay all completed actions and preserve the reported stop",
        )
    return {
        "control_steps": 80 + count,
        "native_steps": 25 * (80 + count),
        "continuation_steps": count,
        "prefix_inner_samples": 2000,
        "continuation_inner_samples": 25 * count,
    }


def check_model_branch(model, branch):
    stream = branch["runtime"]["continuation"]
    requests = stream["requests_issued"]
    prefix.require(
        model["complete"] is model["passed"] is True
        and model["model_initializations"] == 1
        and model["model_requests_attempted"]
        == model["model_requests_completed"]
        == model["responses_sent"]
        == requests
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 35 * requests
        and 0 <= requests <= 6,
        "Model work must equal the actually accepted physical requests",
    )
    if requests < 6:
        stop = branch["runtime"]["stream_stop_frame"]
        prefix.require(
            stream["complete"] is False
            and stop["required"] is stop["send_attempted"] is stop["sent"] is True
            and model["model_sequence_complete"] is False
            and model["model_sequence_stop"] == stop["frame"],
            "A short stream requires the same explicit stop on both workers",
        )
    else:
        prefix.require(
            model["model_sequence_complete"] is True and model["model_sequence_stop"] is None,
            "The full six-request model budget must close once",
        )


def execute(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered phase required")
    inputs = admission.load_inputs()
    before = check_cpu_admission(inputs)
    checked_preflight(inputs)
    sources = source_refs()
    producer_ref, producer = None, None
    if phase == "checker":
        prior = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            prior["complete"] is prior["passed"] is prior["sources_unchanged"] is prior["gpu_released"] is True,
            "The producer and its GPU worker must be fully closed",
        )
        producer_ref = prior["branch_ref"]
        producer = prefix.read(producer_ref)
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    (directory / "simulation").mkdir()
    bundle_ref = None
    if phase == "producer":
        (directory / "model").mkdir()
        bundle = model_bundle(inputs)
        bundle_ref = publish(directory / "model/registered_inputs.json", bundle)
    registration = publish(
        directory / "registration.json",
        {
            "phase": phase,
            "inputs": admission.input_receipt(inputs),
            "source_refs": sources,
            "producer_branch_ref": producer_ref,
            "model_bundle_ref": bundle_ref,
        },
    )
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "control_qualified": False,
        "registration_ref": registration,
        "gpu_uuid": GPU,
        "automatic_retry": False,
        "model_initializations": None,
        "model_requests_attempted": None,
        "denoise_calls_attempted": None,
    }
    try:
        simulation_env = physical._cpu_environment(directory / "simulation")
        command = [sys.executable, "-m", MODULE]
        if phase == "producer":
            before_gpu = engine.gpu_observation()
            prefix.require(
                before_gpu["memory_mib"] == 0 and before_gpu["compute_apps"] == [], "Registered GPU must be idle"
            )
            result["gpu_before"] = before_gpu
            model_env = previous._model_environment(directory / "model")
            model_python = inputs["prepared"]["model_config"]["model_python"]
            specs = {
                "simulation": {
                    "argv": command + ["--worker", "simulation"],
                    "env": simulation_env,
                    "log": str((directory / "simulation.log").relative_to(ROOT)),
                },
                "model": {
                    "argv": [model_python, "-m", MODULE, "--worker", "model"],
                    "env": model_env,
                    "log": str((directory / "model.log").relative_to(ROOT)),
                },
            }
            try:
                lifecycle = run_owned_socket_pair(
                    specs, timeout_seconds=inputs["live_config"]["worker_timeout_seconds"]
                )
            except OwnedSocketPairInterrupted as error:
                result["workers_ref"] = publish(directory / "workers.json", error.result)
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            prefix.require(lifecycle["passed"] is True, "Both long-stream workers must exit and be reaped")
            model_ref = prefix.reference(directory / "model/result.json")
            model = prefix.read(model_ref)
            result["model_ref"] = model_ref
            result.update(
                model_initializations=model["model_initializations"],
                model_requests_attempted=model["model_requests_attempted"],
                denoise_calls_attempted=model["denoise_calls_attempted"],
            )
            after_gpu = engine.gpu_observation()
            result["gpu_after"] = after_gpu
            result["gpu_released"] = after_gpu["memory_mib"] == 0 and after_gpu["compute_apps"] == []
            prefix.require(result["gpu_released"] is True, "The selected GPU must be released")
        else:
            spec = {"key": "checker", "argv": command + ["--worker", "checker"], "log": directory / "checker.log"}
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=inputs["live_config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            physical.historical.registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
            result.update(
                gpu_released=True, model_initializations=0, model_requests_attempted=0, denoise_calls_attempted=0
            )
        branch_ref = prefix.reference(directory / "simulation/result.json")
        branch = prefix.read(branch_ref)
        counts = summarize_branch(branch, phase=phase, producer=producer)
        if phase == "producer":
            check_model_branch(model, branch)
        else:
            prefix.require(counts == prior["counts"], "Checker and producer physical counts must match")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Sources changed during execution"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    finally:
        if phase == "producer":
            model_path = directory / "model/result.json"
            if model_path.is_file():
                result["available_model_ref"] = prefix.reference(model_path)
                model = json.loads(model_path.read_text())
                result.update(
                    model_initializations=model.get("model_initializations"),
                    model_requests_attempted=model.get("model_requests_attempted"),
                    denoise_calls_attempted=model.get("denoise_calls_attempted"),
                )
            if "gpu_after" not in result:
                try:
                    result["gpu_after"] = engine.gpu_observation()
                    result["gpu_released"] = (
                        result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
                    )
                except Exception as error:
                    result["gpu_after_error"] = {"type": type(error).__name__, "message": str(error)}
        branch_path = directory / "simulation/result.json"
        if branch_path.is_file():
            result["available_branch_ref"] = prefix.reference(branch_path)
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", choices=("simulation", "model", "checker"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.preflight:
        preflight()
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    elif args.worker == "simulation":
        simulation.simulation_worker(output=OUTPUT, loader=worker_inputs)
    elif args.worker == "checker":
        simulation.checker_worker(output=OUTPUT, loader=worker_inputs)
    else:
        live_stream_model_48.model_worker(output=OUTPUT, loader=model_worker_inputs)


if __name__ == "__main__":
    main()
