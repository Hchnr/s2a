"""Own one reset-to-current-image WAM producer and its independent replay."""

import argparse
import json
import os
import random
import socket
import sys
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_second_admission as admission
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import run_causal_stream_live_wam as prior_owner

ROOT, OUTPUT, QA, DRY, GPU = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY, admission.GPU
MODULE = "scripts.simulation.runtime.run_reset_live_second_window"


def dry_run():
    inputs = admission.load_inputs()
    result = dict(
        complete=True,
        passed=True,
        inputs=admission.input_receipt(inputs),
        source_refs=admission.source_refs(),
        model_initializations=0,
        model_requests=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(DRY, result), flush=True)


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A complete 32-worker source-matched regression is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot differs")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == admission.source_refs()
        and dry["model_initializations"] == dry["model_requests"] == dry["physics_steps"] == 0,
        "A matching zero-model zero-physics dry is required",
    )
    return before


def preflight():
    prefix.environment()
    inputs = admission.load_inputs()
    before = check_qa_and_dry(inputs)
    prefix.require(not OUTPUT.exists(), "Preflight output must be fresh")
    OUTPUT.mkdir()
    started = {"complete": False, "pid": os.getpid(), "config_ref": inputs["refs"]["config_ref"]}
    publish(OUTPUT / "preflight_started.json", started)
    try:
        sources = admission.source_refs()
        gpu = engine.gpu_observation()
        prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "The registered GPU must be idle")
        weights = inputs["weight_hashes"]
        before_stats = prior_owner.file_stats(weights)
        repro.verify_manifest(inputs["model_source_hashes"], workers=32)
        repro.verify_manifest(weights, workers=32)
        after_stats = prior_owner.file_stats(weights)
        prefix.require(before_stats == after_stats, "Checkpoint files changed while being verified")
        probe_dir = OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
        probe_ref = publish(probe_dir / "result.json", probe)
        prefix.require(
            snapshot(ROOT)["sources"] == before and admission.source_refs() == sources, "Preflight sources changed"
        )
        result = dict(
            complete=True,
            passed=True,
            inputs=admission.input_receipt(inputs),
            source_refs=sources,
            qa_ref=prefix.reference(QA),
            dry_ref=prefix.reference(DRY),
            gpu_uuid=GPU,
            gpu_before=gpu,
            cv2_probe_ref=probe_ref,
            source_files_verified=len(inputs["model_source_hashes"]),
            weight_files_verified=len(weights),
            weight_stats=after_stats,
            model_python_sha256=repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
            model_initializations=0,
            model_requests=0,
            physics_steps=0,
            automatic_retry=False,
        )
        print(publish(OUTPUT / "preflight.json", result), flush=True)
    except BaseException as error:
        publish(
            OUTPUT / "preflight_failed.json",
            {**started, "error": {"type": type(error).__name__, "message": str(error)}},
        )
        raise


def checked_preflight(inputs):
    value = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    prefix.require(
        value["complete"] is value["passed"] is True
        and value["inputs"] == admission.input_receipt(inputs)
        and value["source_refs"] == admission.source_refs()
        and value["qa_ref"] == prefix.reference(QA)
        and value["dry_ref"] == prefix.reference(DRY)
        and value["gpu_uuid"] == GPU
        and value["source_files_verified"] == len(inputs["model_source_hashes"])
        and value["weight_files_verified"] == len(inputs["weight_hashes"]) == 163
        and value["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"])
        and value["weight_stats"] == prior_owner.file_stats(inputs["weight_hashes"]),
        "The model preflight, source identity and checkpoint statistics must match",
    )
    return value


def worker_inputs(phase):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
    inputs = admission.load_inputs(verify_assets=False)
    checked_preflight(inputs)
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == admission.source_refs(),
        "Worker inputs differ from registration",
    )
    return inputs, registration


def model_bundle(inputs):
    return dict(
        input_receipt=admission.input_receipt(inputs),
        config=inputs["config"],
        prepared=inputs["prepared"],
        bridge=inputs["bridge"],
        template=inputs["template"],
        weight_hashes=inputs["weight_hashes"],
        model_registration=admission.model_registration(inputs, OUTPUT / "producer/simulation/runtime"),
    )


def model_worker_inputs():
    prefix.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and all(os.environ.get(key) == "1" for key in prefix.THREAD_VARS)
        and Path.cwd().resolve() == ROOT,
        "The model worker requires exactly its registered GPU and environment",
    )
    registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
    bundle = prefix.read(registration["model_bundle_ref"])
    preflight_result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    admission.validate_config(bundle["config"])
    expected_weights = {path: detail["sha256"] for path, detail in bundle["prepared"]["weight_files"].items()}
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == bundle["input_receipt"] == preflight_result["inputs"]
        and registration["source_refs"] == preflight_result["source_refs"] == admission.source_refs()
        and preflight_result["complete"] is preflight_result["passed"] is True
        and preflight_result["gpu_uuid"] == GPU
        and preflight_result["weight_files_verified"] == 163
        and preflight_result["qa_ref"] == prefix.reference(QA)
        and preflight_result["dry_ref"] == prefix.reference(DRY)
        and bundle["weight_hashes"] == expected_weights
        and preflight_result["weight_stats"] == prior_owner.file_stats(expected_weights)
        and preflight_result["model_python_sha256"] == repro.sha256(bundle["prepared"]["model_config"]["model_python"]),
        "A complete matching model bundle and preflight are required",
    )
    return bundle, preflight_result


def model_worker():
    directory = OUTPUT / "producer/model"
    result = dict(
        complete=False,
        passed=False,
        gpu_uuid=GPU,
        model_initialization_attempts=0,
        model_initializations=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        denoise_calls_attempted=0,
        denoise_calls_completed=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    channel = counter = evidence = torch = None
    try:
        bundle, preflight_result = model_worker_inputs()
        model_config = bundle["prepared"]["model_config"]
        prefix.require(
            Path(sys.executable).resolve() == Path(model_config["model_python"]).resolve(), "Model Python differs"
        )
        prefix.require(
            not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules),
            "Cosmos imported before admission",
        )
        result["cv2_admission"] = live_cv2_admission.admit_model_cv2(
            prefix.read(preflight_result["cv2_probe_ref"]), gpu_uuid=GPU
        )
        sys.path.insert(0, str(Path(model_config["cosmos_source_root"]).resolve()))
        run_dir = directory / "model_run"
        run_dir.mkdir()
        import numpy as np
        import torch as torch_module

        from s2a.backends.wam.runtime import install_active_slot_mask, load_service
        from scripts.verification.wam.model_contracts.audit_wam_current_numerics import require_eval
        from scripts.wam.readout.run_wam_readout_features import BoundaryCounter, actual_context
        from scripts.wam.tracking.preview.reset_live_request import serve_one
        from scripts.wam.training.clean.qualify_wam_clean_bridge import imported_cosmos_sources

        torch = torch_module
        prefix.require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device is required"
        )
        device = torch.cuda.get_device_properties(0)
        prefix.require(str(device.uuid).removeprefix("GPU-") == GPU.removeprefix("GPU-"), "Actual GPU UUID differs")
        result.update(torch_version=torch.__version__, compiled_cuda=torch.version.cuda, device_name=device.name)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        (run_dir / "model").mkdir()
        result["model_initialization_attempts"] = 1
        checkpoint = model_config["checkpoint"]
        service = load_service({"checkpoints": [checkpoint]}, "8400_masked", run_dir / "model")
        result["model_initializations"] = 1
        expected = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
        prefix.require(
            Path(service.info["checkpoint_path"]).resolve() == expected and service.info["ema_weights"] is False,
            "Loaded checkpoint or EMA differs",
        )
        require_eval(service.model)
        for parameter in service.model.parameters():
            parameter.requires_grad_(False)
        evidence = install_active_slot_mask(service.model, [True] * 10 + [False] * 118)
        counter = BoundaryCounter(service.model, 35)
        imported_cosmos_sources(bundle["bridge"])
        from cosmos_framework.inference.args import OmniSampleOverrides

        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
        )
        serve_one(
            registration=bundle["model_registration"],
            channel=channel,
            image_root=OUTPUT / "producer/simulation/runtime",
            directory=run_dir,
            template=bundle["template"],
            checkpoint=checkpoint,
            service=service,
            torch=torch,
            counter=counter,
            overrides=OmniSampleOverrides,
            domain_reader=actual_context,
            mask_evidence=evidence,
            result=result,
        )
        requests = result["responses_sent"]
        prefix.require(
            result["model_requests_attempted"] == result["model_requests_completed"] == requests
            and result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 35 * requests
            and (requests == 1 or (requests == 0 and result["model_sequence_stop"] is not None)),
            "One completed response or explicit pre-request stop is required",
        )
        model_worker_inputs()
        result["imported_cosmos_sources"] = imported_cosmos_sources(bundle["bridge"])
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if counter is not None:
            result["denoise_calls_attempted"] = len(counter.calls)
            result["denoise_calls_completed"] = sum(bool(call["completed"]) for call in counter.calls)
            result["denoise_evidence"] = counter.calls
        if evidence is not None:
            result["mask_evidence"] = evidence
        cleanup_error = None
        try:
            if channel is not None:
                channel.close()
            if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except BaseException as error:
            cleanup_error = error
            result.update(
                complete=False, passed=False, cleanup_error={"type": type(error).__name__, "message": str(error)}
            )
        print(publish(directory / "result.json", result), flush=True)
        if cleanup_error is not None:
            raise cleanup_error


def simulation_worker(phase):
    inputs, registration = worker_inputs(phase)
    directory = OUTPUT / phase / "simulation"
    result = dict(
        complete=False, passed=False, control_qualified=False, phase=phase, automatic_retry=False, fresh_cases=0
    )
    channel = None
    try:
        case = inputs["case"]
        if phase == "checker":
            isolation.checker_isolation()
            producer = prefix.read(registration["producer_branch_ref"])
            model = prefix.read(registration["producer_model_ref"])
            prefix.require(
                producer["complete"] is producer["passed"] is model["complete"] is model["passed"] is True,
                "Closed producer and model required",
            )
            model_response = (
                prefix.read(model["responses"][0]["actual_response_ref"]) if model["responses_sent"] else None
            )
        else:
            producer = model_response = None
            channel = FramedJSONChannel(
                socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
            )
        cv2 = inputs["base"]["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.reset_two_window_runtime import ResetTwoWindowRuntime

        adapter = ResetTwoWindowRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        runtime_output = directory / "runtime"
        model_registration = admission.model_registration(inputs, OUTPUT / "producer/simulation/runtime")

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if phase == "producer":
                from scripts.simulation.runtime.reset_live_two_window import run_producer

                return run_producer(
                    env,
                    terminal,
                    observers,
                    receipt,
                    case=case,
                    baseline=inputs["baseline_periods"],
                    registration=model_registration,
                    channel=channel,
                    runtime_output=runtime_output,
                    progress=progress,
                )
            from scripts.simulation.runtime.reset_live_two_window_replay import run_checker

            return run_checker(
                env,
                terminal,
                observers,
                receipt,
                case=case,
                producer=producer,
                registration=model_registration,
                model_response=model_response,
                runtime_output=runtime_output,
                progress=progress,
            )

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                inputs["base"]["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
                continuation=continuation,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in isolation.FORBIDDEN)
        prefix.require(phase != "checker" or unloaded is True, "Checker must not load any candidate strategy")
        result.update(
            complete=True, passed=True, case=case["case"], runtime=runtime, candidate_modules_unloaded=unloaded
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if channel is not None:
            channel.close()
        print(publish(directory / "result.json", result), flush=True)


def summarize_branch(branch, phase, producer=None):
    runtime, continuation = branch["runtime"], branch["runtime"]["continuation"]
    first = continuation["first_window"]["periods"]
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and len(first) == 8
        and [period["step"] for period in first] == list(range(8)),
        "The actual reset and completed first window are required",
    )
    if phase == "producer":
        stream = continuation["stream"]
        prefix.require(
            len(stream["segments"]) <= 1
            and stream["requests_issued"] == stream["responses_accepted"] == len(stream["segments"]),
            "At most one admitted live segment is required",
        )
        periods = [] if not stream["segments"] else stream["segments"][0]["continuation"]["periods"]
        prefix.require(
            stream["control_steps_attempted"]
            == stream["control_steps_completed"]
            == stream["feedback_periods_committed"]
            == len(periods)
            and (
                (stream["complete"] is True and stream["stop"] is None and len(periods) == 8)
                or (
                    stream["complete"] is False
                    and type(stream["stop"]) is dict
                    and stream["stop"].get("next_action_dispatched") is False
                )
            ),
            "The live segment must complete or explicitly stop before another action",
        )
        prefix.require(
            runtime["newly_selected_commands"] == 8 + len(periods), "Each producer action must be newly selected"
        )
        segments = runtime["segment_receipts"]
        prefix.require(len(segments) == 1 + len(stream["segments"]), "One receipt per selected policy segment required")
        quality = sum(item["quality_refiner"]["quality_response_evaluations"] for item in segments)
        baseline = sum(item["quality_refiner"]["baseline_response_evaluations"] for item in segments)
        gripper = sum(item["gripper_proposals"]["candidate_sequences"] for item in segments)
        endpoints = sum(item["quality_refiner"]["endpoint_queries_charged"] for item in segments)
        transitions = sum(item["quality_refiner"]["transition_steps_charged"] for item in segments)
        prefix.require(
            quality <= 16 * 320
            and baseline <= 16 * 480
            and gripper <= 16 * 6561
            and endpoints <= 3200
            and transitions <= 41600,
            "Candidate computation exceeds the registered budget",
        )
        model_requests = stream["requests_issued"]
        stop = stream["stop"]
    else:
        prefix.require(
            phase == "checker" and branch["candidate_modules_unloaded"] is True and producer is not None,
            "The isolated checker and producer are required",
        )
        replay = continuation["replay"]
        periods = [] if replay is None else replay["periods"]
        producer_stream = producer["runtime"]["continuation"]["stream"]
        expected = [] if not producer_stream["segments"] else producer_stream["segments"][0]["continuation"]["periods"]
        prefix.require(
            len(periods) == len(expected) and runtime["newly_selected_commands"] == 0,
            "Checker must replay exactly the dispatched live commands",
        )
        model_requests, stop = 0, continuation["reported_producer_stop"]
    count = 8 + len(periods)
    counters = runtime["counters"]
    prefix.require(
        count <= 16
        and runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == count * 25
        and counters["model_construct"]["completed"] == counters["context"]["completed"] == 3
        and [period["step"] for period in periods] == list(range(8, count))
        and all(len(period["actual_inner"]) == 25 for period in first + periods),
        "All actual model and physical counts must match",
    )
    return dict(
        control_steps=count,
        native_steps=count * 25,
        models=3,
        contexts=3,
        live_control_steps=len(periods),
        model_requests=model_requests,
        stop=stop,
    )


def execute(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "Known phase required")
    inputs = admission.load_inputs()
    before = check_qa_and_dry(inputs)
    checked_preflight(inputs)
    sources = admission.source_refs()
    producer_ref = model_ref = producer = None
    if phase == "checker":
        prior = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            prior["complete"] is prior["passed"] is prior["sources_unchanged"] is prior["gpu_released"] is True,
            "The producer and GPU worker must be fully closed",
        )
        producer_ref, model_ref = prior["branch_ref"], prior["model_ref"]
        producer = prefix.read(producer_ref)
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    (directory / "simulation").mkdir()
    bundle_ref = None
    if phase == "producer":
        (directory / "model").mkdir()
        bundle_ref = publish(directory / "model/registered_inputs.json", model_bundle(inputs))
    registration = publish(
        directory / "registration.json",
        dict(
            phase=phase,
            inputs=admission.input_receipt(inputs),
            source_refs=sources,
            producer_branch_ref=producer_ref,
            producer_model_ref=model_ref,
            model_bundle_ref=bundle_ref,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        registration_ref=registration,
        gpu_uuid=GPU,
        automatic_retry=False,
        model_initializations=None,
        model_requests_attempted=None,
        denoise_calls_attempted=None,
    )
    try:
        simulation_env = physical._cpu_environment(directory / "simulation")
        command = [sys.executable, "-m", MODULE]
        if phase == "producer":
            before_gpu = engine.gpu_observation()
            prefix.require(
                before_gpu["memory_mib"] == 0 and before_gpu["compute_apps"] == [], "The registered GPU must be idle"
            )
            result["gpu_before"] = before_gpu
            model_env = prior_owner._model_environment(directory / "model")
            model_python = inputs["prepared"]["model_config"]["model_python"]
            specs = {
                "simulation": dict(
                    argv=command + ["--worker", "simulation"],
                    env=simulation_env,
                    log=str((directory / "simulation.log").relative_to(ROOT)),
                ),
                "model": dict(
                    argv=[model_python, "-m", MODULE, "--worker", "model"],
                    env=model_env,
                    log=str((directory / "model.log").relative_to(ROOT)),
                ),
            }
            try:
                lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
            except OwnedSocketPairInterrupted as error:
                result["workers_ref"] = publish(directory / "workers.json", error.result)
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            prefix.require(lifecycle["passed"] is True, "Both model and simulation workers must exit and be reaped")
            model_ref = prefix.reference(directory / "model/result.json")
            model = prefix.read(model_ref)
            result["model_ref"] = model_ref
            result.update(
                model_initializations=model["model_initializations"],
                model_requests_attempted=model["model_requests_attempted"],
                denoise_calls_attempted=model["denoise_calls_attempted"],
            )
            result["gpu_after"] = engine.gpu_observation()
            result["gpu_released"] = (
                result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
            )
            prefix.require(result["gpu_released"] is True, "The selected GPU must be released")
        else:
            spec = dict(key="checker", argv=command + ["--worker", "checker"], log=directory / "checker.log")
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
            result.update(
                gpu_released=True, model_initializations=0, model_requests_attempted=0, denoise_calls_attempted=0
            )
        branch_ref = prefix.reference(directory / "simulation/result.json")
        branch = prefix.read(branch_ref)
        counts = summarize_branch(branch, phase, producer=producer)
        if phase == "producer":
            prefix.require(
                model["complete"] is model["passed"] is True
                and model["model_initializations"] == 1
                and model["model_requests_completed"] == model["responses_sent"] == counts["model_requests"]
                and model["denoise_calls_completed"] == 35 * counts["model_requests"],
                "The live model and physical exchange counts must match",
            )
        else:
            prefix.require(
                counts["control_steps"] == prior["counts"]["control_steps"]
                and counts["native_steps"] == prior["counts"]["native_steps"],
                "Checker and producer physical counts differ",
            )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            admission.source_refs() == sources and snapshot(ROOT)["sources"] == before,
            "Sources changed during execution",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    finally:
        if phase == "producer":
            model_path = directory / "model/result.json"
            if model_path.is_file():
                result["available_model_ref"] = prefix.reference(model_path)
                available = json.loads(model_path.read_text())
                result.update(
                    model_initializations=available.get("model_initializations"),
                    model_requests_attempted=available.get("model_requests_attempted"),
                    denoise_calls_attempted=available.get("denoise_calls_attempted"),
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
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("simulation", "model", "checker"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.preflight:
        preflight()
    elif args.phase:
        execute(args.phase)
    elif args.worker == "model":
        model_worker()
    else:
        simulation_worker("producer" if args.worker == "simulation" else "checker")


if __name__ == "__main__":
    main()
