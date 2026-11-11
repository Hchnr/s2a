"""Run recovery workers with a five-request cumulative denoise boundary."""

import os
import random
import socket
import sys
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_live_admission_v2 as admission
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.wam.tracking.preview import live_cv2_admission


def case_directory(phase, index):
    prefix.require(
        phase in ("producer", "checker") and index in (0, 4, 8), "One registered phase and case are required"
    )
    return admission.OUTPUT / phase / f"case_{index:02d}"


def model_worker(index, load_model_inputs):
    """Load exactly one registered GPU model and serve up to five fresh images."""
    directory = case_directory("producer", index) / "model"
    spec = next(item for item in admission.CASES if item["case_index"] == index)
    gpu = spec["gpu_uuid"]
    result = dict(
        complete=False,
        passed=False,
        case_index=index,
        gpu_uuid=gpu,
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
        bundle, preflight = load_model_inputs(index)
        model_config = bundle["prepared"]["model_config"]
        prefix.require(
            Path(sys.executable).resolve() == Path(model_config["model_python"]).resolve(), "Model Python differs"
        )
        prefix.require(
            not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules),
            "Cosmos imported before admission",
        )
        result["cv2_admission"] = live_cv2_admission.admit_model_cv2(
            prefix.read(preflight["cv2_probe_ref"]), gpu_uuid=gpu
        )
        sys.path.insert(0, str(Path(model_config["cosmos_source_root"]).resolve()))
        run_dir = directory / "model_run"
        run_dir.mkdir()
        import numpy as np
        import torch as torch_module

        from s2a.backends.wam.runtime import install_active_slot_mask, load_service
        from scripts.verification.wam.model_contracts.audit_wam_current_numerics import require_eval
        from scripts.wam.readout.run_wam_readout_features import actual_context
        from scripts.wam.tracking.preview.reset_multitask_counter import make_counter
        from scripts.wam.tracking.preview.reset_multitask_live_requests import serve_live_requests
        from scripts.wam.training.clean.qualify_wam_clean_bridge import imported_cosmos_sources

        torch = torch_module
        prefix.require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device is required"
        )
        device = torch.cuda.get_device_properties(0)
        prefix.require(str(device.uuid).removeprefix("GPU-") == gpu.removeprefix("GPU-"), "Actual GPU UUID differs")
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
        counter = make_counter(service.model, bundle["model_registration"])
        imported_cosmos_sources(bundle["bridge"])
        from cosmos_framework.inference.args import OmniSampleOverrides

        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=3350
        )
        serve_live_requests(
            registration=bundle["model_registration"],
            channel=channel,
            image_root=case_directory("producer", index) / "simulation/runtime",
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
            0 <= requests <= 5
            and result["model_requests_attempted"] == result["model_requests_completed"] == requests
            and result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 35 * requests
            and len(result["responses"]) == requests
            and (
                (requests == 5 and result["model_sequence_complete"] is True and result["model_sequence_stop"] is None)
                or (
                    requests < 5
                    and result["model_sequence_complete"] is False
                    and result["model_sequence_stop"] is not None
                )
            ),
            "The actual model sequence must finish or receive a registered stop",
        )
        load_model_inputs(index)
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


def simulation_worker(index, phase, load_worker_inputs):
    """Execute from reset or independently replay one case's actual commands."""
    directory = case_directory(phase, index) / "simulation"
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case_index=index,
        automatic_retry=False,
        fresh_cases=0,
    )
    channel = None
    try:
        inputs, registration = load_worker_inputs(index, phase)
        selected = inputs["selected"][index]
        case = selected["case"]
        if phase == "checker":
            isolation.checker_isolation()
            pair = registration["producer_cases"][str(index)]
            producer, model = prefix.read(pair["branch_ref"]), prefix.read(pair["model_ref"])
            prefix.require(
                producer["complete"] is producer["passed"] is model["complete"] is model["passed"] is True
                and model["case_index"] == index
                and model["responses_sent"] == len(model["responses"]) <= 5
                and all(row["reference_sent"] is True for row in model["responses"]),
                "The closed matching producer and all actual model responses are required",
            )
            model_responses = [prefix.read(row["actual_response_ref"]) for row in model["responses"]]
        else:
            producer = model_responses = None
            channel = FramedJSONChannel(
                socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=3350
            )
        cv2 = inputs["previous"]["base"]["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime_48 import PrefixReplayFailure
        from s2a.research.native_runtime.reset_multi_window_runtime import ResetMultiWindowRuntime

        adapter = ResetMultiWindowRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        runtime_output = directory / "runtime"
        model_registration = admission.model_registration(
            inputs, index, case_directory("producer", index) / "simulation/runtime"
        )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if phase == "producer":
                from scripts.simulation.runtime.reset_live_multi_window import run_producer

                return run_producer(
                    env,
                    terminal,
                    observers,
                    receipt,
                    case=case,
                    baseline=selected["baseline_periods"],
                    registration=model_registration,
                    channel=channel,
                    runtime_output=runtime_output,
                    progress=progress,
                )
            from scripts.simulation.runtime.reset_live_multi_window_replay import run_checker

            return run_checker(
                env,
                terminal,
                observers,
                receipt,
                case=case,
                producer=producer,
                registration=model_registration,
                model_responses=model_responses,
                runtime_output=runtime_output,
                progress=progress,
            )

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                inputs["previous"]["base"]["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
                continuation=continuation,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in isolation.FORBIDDEN)
        prefix.require(phase != "checker" or unloaded is True, "Checker must not load candidate strategy modules")
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
