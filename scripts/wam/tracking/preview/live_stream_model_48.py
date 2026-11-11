"""Own one six-window WAM worker with source-pinned model admission."""

import os
import random
import socket
import sys
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import causal_stream_live_48_admission as admission
from scripts.wam.tracking.preview import live_cv2_admission

GPU = admission.GPU


def model_worker(*, output, loader):
    output = Path(output)
    max_requests = 6
    directory = output / "producer/model"
    result = {
        "complete": False,
        "passed": False,
        "gpu_uuid": GPU,
        "model_initialization_attempts": 0,
        "model_initializations": 0,
        "model_requests_attempted": 0,
        "model_requests_completed": 0,
        "denoise_calls_attempted": 0,
        "denoise_calls_completed": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    channel, counter, evidence, torch = None, None, None, None
    try:
        inputs = loader()
        prefix.require(
            max_requests == inputs["live_config"].get("max_model_requests") == 6
            and inputs["live_config"].get("max_denoise_calls") == 210,
            "Explicit six-window and 210-denoise model limits are required",
        )
        prefix.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and os.environ.get("LD_LIBRARY_PATH") == "",
            "The model worker must expose only its registered GPU UUID",
        )
        model_config = inputs["prepared"]["model_config"]
        prefix.require(
            Path(sys.executable).resolve() == Path(model_config["model_python"]).resolve(), "Model Python differs"
        )
        prefix.require(
            not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules),
            "Cosmos was imported before its source admission",
        )
        probe_ref = prefix.read(prefix.reference(output / "preflight.json"))["cv2_probe_ref"]
        result["cv2_admission"] = live_cv2_admission.admit_model_cv2(prefix.read(probe_ref), gpu_uuid=GPU)
        sys.path.insert(0, str(Path(model_config["cosmos_source_root"]).resolve()))
        run_dir = directory / "model_run"
        run_dir.mkdir()
        import numpy as np
        import torch as torch_module

        from s2a.backends.wam.runtime import install_active_slot_mask, load_service
        from scripts.verification.wam.model_contracts.audit_wam_current_numerics import require_eval
        from scripts.wam.readout.run_wam_readout_features import BoundaryCounter, actual_context
        from scripts.wam.tracking.preview.live_stream_requests_48 import execute_live_requests
        from scripts.wam.training.clean.qualify_wam_clean_bridge import imported_cosmos_sources

        torch = torch_module
        prefix.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device required")
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
            "Loaded WAM checkpoint or EMA differs",
        )
        require_eval(service.model)
        for parameter in service.model.parameters():
            parameter.requires_grad_(False)
        evidence = install_active_slot_mask(service.model, [True] * 10 + [False] * 118)
        counter = BoundaryCounter(service.model, 35 * max_requests)
        imported_cosmos_sources(inputs["bridge"])
        from cosmos_framework.inference.args import OmniSampleOverrides

        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
        )
        execute_live_requests(
            inputs,
            channel=channel,
            image_root=output / "producer/simulation/runtime",
            directory=run_dir,
            service=service,
            torch=torch,
            counter=counter,
            overrides=OmniSampleOverrides,
            domain_reader=actual_context,
            mask_evidence=evidence,
            result=result,
            max_requests=max_requests,
        )
        requests = result["responses_sent"]
        prefix.require(
            result["model_requests_attempted"] == result["model_requests_completed"] == requests
            and result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 35 * requests
            and 0 <= requests <= max_requests
            and (
                (
                    requests == max_requests
                    and result["model_sequence_complete"]
                    and result["model_sequence_stop"] is None
                )
                or (requests < max_requests and result["model_sequence_stop"] is not None)
            ),
            "Every completed response must have 35 actual denoise calls or an explicit bounded stop",
        )
        loader()
        result["imported_cosmos_sources"] = imported_cosmos_sources(inputs["bridge"])
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
