"""Own one explicitly admitted GPU model for an adaptive reference stream."""

import os
import random
import signal
import socket
import sys
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import live_cv2_admission


def validate_registration(registration, output):
    """Validate the complete model/physical request budget before model loading."""
    from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission
    from scripts.wam.tracking.preview.reset_multitask_live_requests import TASKS

    output = Path(output).resolve()
    prefix.require(
        type(registration) is dict
        and type(registration.get("max_requests")) is int
        and 1 <= registration["max_requests"] <= 75
        and registration.get("denoise_steps") == 35
        and registration.get("seed") == 42
        and registration.get("task") in TASKS
        and type(registration.get("max_periods")) is int
        and type(registration.get("initial")) is dict
        and type(registration["initial"].get("step")) is int
        and registration["initial"]["step"] + registration["max_periods"] <= 600
        and type(registration.get("image_root")) is str
        and (prefix.REPO / registration["image_root"]).resolve() == output / "producer/simulation"
        and (output / "producer/simulation").is_dir(),
        "Explicit bounded request and sampling limits are required before model initialization",
    )
    return AdaptiveModelAdmission(**registration)


def run_model_worker(*, output, gpu_uuid, loader):
    """Load one admitted model and serve the registered bounded request loop.

    The loader must revalidate the immutable registration, source and checkpoint
    preflight both before and after serving. The parent owns this process,
    deadline, GPU release verification and all partial-failure accounting.
    """
    output = Path(output).resolve()
    directory = output / "producer/model"
    if (
        not output.is_relative_to(prefix.REPO / "outputs")
        or not directory.is_dir()
        or type(gpu_uuid) is not str
        or not gpu_uuid.startswith("GPU-")
        or not callable(loader)
    ):
        raise ValueError("A registered GPU, existing local model output and loader are required")
    result = dict(
        complete=False,
        passed=False,
        gpu_uuid=gpu_uuid,
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

    def terminate(signum, _frame):
        raise InterruptedError(f"Model worker received signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        prefix.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == gpu_uuid
            and os.environ.get("LD_LIBRARY_PATH") == ""
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            and all(os.environ.get(key) == "1" for key in prefix.THREAD_VARS)
            and Path.cwd().resolve() == prefix.REPO,
            "The model worker requires its registered UUID and single-thread environment",
        )
        bundle, preflight = loader()
        registration = bundle["model_registration"]
        validate_registration(registration, output)
        model_config = bundle["prepared"]["model_config"]
        prefix.require(
            Path(sys.executable).resolve() == Path(model_config["model_python"]).resolve(), "Model Python differs"
        )
        prefix.require(
            not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules),
            "Cosmos was imported before model source admission",
        )
        result["cv2_admission"] = live_cv2_admission.admit_model_cv2(
            prefix.read(preflight["cv2_probe_ref"]), gpu_uuid=gpu_uuid
        )
        source_root = Path(model_config["cosmos_source_root"]).resolve()
        prefix.require(source_root.is_relative_to(prefix.REPO / "outputs"), "The admitted model source must be local")
        sys.path.insert(0, str(source_root))
        run_dir = directory / "model_run"
        run_dir.mkdir()
        import numpy as np
        import torch as torch_module

        from s2a.backends.wam.runtime import install_active_slot_mask, load_service
        from scripts.verification.wam.model_contracts.audit_wam_current_numerics import require_eval
        from scripts.wam.readout.run_wam_readout_features import BoundaryCounter, actual_context
        from scripts.wam.tracking.preview.adaptive_live_requests import serve_adaptive_requests
        from scripts.wam.training.clean.qualify_wam_clean_bridge import imported_cosmos_sources

        torch = torch_module
        prefix.require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device is required"
        )
        device = torch.cuda.get_device_properties(0)
        prefix.require(
            str(device.uuid).removeprefix("GPU-") == gpu_uuid.removeprefix("GPU-"), "Actual GPU UUID differs"
        )
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
        counter = BoundaryCounter(service.model, 35 * registration["max_requests"])
        imported_cosmos_sources(bundle["bridge"])
        from cosmos_framework.inference.args import OmniSampleOverrides

        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
        )
        serve_adaptive_requests(
            registration=registration,
            channel=channel,
            image_root=prefix.REPO / registration["image_root"],
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
            and 0 <= requests <= registration["max_requests"]
            and (
                (requests == registration["max_requests"] and result["model_sequence_complete"])
                or (requests < registration["max_requests"] and result["model_sequence_stop"] is not None)
            ),
            "Every response must have 35 actual denoise calls or an explicit bounded stop",
        )
        loader()
        result["imported_cosmos_sources"] = imported_cosmos_sources(bundle["bridge"])
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
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
