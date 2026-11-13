"""Verify model-worker seed and device admission without loading a GPU model."""

import copy
import hashlib
import json
import os
import signal
import socket
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from scripts.wam.tracking.preview import adaptive_multiseed_model_worker as worker
from scripts.wam.tracking.preview import adaptive_multiseed_requests as request_loop
from tests.research.wam.test_adaptive_model_worker import GPU, bundle, directories, environment, read_result
from tests.research.wam.test_adaptive_multiseed_contract import template
from tests.research.wam.test_reset_live_request import Counter, Overrides


@pytest.mark.parametrize("seed", [42, 43])
def test_registered_seed_reaches_worker_preflight_before_model_loading(tmp_path, monkeypatch, seed):
    output = directories(tmp_path)
    environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    value = bundle(output)
    value["model_registration"]["seed"] = seed
    handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="Model Python differs"):
        worker.run_model_worker(output=output, gpu_uuid=GPU, loader=lambda: (value, {}))
    result = read_result(output)
    assert result["sampling_seed"] == seed
    assert result["model_initialization_attempts"] == result["model_initializations"] == 0
    assert result["model_requests_attempted"] == result["denoise_calls_attempted"] == 0
    assert not result["passed"] and signal.getsignal(signal.SIGTERM) is handler


@pytest.mark.parametrize("change", ["seed", "boolean_seed", "requests", "periods", "task", "denoise", "image"])
def test_invalid_registration_stops_worker_before_model_initialization(tmp_path, monkeypatch, change):
    output = directories(tmp_path)
    environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    value = bundle(output)
    registration = value["model_registration"]
    key, replacement = {
        "seed": ("seed", 44),
        "boolean_seed": ("seed", True),
        "requests": ("max_requests", 76),
        "periods": ("max_periods", 494),
        "task": ("task", "unregistered task"),
        "denoise": ("denoise_steps", 34),
        "image": ("image_root", str(output.relative_to(worker.prefix.REPO))),
    }[change]
    registration[key] = replacement
    with pytest.raises(ValueError):
        worker.run_model_worker(output=output, gpu_uuid=GPU, loader=lambda: (value, {}))
    result = read_result(output)
    assert "sampling_seed" not in result
    assert result["model_initialization_attempts"] == result["model_initializations"] == 0
    assert result["model_requests_attempted"] == result["denoise_calls_attempted"] == 0


def test_wrong_cuda_mask_rejects_before_loader(tmp_path, monkeypatch):
    output = directories(tmp_path)
    environment(monkeypatch)
    calls = []
    with pytest.raises(ValueError, match="registered UUID"):
        worker.run_model_worker(output=output, gpu_uuid=GPU, loader=lambda: calls.append(True))
    assert calls == [] and read_result(output)["model_initialization_attempts"] == 0


@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("matching_device", [True, False])
def test_registered_seed_reaches_all_rngs_before_initialization(tmp_path, monkeypatch, seed, matching_device):
    output = directories(tmp_path)
    environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    value = bundle(output)
    value["model_registration"]["seed"] = seed
    source = output / "source"
    source.mkdir()
    value["prepared"]["model_config"] = {
        "model_python": sys.executable,
        "cosmos_source_root": str(source),
        "checkpoint": {"run": str(output / "weights"), "iteration": "iter_8400"},
    }
    calls = []
    monkeypatch.setattr(worker, "random", SimpleNamespace(seed=lambda x: calls.append(("python", x))))
    monkeypatch.setattr(np.random, "seed", lambda x: calls.append(("numpy", x)))
    monkeypatch.setattr(worker.prefix, "read", lambda _ref: {})
    monkeypatch.setattr(worker.live_cv2_admission, "admit_model_cv2", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "path", list(sys.path))

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "synthetic"
    fake_torch.version = SimpleNamespace(cuda="synthetic")
    fake_torch.manual_seed = lambda x: calls.append(("torch", x))
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(
            uuid=GPU if matching_device else "GPU-different", name="synthetic"
        ),
        manual_seed_all=lambda x: calls.append(("cuda", x)),
    )
    fake_torch.distributed = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fake_module(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    def load_service(*_args):
        calls.append(("load", None))
        raise RuntimeError("synthetic load boundary")

    fake_module("s2a.backends.wam.runtime", install_active_slot_mask=None, load_service=load_service)
    fake_module("scripts.verification.wam.model_contracts.audit_wam_current_numerics", require_eval=None)
    fake_module("scripts.wam.readout.run_wam_readout_features", BoundaryCounter=None, actual_context=None)
    fake_module("scripts.wam.training.clean.qualify_wam_clean_bridge", imported_cosmos_sources=None)

    def loader():
        return value, {"cv2_probe_ref": {}}

    if matching_device:
        with pytest.raises(RuntimeError, match="synthetic load boundary"):
            worker.run_model_worker(output=output, gpu_uuid=GPU, loader=loader)
        assert calls == [(name, seed) for name in ("python", "numpy", "torch", "cuda")] + [("load", None)]
        assert read_result(output)["model_initialization_attempts"] == 1
    else:
        with pytest.raises(ValueError, match="Actual GPU UUID differs"):
            worker.run_model_worker(output=output, gpu_uuid=GPU, loader=loader)
        assert calls == [] and read_result(output)["model_initialization_attempts"] == 0
    result = read_result(output)
    assert result["sampling_seed"] == seed and result["model_initializations"] == 0
    assert result["model_requests_attempted"] == result["denoise_calls_attempted"] == 0


def test_worker_refuses_output_outside_repository():
    with pytest.raises(ValueError, match="existing local model output"):
        worker.run_model_worker(output=Path("/tmp/not-a-registered-model-output"), gpu_uuid=GPU, loader=lambda: None)


@pytest.mark.parametrize("seed", [42, 43])
def test_worker_serves_registered_seed_over_real_framed_socket(tmp_path, monkeypatch, seed):
    output = directories(tmp_path)
    environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    value = bundle(output)
    registration = value["model_registration"]
    registration.update(seed=seed, max_requests=1)
    image = b"synthetic current observation"
    (output / "producer/simulation/current.png").write_bytes(image)
    ledger = AdaptiveStreamLedger(**registration)
    request = ledger.issue(
        before_render=ledger.current,
        after_render=ledger.current,
        image={"path": "current.png", "sha256": hashlib.sha256(image).hexdigest()},
    )
    source = output / "source"
    source.mkdir()
    checkpoint = {"run": str(output / "weights"), "iteration": "iter_8400"}
    expected_model = str(Path(checkpoint["run"]) / "checkpoints/iter_8400/model")
    value["prepared"]["model_config"] = {
        "model_python": sys.executable,
        "cosmos_source_root": str(source),
        "checkpoint": checkpoint,
    }
    value.update(template=template(), bridge={})
    calls, evidence, counter = [], [], Counter()
    monkeypatch.setattr(worker, "random", SimpleNamespace(seed=lambda x: calls.append(("python", x))))
    monkeypatch.setattr(request_loop, "random", worker.random)
    monkeypatch.setattr(np.random, "seed", lambda x: calls.append(("numpy", x)))
    monkeypatch.setattr(worker.prefix, "read", lambda _ref: {})
    monkeypatch.setattr(worker.live_cv2_admission, "admit_model_cv2", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "path", list(sys.path))

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "synthetic"
    fake_torch.version = SimpleNamespace(cuda="synthetic")
    fake_torch.manual_seed = lambda x: calls.append(("torch", x))
    fake_torch.inference_mode = nullcontext
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(uuid=GPU, name="synthetic"),
        manual_seed_all=lambda x: calls.append(("cuda", x)),
    )
    fake_torch.distributed = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fake_module(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    class FakeService:
        def __init__(self):
            self.info = {"checkpoint_path": expected_model, "ema_weights": False}
            self.model = SimpleNamespace(parameters=lambda: [])
            self.request = None

        def prepare(self, model_request, _directory):
            self.request = model_request
            task = registration["task"]
            return {}, {"ai_caption": [json.dumps({"task_title": task, "command": task, "task_plan": []})]}

        def execute(self, _sample, _batch):
            counter.calls.extend({"completed": True} for _ in range(35))
            evidence.append({"stage": "initial_noise", "inactive_rms_before": 1.0, "inactive_rms_after": 0.0})
            action = [[0.0] * 128 for _ in range(40)]
            identity = {
                "seed": seed,
                "action_fps": 20,
                "checkpoint_path": expected_model,
                "action_units": "raw_physical_or_source_units",
                "action_dim_mask": [True] * 10 + [False] * 118,
                "normalized_action": action,
            }
            response = {
                "status": "success",
                "args": copy.deepcopy(self.request),
                "outputs": [{"content": {"action": action, "action128": identity}}],
            }
            return SimpleNamespace(model_dump=lambda mode: response)

    service = FakeService()

    def load_service(*_args):
        calls.append(("load", None))
        fake_module("cosmos_framework", __path__=[])
        fake_module("cosmos_framework.inference", __path__=[])
        fake_module("cosmos_framework.inference.args", OmniSampleOverrides=Overrides)
        return service

    fake_module("s2a.backends.wam.runtime", install_active_slot_mask=lambda *_args: evidence, load_service=load_service)
    fake_module("scripts.verification.wam.model_contracts.audit_wam_current_numerics", require_eval=lambda _model: None)
    fake_module(
        "scripts.wam.readout.run_wam_readout_features",
        BoundaryCounter=lambda *_args: counter,
        actual_context=lambda *_args: SimpleNamespace(is_libero=True),
    )
    fake_module(
        "scripts.wam.training.clean.qualify_wam_clean_bridge",
        imported_cosmos_sources=lambda _bridge: ["synthetic"],
    )
    loads = []

    def loader():
        loads.append(True)
        return value, {"cv2_probe_ref": {}}

    left, right = socket.socketpair()
    channel = FramedJSONChannel(left, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    descriptor = right.detach()
    monkeypatch.setenv("S2A_CHANNEL_FD", str(descriptor))
    try:
        channel.send(request)
        worker.run_model_worker(output=output, gpu_uuid=GPU, loader=loader)
        response = channel.receive()
    finally:
        channel.close()
        try:
            os.close(descriptor)
        except OSError:
            pass
    assert loads == [True, True]
    assert calls == [(name, seed) for name in ("python", "numpy", "torch", "cuda")] + [("load", None)] + [
        (name, seed) for name in ("python", "numpy", "torch", "cuda")
    ]
    assert service.request["seed"] == response["seed"] == seed
    result = read_result(output)
    assert result["complete"] and result["passed"] and result["sampling_seed"] == seed
    assert result["model_initializations"] == result["responses_sent"] == 1
    assert result["denoise_calls_completed"] == 35 and response["denoise_calls"] == 35
