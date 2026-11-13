"""Exercise the seed-43 worker with an actual socket and synthetic model."""

import copy
import json
import os
import socket
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from scripts.simulation.runtime import reset_seed43_live_admission as admission
from scripts.wam.tracking.preview import reset_live_request as legacy
from scripts.wam.tracking.preview import reset_seed43_model_worker as worker
from tests.research.wam.test_adaptive_model_worker import environment, read_result
from tests.research.wam.test_adaptive_multiseed_contract import template
from tests.research.wam.test_reset_live_request import Counter, Overrides
from tests.research.wam.test_reset_seed43_live_request import setup


def test_preflight_import_does_not_load_cv2_or_gpu_libraries(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    for name in worker.prefix.THREAD_VARS:
        env[name] = "1"
    env["TMPDIR"] = str(tmp_path)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.wam.tracking.preview.reset_seed43_model_worker; "
            "assert not any(x in sys.modules for x in ('cv2', 'torch', 'cosmos_framework'))",
        ],
        cwd=worker.prefix.REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


@pytest.mark.parametrize("change", ["seed", "state", "time", "session", "requests"])
def test_invalid_registration_stops_before_model_loading(tmp_path, monkeypatch, change):
    items = setup(tmp_path)
    output, registration = items[:2]
    environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", admission.GPU)
    key, value = {
        "seed": ("seed", 42),
        "state": ("initial", dict(registration["initial"], state_sha256="0" * 64)),
        "time": ("initial", dict(registration["initial"], sim_time_s=0.65)),
        "session": ("session", "another-session"),
        "requests": ("max_requests", 2),
    }[change]
    registration[key] = value
    bundle = {"model_registration": registration}
    with pytest.raises(ValueError, match="registered seed-43"):
        worker.run_model_worker(output=output, gpu_uuid=admission.GPU, loader=lambda: (bundle, {}))
    result = read_result(output)
    assert result["model_initialization_attempts"] == result["model_initializations"] == 0
    assert result["model_requests_attempted"] == result["denoise_calls_attempted"] == 0
    assert not result["passed"]


def test_wrong_gpu_is_rejected_before_loader(tmp_path, monkeypatch):
    items = setup(tmp_path)
    environment(monkeypatch)
    calls = []
    with pytest.raises(ValueError, match="registered GPU"):
        worker.run_model_worker(output=items[0], gpu_uuid="GPU-not-registered", loader=lambda: calls.append(True))
    assert calls == []


def test_image_directory_need_not_exist_before_model_initialization(tmp_path, monkeypatch):
    items = setup(tmp_path)
    output, registration, image_root = items[:3]
    (image_root / "current.png").unlink()
    image_root.rmdir()
    environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", admission.GPU)
    bundle = {
        "model_registration": registration,
        "prepared": {"model_config": {"model_python": "/not/the/current/python"}},
    }
    with pytest.raises(ValueError, match="Model Python differs"):
        worker.run_model_worker(output=output, gpu_uuid=admission.GPU, loader=lambda: (bundle, {}))
    result = read_result(output)
    assert result["sampling_seed"] == 43 and result["model_initialization_attempts"] == 0
    assert not image_root.exists()


@pytest.mark.parametrize("scenario", ["request", "stop", "partial_denoise"])
def test_worker_serves_seed43_or_accounts_for_bounded_stop(tmp_path, monkeypatch, scenario):
    items = setup(tmp_path)
    output, registration, _image_root, run_dir, request = items[:5]
    run_dir.rmdir()
    environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", admission.GPU)
    source = output / "source"
    source.mkdir()
    checkpoint = {"run": str(output / "weights"), "iteration": "iter_000008400"}
    expected_model = str(Path(checkpoint["run"]) / "checkpoints/iter_000008400/model")
    probe_path = output / "probe.json"
    probe_path.write_text("{}", encoding="utf-8")
    bundle = {
        "model_registration": registration,
        "prepared": {
            "model_config": {
                "model_python": sys.executable,
                "cosmos_source_root": str(source),
                "checkpoint": checkpoint,
            }
        },
        "template": template(),
        "bridge": {},
    }
    calls, evidence, counter = [], [], Counter()
    fake_random = SimpleNamespace(seed=lambda value: calls.append(("python", value)))
    monkeypatch.setattr(worker, "random", fake_random)
    monkeypatch.setattr(legacy, "random", fake_random)
    monkeypatch.setattr(np.random, "seed", lambda value: calls.append(("numpy", value)))
    monkeypatch.setattr(worker.live_cv2_admission, "admit_model_cv2", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "path", list(sys.path))

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "synthetic"
    fake_torch.version = SimpleNamespace(cuda="synthetic")
    fake_torch.manual_seed = lambda value: calls.append(("torch", value))
    fake_torch.inference_mode = nullcontext
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(uuid=admission.GPU, name="synthetic"),
        manual_seed_all=lambda value: calls.append(("cuda", value)),
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
            return {}, {"ai_caption": [json.dumps(dict(task_title=task, command=task, task_plan=[]))]}

        def execute(self, _sample, _batch):
            count = 3 if scenario == "partial_denoise" else 35
            counter.calls.extend({"completed": True} for _ in range(count))
            if scenario == "partial_denoise":
                raise RuntimeError("synthetic partial denoise")
            evidence.append(dict(stage="initial_noise", inactive_rms_before=1.0, inactive_rms_after=0.0))
            action = [[0.0] * 128 for _ in range(40)]
            identity = dict(
                seed=43,
                action_fps=20,
                checkpoint_path=expected_model,
                action_units="raw_physical_or_source_units",
                action_dim_mask=[True] * 10 + [False] * 118,
                normalized_action=action,
            )
            response = dict(
                status="success",
                args=copy.deepcopy(self.request),
                outputs=[dict(content=dict(action=action, action128=identity))],
            )
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
        "scripts.wam.training.clean.qualify_wam_clean_bridge", imported_cosmos_sources=lambda _bridge: ["synthetic"]
    )
    loads = []

    def loader():
        loads.append(True)
        return bundle, {"cv2_probe_ref": worker.prefix.reference(probe_path)}

    left, right = socket.socketpair()
    channel = FramedJSONChannel(left, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    descriptor = right.detach()
    monkeypatch.setenv("S2A_CHANNEL_FD", str(descriptor))
    try:
        channel.send(make_stop(session=admission.SESSION, next_step=8) if scenario == "stop" else request)
        if scenario == "partial_denoise":
            with pytest.raises(RuntimeError, match="synthetic partial denoise"):
                worker.run_model_worker(output=output, gpu_uuid=admission.GPU, loader=loader)
        else:
            worker.run_model_worker(output=output, gpu_uuid=admission.GPU, loader=loader)
        response = channel.receive() if scenario == "request" else None
    finally:
        channel.close()
        try:
            os.close(descriptor)
        except OSError:
            pass

    result = read_result(output)
    initial_seeds = [(name, 43) for name in ("python", "numpy", "torch", "cuda")]
    assert calls[:5] == initial_seeds + [("load", None)]
    assert result["sampling_seed"] == 43 and result["model_initializations"] == 1
    if scenario == "request":
        assert calls[5:] == initial_seeds and loads == [True, True]
        assert response["seed"] == service.request["seed"] == 43
        assert result["passed"] and result["responses_sent"] == 1 and result["denoise_calls_completed"] == 35
    elif scenario == "stop":
        assert calls[5:] == [] and loads == [True, True] and service.request is None
        assert result["passed"] and result["responses_sent"] == result["denoise_calls_completed"] == 0
        assert result["model_sequence_stop"]["next_step"] == 8
    else:
        assert calls[5:] == initial_seeds and loads == [True]
        assert not result["passed"] and result["responses_sent"] == 0
        assert result["model_requests_attempted"] == 1 and result["model_requests_completed"] == 0
        assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 3
