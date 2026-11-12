"""Test model-worker admission and termination without loading a GPU model."""

import json
import signal
from pathlib import Path

import pytest

from s2a.infrastructure.repository_paths import repository_root
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview.adaptive_model_worker import run_model_worker, validate_registration

GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"


def directories(tmp_path):
    output = tmp_path / "trial"
    (output / "producer/model").mkdir(parents=True)
    (output / "producer/simulation").mkdir(parents=True)
    return output


def environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for key in prefix.THREAD_VARS:
        monkeypatch.setenv(key, "1")


def bundle(output):
    return {
        "model_registration": {
            "session": "synthetic-worker",
            "initial": {"step": 107, "sim_time_s": 5.6, "state_sha256": "a" * 64},
            "image_root": str((output / "producer/simulation").relative_to(repository_root())),
            "task": "put the bowl on the plate",
            "seed": 42,
            "model_sha256": "b" * 64,
            "max_requests": 2,
            "max_periods": 8,
            "denoise_steps": 35,
        },
        "prepared": {"model_config": {"model_python": "/not/the/current/python"}},
    }


def read_result(output):
    return json.loads((output / "producer/model/result.json").read_text())


def test_wrong_gpu_mask_stops_before_loading_any_model(tmp_path, monkeypatch):
    output = directories(tmp_path)
    environment(monkeypatch)
    calls = []
    handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="registered UUID"):
        run_model_worker(output=output, gpu_uuid=GPU, loader=lambda: calls.append(True))
    result = read_result(output)
    assert calls == [] and result["model_initialization_attempts"] == result["model_initializations"] == 0
    assert result["model_requests_attempted"] == result["denoise_calls_attempted"] == 0
    assert not result["complete"] and not result["passed"]
    assert signal.getsignal(signal.SIGTERM) is handler


@pytest.mark.parametrize("change", ["requests", "periods", "seed", "denoise", "task", "image_root"])
def test_unregistered_model_budget_or_input_stops_before_initialization(tmp_path, change):
    output = directories(tmp_path)
    value = bundle(output)
    registration = value["model_registration"]
    if change == "requests":
        registration["max_requests"] = 0
    elif change == "periods":
        registration["max_periods"] = 494
    elif change == "seed":
        registration["seed"] = 43
    elif change == "denoise":
        registration["denoise_steps"] = 34
    elif change == "task":
        registration["task"] = "not registered"
    else:
        registration["image_root"] = str(output.relative_to(repository_root()))
    with pytest.raises(ValueError, match="bounded request"):
        validate_registration(registration, output)


def test_registered_budget_can_be_validated_without_gpu_use(tmp_path):
    output = directories(tmp_path)
    assert not (output / "producer/simulation/runtime").exists()
    admission = validate_registration(bundle(output)["model_registration"], output)
    assert admission.phase == "ready" and admission.admitted_requests == 0


def test_worker_refuses_outputs_outside_repository(tmp_path):
    with pytest.raises(ValueError, match="existing local model output"):
        run_model_worker(output=Path("/tmp/not-a-registered-model-output"), gpu_uuid=GPU, loader=lambda: None)
