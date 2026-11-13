"""Verify seed-43 producer/checker wiring without creating a physics model."""

import json
import os
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.simulation.runtime import reset_seed43_simulation_worker as worker


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return worker.prefix.reference(path)


def setup_trial(tmp_path, phase):
    output = tmp_path / "trial"
    (output / phase / "simulation").mkdir(parents=True)
    inputs = {
        "case": {
            "case": {"sampling_seed": 43},
            "expected": {},
            "expected_receipt": {},
            "chunk": {},
            "controller": {},
            "tool": {},
            "config": {},
            "record": {},
            "record_xml": "<synthetic/>",
        },
        "baseline_periods": ["historical-period"] * 8,
        "base": {"cv2_admission": {"cv2_probe_ref": {}, "cv2_runtime_source_refs": []}, "asset_refs": []},
    }
    receipt, sources, bundle = {"case": "seed43"}, [{"source": "bound"}], {"registered": "bundle"}
    trial = SimpleNamespace(
        output=output,
        gpu_uuid="GPU-registered",
        load_inputs=lambda **_kwargs: inputs,
        check_preflight=lambda _inputs: None,
        input_receipt=lambda _inputs: receipt,
        source_refs=lambda: sources,
        model_bundle=lambda _inputs: bundle,
        model_registration=lambda _inputs, root: {"seed": 43, "image_root": str(root)},
    )
    registration = dict(
        phase=phase,
        inputs=receipt,
        source_refs=sources,
        model_bundle_ref=None,
        producer_branch_ref=None,
        producer_model_ref=None,
    )
    if phase == "producer":
        registration["model_bundle_ref"] = write(output / "producer/model/registered_inputs.json", bundle)
    else:
        branch = write(output / "producer/simulation/result.json", {"complete": True, "passed": True})
        response = write(output / "producer/model/response.json", {"seed": 43})
        model = write(
            output / "producer/model/result.json",
            dict(
                complete=True,
                passed=True,
                gpu_uuid=trial.gpu_uuid,
                model_initialization_attempts=1,
                model_initializations=1,
                model_requests_attempted=1,
                model_requests_completed=1,
                denoise_calls_attempted=35,
                denoise_calls_completed=35,
                responses_sent=1,
                responses=[dict(actual_response_ref=response)],
            ),
        )
        write(
            output / "producer/result.json",
            dict(
                complete=True,
                passed=True,
                sources_unchanged=True,
                gpu_released=True,
                branch_ref=branch,
                model_ref=model,
            ),
        )
        registration.update(producer_branch_ref=branch, producer_model_ref=model)
    write(output / phase / "registration.json", registration)
    return trial, inputs, registration


def test_import_for_checker_keeps_simulator_and_candidate_unloaded(tmp_path):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    for name in worker.prefix.THREAD_VARS:
        environment[name] = "1"
    environment["TMPDIR"] = str(tmp_path)
    code = (
        "from scripts.simulation.runtime import reset_seed43_simulation_worker as w; "
        "w.isolation.checker_isolation(); w.original.require_no_simulator_import()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=worker.prefix.REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("phase", ["producer", "checker"])
def test_worker_routes_actual_registration_and_corrected_checker(tmp_path, monkeypatch, phase):
    trial, inputs, _registration = setup_trial(tmp_path, phase)
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda *_args: {"passed": True})
    monkeypatch.setattr(worker.original, "require_no_simulator_import", lambda: None)
    monkeypatch.setattr(worker.isolation, "FORBIDDEN", ("synthetic.candidate",))
    calls = []

    class Adapter:
        def __init__(self, **_kwargs):
            pass

        def run_feedback(self, *_args, registered_seed, continuation):
            calls.append(("seed", registered_seed))
            continuation("env", "terminal", "observers", "receipt", progress="progress")
            return {"control_steps_completed": 1}

    adapter_module = ModuleType("s2a.research.native_runtime.reset_two_window_runtime")
    adapter_module.ResetTwoWindowRuntime = Adapter
    monkeypatch.setitem(sys.modules, adapter_module.__name__, adapter_module)
    name = (
        "scripts.simulation.runtime.reset_live_two_window"
        if phase == "producer"
        else "scripts.simulation.runtime.reset_live_two_window_replay_v2"
    )
    implementation = ModuleType(name)

    def execute(*_args, **kwargs):
        calls.append((phase, kwargs))
        return {}

    setattr(implementation, "run_producer" if phase == "producer" else "run_checker", execute)
    monkeypatch.setitem(sys.modules, name, implementation)
    left = right = None
    if phase == "producer":
        left, right = socket.socketpair()
        monkeypatch.setenv("S2A_CHANNEL_FD", str(right.detach()))
    try:
        worker.run_worker(trial, phase)
    finally:
        if left is not None:
            left.close()
    result = json.loads((trial.output / phase / "simulation/result.json").read_text())
    assert result["complete"] and result["passed"] and calls[0] == ("seed", 43)
    options = calls[1][1]
    assert options["registration"]["image_root"] == str(trial.output / "producer/simulation/runtime")
    assert options["progress"] == "progress" and options["case"] == inputs["case"]
    if phase == "producer":
        assert options["baseline"] == inputs["baseline_periods"] and options["channel"].closed
    else:
        assert options["producer"]["complete"] and options["model_response"] == {"seed": 43}
        assert result["candidate_modules_unloaded"] is True


@pytest.mark.parametrize("field", ["inputs", "source_refs", "model_bundle_ref"])
def test_changed_producer_registration_is_rejected_before_runtime(tmp_path, field):
    trial, _inputs, registration = setup_trial(tmp_path, "producer")
    registration[field] = None
    path = trial.output / "producer/registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        worker.worker_inputs(trial, "producer")
    assert not (trial.output / "producer/simulation/result.json").exists()


def test_checker_rejects_producer_receipt_substitution(tmp_path):
    trial, _inputs, registration = setup_trial(tmp_path, "checker")
    registration["producer_branch_ref"] = registration["producer_model_ref"]
    (trial.output / "checker/registration.json").write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(ValueError, match="exact model and branch receipts"):
        worker.worker_inputs(trial, "checker")


def test_partial_runtime_receipt_is_preserved_on_physics_failure(tmp_path, monkeypatch):
    from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure

    trial, _inputs, _registration = setup_trial(tmp_path, "producer")
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda *_args: {})
    monkeypatch.setattr(worker.original, "require_no_simulator_import", lambda: None)
    module = ModuleType("s2a.research.native_runtime.reset_two_window_runtime")

    class Adapter:
        def __init__(self, **_kwargs):
            pass

        def run_feedback(self, *_args, **_kwargs):
            raise PrefixReplayFailure(
                "synthetic physical stop", {"control_steps_attempted": 3, "control_steps_completed": 2}
            )

    module.ResetTwoWindowRuntime = Adapter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    left, right = socket.socketpair()
    monkeypatch.setenv("S2A_CHANNEL_FD", str(right.detach()))
    try:
        with pytest.raises(PrefixReplayFailure, match="synthetic physical stop"):
            worker.run_worker(trial, "producer")
    finally:
        left.close()
    result = json.loads((trial.output / "producer/simulation/result.json").read_text())
    assert not result["complete"] and not result["passed"]
    assert result["runtime"] == {"control_steps_attempted": 3, "control_steps_completed": 2}
    assert result["error"]["type"] == "PrefixReplayFailure"
