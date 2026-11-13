"""Route case identity, state, model and replay through the shared worker."""

import json
import os
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.simulation.runtime import reset_multitask_adaptive_worker as worker


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return worker.prefix.reference(path)


def setup_trial(tmp_path, phase, seed):
    output = tmp_path / "trial"
    (output / phase / "simulation").mkdir(parents=True)
    inputs = {
        "config": {"seed": seed, "max_control_steps_per_phase": 600},
        "case": {
            "case": {"sampling_seed": seed},
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
    receipt, sources, bundle = {"case": seed}, [{"source": "bound"}], {"registered": "bundle"}
    trial = SimpleNamespace(
        output=output,
        check_preflight=lambda _inputs: None,
        input_receipt=lambda _inputs: receipt,
        source_refs=lambda: sources,
        model_bundle=lambda _inputs: bundle,
        model_registration=lambda _inputs, root: {"seed": seed, "image_root": str(root)},
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
        model = write(output / "producer/model/result.json", {"complete": True, "passed": True})
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
    return trial, inputs


def test_worker_import_keeps_simulator_and_candidate_unloaded(tmp_path):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    for name in worker.prefix.THREAD_VARS:
        environment[name] = "1"
    environment["TMPDIR"] = str(tmp_path)
    code = (
        "from scripts.simulation.runtime import reset_multitask_adaptive_worker as w; "
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
@pytest.mark.parametrize("seed", [42, 43])
def test_worker_routes_registered_case_and_600_period_budget(tmp_path, monkeypatch, phase, seed):
    trial, inputs = setup_trial(tmp_path, phase, seed)
    monkeypatch.setattr(worker.admission, "make_trial", lambda _index: trial)
    monkeypatch.setattr(worker.admission, "load_inputs", lambda _index, **_kwargs: inputs)
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda *_args: {"passed": True})
    monkeypatch.setattr(worker.original, "require_no_simulator_import", lambda: None)
    monkeypatch.setattr(worker.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(worker.isolation, "FORBIDDEN", ("synthetic.candidate",))
    if phase == "checker":
        # Other co-collected tests import these real candidate modules at module
        # scope. This routing test uses a synthetic stream; the clean-process
        # isolation test above exercises the real checker import boundary.
        for name in (
            "s2a.research.joint_control.preview.causal_reference_replan",
            "s2a.research.wam.tracking.runtime.adaptive_stream_reference",
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(worker.audit, "verify_model_exchange", lambda *_args: {})
    monkeypatch.setattr(worker.historical, "load_recorded_responses", lambda *_args: ["actual-response"])
    calls = []

    class Adapter:
        def __init__(self, **_kwargs):
            pass

        def run_feedback(self, *_args, registered_seed, continuation, max_control_steps):
            calls.append(("seed-budget", registered_seed, max_control_steps))
            continuation("env", "terminal", "observers", "receipt", progress="progress")
            return {"complete": True, "passed": True}

    adapter_module = ModuleType("s2a.research.native_runtime.reset_bounded_runtime")
    adapter_module.ResetBoundedRuntime = Adapter
    monkeypatch.setitem(sys.modules, adapter_module.__name__, adapter_module)
    implementation = ModuleType("scripts.simulation.runtime.reset_adaptive_stream")

    def execute(*_args, **kwargs):
        calls.append((phase, kwargs))
        return {}

    setattr(implementation, "run_producer", execute)
    setattr(implementation, "run_checker", execute)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    import scripts.simulation.runtime as runtime_package

    monkeypatch.setattr(runtime_package, "reset_adaptive_stream", implementation, raising=False)
    left = right = None
    if phase == "producer":
        left, right = socket.socketpair()
        monkeypatch.setenv("S2A_CHANNEL_FD", str(right.detach()))
    try:
        worker.run_worker(0, phase)
    finally:
        if left is not None:
            left.close()
    result = json.loads((trial.output / phase / "simulation/result.json").read_text())
    assert result["complete"] and result["passed"] and calls[0] == ("seed-budget", seed, 600)
    options = calls[1][1]
    assert options["registration"]["image_root"] == str(trial.output / phase / "simulation")
    assert options["progress"] == "progress" and options["case"] == inputs["case"]
    if phase == "producer":
        assert options["baseline"] == inputs["baseline_periods"] and options["channel"].closed
    else:
        assert options["producer"]["complete"] and options["responses"] == ["actual-response"]
        assert result["candidate_modules_unloaded"] is True


def test_changed_registration_is_rejected_before_physics(tmp_path, monkeypatch):
    trial, inputs = setup_trial(tmp_path, "producer", 43)
    monkeypatch.setattr(worker.admission, "make_trial", lambda _index: trial)
    monkeypatch.setattr(worker.admission, "load_inputs", lambda _index, **_kwargs: inputs)
    path = trial.output / "producer/registration.json"
    registration = json.loads(path.read_text())
    registration["inputs"] = {"case": 42}
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(ValueError, match="registered case"):
        worker.run_worker(0, "producer")
    result = json.loads((trial.output / "producer/simulation/result.json").read_text())
    assert not result["complete"] and result["error"]["type"] == "ValueError"
