"""Verify adaptive worker admission and partial receipt preservation on CPU."""

import json
import os
import subprocess
import sys

import pytest

from s2a.infrastructure.lifecycle.interruptible_worker import WorkerTermination
from s2a.research.native_runtime.development_prefix_runtime_bounded import PrefixReplayFailure
from scripts.simulation.runtime import managed_adaptive_simulation as worker
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def test_changed_registration_fails_before_simulator_or_model_start(tmp_path):
    output = tmp_path / "trial"
    (output / "producer/simulation").mkdir(parents=True)
    write(output / "producer/registration.json", dict(phase="producer", inputs={"different": 1}, source_refs=[]))
    with pytest.raises(ValueError, match="exact registered inputs and sources"):
        worker.run_adaptive_simulation_worker(
            "producer",
            output=output,
            load_inputs=lambda: {},
            input_receipt=lambda _inputs: {"expected": 1},
            source_refs=lambda: [],
            model_registration=lambda *_args: {},
        )
    result = json.loads((output / "producer/simulation/result.json").read_text())
    assert not result["complete"] and result["error"]["type"] == "ValueError" and "runtime" not in result


def test_interrupted_runtime_receipt_is_published_by_the_managed_worker(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    (output / "checker/simulation").mkdir(parents=True)
    producer_ref = write(output / "producer_branch.json", dict(complete=True, passed=True))
    model_ref = write(output / "producer_model.json", dict(complete=True, passed=True))
    write(
        output / "checker/registration.json",
        dict(
            phase="checker",
            inputs={"case": "registered"},
            source_refs=[],
            producer_branch_ref=producer_ref,
            producer_model_ref=model_ref,
        ),
    )
    case = dict(
        expected={}, expected_receipt={}, chunk={}, controller={}, tool={}, config={}, record={}, record_xml="", case={}
    )
    old_runtime = dict(cv2_admission=dict(cv2_probe_ref={}, cv2_runtime_source_refs=[]), asset_refs=[])
    base = {"base": {"case": case, "base": {"previous": {"base": old_runtime}}}}
    inputs = dict(
        prior={"base": {"source": {"previous": base}}},
        config={"seed": 42, "max_control_steps_per_phase": 115},
    )
    partial = dict(
        complete=False,
        control_steps_attempted=8,
        control_steps_completed=7,
        counters={"mj_step": dict(attempted=198, completed=197)},
    )

    class FakeRuntime:
        def __init__(self, **_options):
            pass

        def run_feedback(self, *_args, **_options):
            try:
                raise WorkerTermination("synthetic SIGTERM")
            except WorkerTermination as error:
                raise PrefixReplayFailure("interrupted runtime", partial) from error

    from s2a.research.native_runtime import reset_bounded_runtime

    monkeypatch.setattr(reset_bounded_runtime, "ResetBoundedRuntime", FakeRuntime)
    monkeypatch.setattr(worker.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(worker.historical, "load_recorded_responses", lambda *_args: [])
    monkeypatch.setattr(worker.historical, "prefix_options", lambda *_args: (None, {}))
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda *_args: {})
    monkeypatch.setattr(worker.original, "require_no_simulator_import", lambda: None)
    with pytest.raises(PrefixReplayFailure, match="interrupted runtime"):
        worker.run_adaptive_simulation_worker(
            "checker",
            output=output,
            load_inputs=lambda: inputs,
            input_receipt=lambda _inputs: {"case": "registered"},
            source_refs=lambda: [],
            model_registration=lambda *_args: {},
        )
    result = json.loads((output / "checker/simulation/result.json").read_text())
    assert not result["complete"] and result["termination_requested"] and result["runtime"] == partial


def test_checker_connects_registered_replay_to_the_managed_runtime(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    directory = output / "checker/simulation"
    directory.mkdir(parents=True)
    producer_ref = write(output / "producer_branch.json", dict(complete=True, passed=True))
    model_ref = write(output / "producer_model.json", dict(complete=True, passed=True))
    write(
        output / "checker/registration.json",
        dict(
            phase="checker",
            inputs={"case": "registered"},
            source_refs=[],
            producer_branch_ref=producer_ref,
            producer_model_ref=model_ref,
        ),
    )
    case = dict(
        expected={},
        expected_receipt={},
        chunk={},
        controller={},
        tool={},
        config={},
        record={},
        record_xml="",
        case={"case_id": "registered"},
    )
    old_runtime = dict(cv2_admission=dict(cv2_probe_ref={}, cv2_runtime_source_refs=[]), asset_refs=[])
    base = {"base": {"case": case, "base": {"previous": {"base": old_runtime}}}}
    inputs = dict(
        prior={"base": {"source": {"previous": base}}},
        config={"seed": 42, "max_control_steps_per_phase": 115},
        expected_capture={"step": 107},
    )
    observed = {}

    class FakeRuntime:
        def __init__(self, **_options):
            pass

        def run_feedback(self, *_args, continuation, registered_seed, max_control_steps):
            assert registered_seed == 42 and max_control_steps == 115
            assert continuation("env", "terminal", "observers", "receipt") == {"replayed": True}
            return dict(complete=True, passed=True, newly_selected_commands=0)

    def replay(_env, _terminal, _observers, _receipt, **options):
        observed.update(options)
        return {"replayed": True}

    from s2a.research.native_runtime import reset_bounded_runtime
    from scripts.simulation.runtime import adaptive_live_bowl

    monkeypatch.setattr(reset_bounded_runtime, "ResetBoundedRuntime", FakeRuntime)
    monkeypatch.setattr(adaptive_live_bowl, "run_checker", replay)
    monkeypatch.setattr(worker.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(worker.historical, "load_recorded_responses", lambda *_args: ["recorded"])
    monkeypatch.setattr(worker.historical, "prefix_options", lambda *_args: ("prefix_runner", {}))
    monkeypatch.setattr(worker, "restore_cv2_environment", lambda *_args: {"passed": True})
    monkeypatch.setattr(worker.original, "require_no_simulator_import", lambda: None)
    forbidden = tuple(worker.isolation.FORBIDDEN) + (
        "s2a.research.joint_control.preview.causal_reference_replan",
        "s2a.research.wam.tracking.runtime.adaptive_stream_reference",
    )
    for name in forbidden:
        monkeypatch.delitem(sys.modules, name, raising=False)
    result = worker.run_adaptive_simulation_worker(
        "checker",
        output=output,
        load_inputs=lambda: inputs,
        input_receipt=lambda _inputs: {"case": "registered"},
        source_refs=lambda: [],
        model_registration=lambda _inputs, _directory: {"session": "registered"},
    )
    assert result["complete"] and result["candidate_modules_unloaded"]
    assert result["runtime"]["newly_selected_commands"] == 0
    assert observed["prefix_runner"] == "prefix_runner" and observed["responses"] == ["recorded"]
    assert observed["registration"] == {"session": "registered"} and observed["runtime_output"] == directory / "runtime"
    assert json.loads((directory / "result.json").read_text())["case"]["case_id"] == "registered"


def test_checker_import_does_not_load_candidate_or_simulator_modules():
    script = """\
import sys
import scripts.simulation.runtime.managed_adaptive_simulation
for name in ('mujoco', 'robosuite', 'libero', 'torch', 'cosmos_framework',
             's2a.research.joint_control.preview.causal_reference_replan',
             's2a.research.wam.tracking.runtime.adaptive_stream_reference'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
