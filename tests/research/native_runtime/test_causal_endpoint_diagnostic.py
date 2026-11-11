"""Exercise the CPU diagnostic's evidence accounting without new physics."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import causal_endpoint_diagnostic_admission as admission
from scripts.simulation.runtime import run_causal_endpoint_diagnostic as diagnostic


def replay_receipt():
    inputs = admission.load_inputs()
    runtime = copy.deepcopy(inputs["branch"]["runtime"])
    runtime["newly_selected_commands"] = 0
    runtime["limits"]["mj_step"] = 3000
    runtime["error"]["message"] = "Replay endpoint observation changed the physical state"
    runtime["streaming_replay_partial"] = {
        "segments": copy.deepcopy(inputs["stream"]["segments"]),
        "control_steps_attempted": 40,
        "control_steps_completed": 40,
    }
    events = [
        {"step": 120, "completed": True, "changed_indices": []},
        {"step": 120, "completed": True, "changed_indices": [2]},
    ]
    return inputs, runtime, events


def test_complete_action_replay_can_reproduce_the_endpoint_change():
    inputs, runtime, events = replay_receipt()
    result = diagnostic.compare_replay(runtime, inputs["stream"], events, reproduced=True)
    assert result["reproduced"] and result["completed_actions"] == 40 and result["native_steps"] == 3000
    assert result["max_difference"] == {"state": 0.0, "q": 0.0, "qvel": 0.0, "time": 0.0}


def test_unmatched_independent_sample_is_rejected():
    inputs, runtime, events = replay_receipt()
    runtime["streaming_replay_partial"]["segments"][0]["continuation"]["periods"][0]["actual_inner"][0]["q"][0] += 0.1
    with pytest.raises(ValueError, match="replay samples differ"):
        diagnostic.compare_replay(runtime, inputs["stream"], events, reproduced=True)


def test_missing_observation_change_cannot_be_called_reproduced():
    inputs, runtime, events = replay_receipt()
    events[-1]["changed_indices"] = []
    with pytest.raises(ValueError, match="must reproduce"):
        diagnostic.compare_replay(runtime, inputs["stream"], events, reproduced=True)


def test_dry_admission_launches_no_model_or_physics(tmp_path, monkeypatch):
    dry = tmp_path / "dry.json"
    monkeypatch.setattr(diagnostic, "DRY", dry)
    diagnostic.dry_run()
    result = json.loads(dry.read_text())
    assert result["complete"] and result["passed"]
    assert result["new_model_requests"] == result["new_denoise_calls"] == result["new_physics_steps"] == 0


def test_diagnostic_import_does_not_load_candidate_policies():
    code = """
from scripts.simulation.runtime import run_causal_endpoint_diagnostic as diagnostic
diagnostic.prior.checker_isolation()
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
