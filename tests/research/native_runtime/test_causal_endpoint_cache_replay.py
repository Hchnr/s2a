"""Check the separately registered endpoint-cache replay without physics."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_endpoint_cache_replay as runner
from scripts.verification.reference_ports import audit_causal_endpoint_diagnostic as audit


def simulated_summary():
    inputs = runner.load_inputs()
    diagnostic = prefix.read(audit.BRANCH)
    runtime = copy.deepcopy(diagnostic["runtime"])
    runtime["complete"] = runtime["passed"] = True
    runtime["continuation"] = runtime["streaming_replay_partial"]
    runtime["continuation"]["complete"] = True
    events = [{"kind": "observed", "step": step} for step in range(81, 121)]
    events += [{"kind": "reused", "step": step} for step in (88, 96, 104, 112, 120, 120)]
    final = inputs["base"]["stream"]["segments"][-1]["continuation"]["periods"][-1]["endpoint_receipt"]
    capture = {
        "frame_index": 120,
        "current_endpoint": {"sim_state_sha256": final["sim_state_sha256"]},
        "render_before": {"sim_state_sha256": final["sim_state_sha256"]},
        "render_after": {"sim_state_sha256": final["sim_state_sha256"]},
        "future_rgb_frames": 0,
    }
    return inputs, runtime, events, capture


def test_registration_binds_the_completed_prior_diagnostic():
    inputs = runner.load_inputs()
    assert inputs["config"]["max_native_steps"] == 3000
    assert inputs["base"]["stream"]["diagnostic_derived_stop"] is True
    assert inputs["audit_ref"]["sha256"] == runner.FIXED["diagnostic_audit_sha256"]


def test_summarizer_requires_all_observed_and_reused_boundaries():
    inputs, runtime, events, capture = simulated_summary()
    result = runner.compare_replay(runtime, inputs["base"]["stream"], events, capture)
    assert result["completed_actions"] == 40 and result["native_steps"] == 3000
    assert all(value == 0 for value in result["max_difference"].values())
    events[-1]["step"] = 119
    with pytest.raises(ValueError, match="cached replay must complete"):
        runner.compare_replay(runtime, inputs["base"]["stream"], events, capture)


def test_changed_capture_state_is_rejected():
    inputs, runtime, events, capture = simulated_summary()
    capture["render_after"]["sim_state_sha256"] = "different"
    with pytest.raises(ValueError, match="cached replay must complete"):
        runner.compare_replay(runtime, inputs["base"]["stream"], events, capture)


@pytest.mark.parametrize(("key", "value"), [("max_native_steps", 3025), ("new_model_requests", 1), ("seed", 43)])
def test_unregistered_budget_is_rejected(key, value):
    config = copy.deepcopy(runner.FIXED)
    config[key] = value
    with pytest.raises(ValueError, match="physical budget"):
        runner.validate_config(config)


def test_dry_run_uses_no_model_or_physics(tmp_path, monkeypatch):
    destination = tmp_path / "dry.json"
    monkeypatch.setattr(runner, "DRY", destination)
    runner.dry_run()
    result = json.loads(destination.read_text())
    assert result["complete"] and result["passed"]
    assert result["new_model_requests"] == result["new_denoise_calls"] == result["new_physics_steps"] == 0


def test_checker_entry_does_not_load_candidate_policies():
    code = """
from scripts.simulation.runtime import run_causal_endpoint_cache_replay as runner
runner.prior.checker_isolation()
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
