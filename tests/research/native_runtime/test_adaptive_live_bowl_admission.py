"""Exercise the frozen live-trial admission without starting a model or simulator."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_admission as admission


@pytest.fixture
def config():
    return json.loads(admission.CONFIG.read_text())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("case_id", "different"),
        ("task", "different"),
        ("seed", 43),
        ("prefix_steps", 108),
        ("gpu_uuid", "GPU-other"),
        ("max_model_initializations", 2),
        ("max_model_requests", 3),
        ("denoise_calls_per_request", 34),
        ("max_adaptive_periods", 9),
        ("max_control_steps_per_phase", 116),
        ("max_native_steps_per_phase", 2900),
        ("previous_control_steps", 1530),
        ("fresh_cases", 1),
        ("automatic_retry", True),
    ],
)
def test_changed_case_device_or_budget_is_rejected(config, key, value):
    changed = copy.deepcopy(config)
    changed[key] = value
    with pytest.raises(ValueError, match="budgets must remain exact"):
        admission.validate_config(changed)


def test_config_fingerprint_is_checked_before_loading_historical_inputs(tmp_path, monkeypatch):
    altered = tmp_path / "config.json"
    altered.write_text(admission.CONFIG.read_text() + " ")
    monkeypatch.setattr(admission, "CONFIG", altered)
    with pytest.raises(ValueError, match="Changed input"):
        admission.load_inputs()


def test_actual_saved_inputs_and_phase_local_registration_bind_exact_history():
    value = admission.load_inputs()
    receipt = admission.input_receipt(value)
    assert value["initial"]["step"] == 107
    assert value["initial"]["state_sha256"] == value["config"]["initial_state_sha256"]
    assert value["expected_capture"]["canvas_png"]["sha256"] == value["config"]["initial_image_sha256"]
    assert len(value["model_source_hashes"]) >= 100 and len(value["weight_hashes"]) == 163
    assert receipt["refs"]["prior_checker"] == admission.PRIOR_CHECKER
    registration = admission.model_registration(value, admission.OUTPUT / "producer/simulation")
    assert registration["initial"] == value["initial"]
    assert registration["max_requests"] == 2 and registration["max_periods"] == 8
    assert registration["denoise_steps"] == 35 and registration["seed"] == 42
    assert registration["model_sha256"] == value["config"]["model_sha256"]
    with pytest.raises(ValueError, match="inside this trial"):
        admission.model_registration(value, admission.ROOT / "outputs/unrelated")


def test_admission_import_and_loading_do_not_load_candidate_policy():
    code = """
import sys
from scripts.simulation.runtime.adaptive_live_bowl_admission import load_inputs
load_inputs()
for module in (
    's2a.research.joint_control.preview.causal_reference_replan',
    's2a.research.joint_control.preview.loaded_gripper_feedback',
    's2a.research.wam.tracking.runtime.adaptive_stream_reference',
):
    assert module not in sys.modules, module
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
