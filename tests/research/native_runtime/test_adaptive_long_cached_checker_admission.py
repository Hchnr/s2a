"""Verify the cached checker binds the real failure and unchanged budgets."""

import json

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_long_admission as previous
from scripts.simulation.runtime import adaptive_long_cached_checker_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def configs():
    return json.loads(admission.CONFIG.read_text()), json.loads(previous.CONFIG.read_text())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("case_id", "another_case"),
        ("session", "new_session"),
        ("max_control_steps", 148),
        ("max_native_steps", 3674),
        ("max_models", 4),
        ("new_model_requests", 1),
        ("new_denoise_calls", False),
        ("automatic_retry", True),
        ("failed_checker_sha256", "0" * 64),
        ("previous_physical_bounds", {"control_steps": {"lower": 2055, "upper": 2055}}),
    ],
)
def test_changed_cached_checker_registration_is_rejected(key, value):
    config, old = configs()
    admission.validate_config(config, old)
    config[key] = value
    with pytest.raises(ValueError, match="identity, accumulated costs and limits"):
        admission.validate_config(config, old)


def test_actual_failed_replay_and_saved_audit_bind_new_capture_root():
    inputs = admission.load_inputs()
    registration = admission.replay_registration(inputs, admission.OUTPUT / "checker/simulation")
    assert registration["session"] == previous.SESSION
    assert registration["image_root"] == str((admission.OUTPUT / "checker/simulation").relative_to(admission.ROOT))
    assert inputs["previous_physical_bounds"]["control_steps"] == dict(lower=2055, upper=2170)
    assert inputs["previous_model_bounds"]["requests_completed"] == dict(lower=27, upper=27)
    assert inputs["failed_checker"]["complete"] is False
    with pytest.raises(ValueError, match="capture root"):
        admission.replay_registration(inputs, previous.OUTPUT / "checker/simulation")


def test_changed_partial_audit_cannot_be_admitted(monkeypatch):
    read = prefix.read

    def changed(ref, seen=None):
        result = read(ref, seen)
        if ref == admission.PARTIAL_AUDIT:
            result["details"]["trailing_success_periods"] += 1
        return result

    monkeypatch.setattr(prefix, "read", changed)
    with pytest.raises(ValueError, match="exact recomputed partial audit"):
        admission.load_inputs()
