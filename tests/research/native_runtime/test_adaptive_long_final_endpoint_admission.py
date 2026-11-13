"""Verify final-endpoint replay binds both failures and the new budget."""

import json

import pytest

from scripts.simulation.runtime import adaptive_long_final_endpoint_admission as admission


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cache_step", 111),
        ("max_control_steps", 148),
        ("max_native_steps", 3674),
        ("max_models", 4),
        ("new_model_requests", 1),
        ("automatic_retry", True),
        ("cached_failed_checker_sha256", "0" * 64),
        ("cached_incident_audit_sha256", "0" * 64),
        ("previous_physical_bounds", {"control_steps": {"lower": 2174, "upper": 2174}}),
    ],
)
def test_changed_final_endpoint_registration_is_rejected(key, value):
    config = json.loads(admission.CONFIG.read_text())
    previous = json.loads(admission.previous.CONFIG.read_text())
    admission.validate_config(config, previous)
    config[key] = value
    with pytest.raises(ValueError, match="identity, accumulated costs and limits"):
        admission.validate_config(config, previous)


def test_actual_failures_and_saved_audit_bind_new_final_endpoint_replay():
    inputs = admission.load_inputs()
    registration = admission.replay_registration(inputs, admission.OUTPUT / "checker/simulation")
    assert registration["session"] == inputs["config"]["session"]
    assert registration["image_root"] == str((admission.OUTPUT / "checker/simulation").relative_to(admission.ROOT))
    assert inputs["previous_physical_bounds"]["control_steps"] == dict(lower=2174, upper=2289)
    assert inputs["previous_model_bounds"]["requests_completed"] == dict(lower=27, upper=27)
    assert inputs["producer"]["complete"] and inputs["previous"]["failed_checker"]["complete"] is False
    with pytest.raises(ValueError, match="capture root"):
        admission.replay_registration(inputs, admission.previous.OUTPUT / "checker/simulation")


def test_changed_cached_failure_audit_cannot_be_admitted(monkeypatch):
    original = admission.incident.audit

    def changed():
        result = original()
        result["failure_cause_proven"] = True
        return result

    monkeypatch.setattr(admission.incident, "audit", changed)
    with pytest.raises(ValueError, match="exact recomputed cached-failure audit"):
        admission.load_inputs()
