"""Audit replay evidence and reject tampered completed physics."""

import copy

import pytest

from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_quality_rate_repair_recorded_feedback as audit


def test_audit_binds_physical_error_and_independent_replay():
    result = audit.audit()
    assert result["continuation_periods"] == 46
    assert result["new_periods_after_previous_stop"] == 1
    assert result["terminal_consecutive_success_periods"] == 39
    assert result["step125"]["actual_pose_caps_satisfied"] is False
    assert result["step125"]["actual_position_error_m"] > 0.01
    assert result["checker_policy_isolated"] is True
    assert result["new_model_requests"] == 0
    assert max(result["independent_replay_max_difference"].values()) <= 1e-10
    assert result["control_qualified"] is False


def test_tampered_replay_cannot_be_reported_as_matching():
    inputs = admission.load_inputs()
    producer = prefix.read(prefix.reference(audit.RUN / "producer/simulation/result.json"))
    checker = prefix.read(prefix.reference(audit.RUN / "checker/simulation/result.json"))
    produced, checked = admission.periods(producer), copy.deepcopy(admission.periods(checker))
    checked[-1]["actual_inner"][-1]["q"][0] += 0.01
    with pytest.raises(AssertionError):
        audit.compare_periods(produced, checked, inputs["prior_periods"])
