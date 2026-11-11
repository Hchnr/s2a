"""The read-only audit must reject altered replay and altered causal history."""

import copy

import pytest

from scripts.simulation.runtime import prepare_early_causal_innovation_physical as admission
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as prior
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_early_causal_innovation_physical as audit


def branches():
    inputs = admission.load_inputs()
    producer = prefix.read(prefix.reference(audit.RUN / "producer/simulation/result.json"))
    checker = prefix.read(prefix.reference(audit.RUN / "checker/simulation/result.json"))
    return inputs, producer, checker


def test_audit_records_actual_error_and_a_budget_stop():
    result = audit.audit()
    assert result["prior_periods_matched"] == 42 and result["continuation_periods"] == 43
    assert result["step122"]["actual_pose_caps_satisfied"] is False
    assert result["step122"]["actual_error"]["position_m"] > 0.01
    assert result["stop_kind"] == "registered_one_period_limit"
    assert result["checker_policy_isolated"] is True and result["control_qualified"] is False
    assert max(result["independent_replay_max_difference"].values()) == 0


def test_tampered_replay_and_innovation_are_rejected():
    inputs, producer, checker = branches()
    produced, checked = prior.periods(producer), copy.deepcopy(prior.periods(checker))
    checked[-1]["actual_inner"][-1]["q"][0] += 0.01
    with pytest.raises(AssertionError):
        audit.compare_periods(produced, checked, inputs["prior_periods"])
    runtime = copy.deepcopy(producer["runtime"])
    runtime["early_innovation_binding"]["translation_innovation_m"][0] += 0.01
    with pytest.raises(AssertionError):
        audit.verify_causal_action(produced, inputs["prior_periods"], runtime)
