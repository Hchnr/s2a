"""Audit retained partial checker evidence without claiming a closed replay."""

import copy

import pytest

from scripts.simulation.runtime import audit_adaptive_long_partial as audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


@pytest.fixture(scope="module")
def saved_branches():
    producer = prefix.read(audit.PRODUCER)
    checker = prefix.read(audit.CHECKER)
    return prefix.read(producer["branch_ref"]), prefix.read(checker["available_branch_ref"])


def test_saved_partial_replay_matches_all_adaptive_actions_and_actual_images(saved_branches):
    producer, checker = saved_branches
    details = audit.compare_partial(producer, checker)
    assert details["adaptive_periods"] == 40 and details["adaptive_inner_samples"] == 1000
    assert all(value == 0 for value in details["adaptive_max_absolute_difference"].values())
    assert details["capture_steps"] == [107, 111, 119, 120, 128, 136, 144]
    assert details["capture_bytes_match"] and details["final_endpoint_receipt_matches"]
    assert details["failed_adaptive_endpoints"] == [129] and details["trailing_success_periods"] == 18
    assert details["checker_complete"] is False and details["checker_error"] == audit.FAILURE


@pytest.mark.parametrize("change", ["command", "inner", "label", "segment", "image", "failure"])
def test_changed_partial_evidence_is_rejected(saved_branches, change):
    producer, original = saved_branches
    checker = copy.deepcopy(original)
    runtime = checker["runtime"]
    replay = runtime["adaptive_replay_partial"]
    period = replay["segments"][0]["continuation"]["periods"][0]
    if change == "command":
        period["command"][0] += 0.01
    elif change == "inner":
        period["actual_inner"][0]["q"][0] += 0.01
    elif change == "label":
        period["task_success"] = not period["task_success"]
    elif change == "segment":
        replay["segments"].pop()
    elif change == "image":
        runtime["adaptive_checker_captures"][0]["canvas_png"]["sha256"] = "0" * 64
    else:
        runtime["error"]["message"] = "another failure"
    with pytest.raises(ValueError):
        audit.compare_partial(producer, checker)


def test_full_audit_retains_failed_checker_and_cumulative_uncertainty():
    result = audit.audit()
    assert result["complete"] and result["passed"] and result["control_qualified"] is False
    assert result["evidence_level"] == "saved_partial_checker_audit"
    assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2055, upper=2170)
    assert result["cumulative_model_bounds"]["requests_completed"] == dict(lower=27, upper=27)
    assert result["new_model_requests"] == result["new_physics_steps"] == 0


def test_changed_failed_checker_cost_cannot_be_audited_as_valid(monkeypatch):
    read = prefix.read

    def changed(ref, seen=None):
        result = read(ref, seen)
        if ref == audit.CHECKER:
            result["cumulative_physical_bounds"]["control_steps"]["upper"] -= 1
        return result

    monkeypatch.setattr(prefix, "read", changed)
    with pytest.raises(ValueError, match="actual costs and previous uncertainty"):
        audit.audit()
