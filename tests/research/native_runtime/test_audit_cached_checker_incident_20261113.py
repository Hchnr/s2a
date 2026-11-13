"""Keep the real partial replay failure and its evidence distinguishable."""

import copy

import pytest

from scripts.simulation.runtime import audit_adaptive_long_partial as previous
from scripts.simulation.runtime import audit_cached_checker_incident_20261113 as audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


@pytest.fixture(scope="module")
def partial_periods():
    producer = prefix.read(prefix.read(previous.PRODUCER)["branch_ref"])
    branch = prefix.read(audit.BRANCH)
    expected = [
        row
        for segment in producer["runtime"]["continuation"]["adaptive"]["segments"]
        for row in segment["continuation"]["periods"]
    ][:12]
    actual = [
        row
        for segment in branch["runtime"]["adaptive_replay_partial"]["segments"]
        for row in segment["continuation"]["periods"]
    ]
    return expected, actual


def test_actual_partial_failure_keeps_hash_difference_and_exact_image_bytes():
    result = audit.audit()
    assert result["complete"] and result["passed"] and result["control_qualified"] is False
    assert result["partial_comparison"]["differing_state_digest_endpoints"] == list(range(112, 120))
    assert result["partial_comparison"]["max_absolute_difference"]["command"] == 0
    assert result["partial_comparison"]["max_absolute_difference"]["state"] < 1e-10
    assert result["failed_capture_image_bytes_match"] and not result["failed_capture_state_digest_matches"]
    assert result["reused_endpoint_steps"] == [111, 119] and result["failure_cause_proven"] is False
    assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=2174, upper=2289)
    assert result["new_model_requests"] == result["new_physics_steps"] == 0


@pytest.mark.parametrize("change", ["command", "state", "label", "inner"])
def test_altered_partial_samples_cannot_be_reported_as_original_evidence(partial_periods, change):
    expected, actual = partial_periods
    altered = copy.deepcopy(actual)
    row = altered[0]
    if change == "command":
        row["command"][0] += 0.01
    elif change == "state":
        row["actual_state"][0] += 0.01
    elif change == "label":
        row["task_success"] = not row["task_success"]
    else:
        row["actual_inner"].pop()
    with pytest.raises(ValueError):
        audit.compare_partial_periods(expected, altered)
