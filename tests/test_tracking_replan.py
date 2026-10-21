"""Physical endpoint chronology and threshold contract for reobservation."""

import numpy as np
import pytest

from s2a.tracking_replan import TrackingReplan, TrackingReplanConfig


def advance(policy, errors):
    for error in errors:
        step = policy.last_observed_step + 1
        assert not policy.decision(step)["query_due"]
        policy.observe(step, error)


def test_earliest_query_uses_past_endpoints_and_mixed_axis_violations():
    policy = TrackingReplan()
    assert policy.decision(0)["query_reason"] == "initial"
    policy.start_query(0)
    high_position = [0.041, 0, 0, 0, 0, 0]
    high_rotation = [0, 0, 0, 0.401, 0, 0]
    advance(policy, [high_position] * 7 + [high_rotation])
    assert policy.decision(8) == dict(
        physical_step=8, age_steps=8, previous_violation_count=8, query_due=True, query_reason="tracking_limit"
    )
    policy.start_query(8)
    policy.observe(8, high_position)
    assert not policy.decision(9)["query_due"]
    assert policy.decision(9)["previous_violation_count"] == 1


def test_strict_boundaries_normal_reset_and_two_new_violations():
    policy = TrackingReplan()
    policy.start_query(0)
    advance(policy, [[0.04, 0, 0, 0.4, 0, 0]] * 8)
    assert policy.decision(8)["previous_violation_count"] == 0
    policy.observe(8, [0.05, 0, 0, 0, 0, 0])
    assert not policy.decision(9)["query_due"]
    policy.observe(9, np.zeros(6))
    assert policy.decision(10)["previous_violation_count"] == 0
    policy.observe(10, [0.05, 0, 0, 0, 0, 0])
    policy.observe(11, [0, 0, 0, 0.5, 0, 0])
    assert policy.decision(12)["query_reason"] == "tracking_limit"


def test_maximum_interval_and_exhaustion_override_and_reference_reset():
    policy = TrackingReplan()
    policy.start_query(0)
    advance(policy, [np.zeros(6)] * 40)
    assert policy.decision(40)["query_reason"] == "maximum_interval"
    assert policy.decision(40, exhausted=True)["query_reason"] == "reference_exhausted"
    policy.start_query(40)
    assert policy.decision(40, exhausted=True)["query_reason"] == "reference_exhausted"
    policy.observe(40, np.zeros(6))
    assert policy.decision(41, exhausted=True)["query_reason"] == "reference_exhausted"
    policy.start_query(41)
    assert policy.decision(41)["age_steps"] == 0


def test_rejects_noncausal_or_duplicate_observations_and_bad_errors():
    policy = TrackingReplan()
    with pytest.raises(ValueError):
        policy.observe(0, np.zeros(6))
    policy.start_query(0)
    with pytest.raises(ValueError):
        policy.start_query(0)
    for step in (True, -1, 1):
        with pytest.raises(ValueError):
            policy.decision(step)
    for error in ([0] * 5, [0, 0, 0, 0, 0, float("nan")], np.full(6, float("inf"))):
        with pytest.raises(ValueError):
            policy.observe(0, error)
    policy.observe(0, np.zeros(6))
    with pytest.raises(ValueError):
        policy.observe(0, np.zeros(6))
    with pytest.raises(ValueError):
        policy.decision(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_interval_steps": True},
        {"min_interval_steps": 0},
        {"max_interval_steps": 2},
        {"consecutive_steps": 0},
        {"position_limit_m": -1},
        {"rotation_limit_rad": float("nan")},
        {"position_limit_m": True},
    ],
)
def test_rejects_invalid_policy_contract(kwargs):
    with pytest.raises(ValueError):
        TrackingReplanConfig(**kwargs)
