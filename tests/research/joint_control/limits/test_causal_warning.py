"""Check direction, causality, and observed-window boundaries of warning rules."""

import numpy as np
import pytest

from s2a.research.joint_control.limits.causal_warning import joint_warnings, summarize_rule

LIMITS = np.tile([-1.0, 1.0], (7, 1))


@pytest.mark.parametrize("side,position,velocity", [("upper", 0.90, 0.30), ("lower", -0.90, -0.30)])
def test_outward_velocity_warns_before_buffer_and_inward_does_not(side, position, velocity):
    q, v = np.zeros(7), np.zeros(7)
    q[3], v[3] = position, velocity
    warnings = joint_warnings(q, v, LIMITS)
    assert warnings["buffer"]["alarm"] is False
    assert warnings["velocity_0p2s"]["alarm"] is True
    assert warnings["velocity_0p2s"]["joint_index"] == 3
    assert warnings["velocity_0p2s"]["side"] == side
    v[3] = -velocity
    assert joint_warnings(q, v, LIMITS)["velocity_0p2s"]["alarm"] is False


def test_velocity_warning_selects_actual_most_risky_joint_not_nearest():
    q, v = np.zeros(7), np.zeros(7)
    q[0], q[5], v[5] = 0.93, 0.85, 0.8
    warnings = joint_warnings(q, v, LIMITS)
    assert warnings["static_0p10"]["joint_index"] == 0
    assert warnings["velocity_0p2s"]["joint_index"] == 5
    assert warnings["velocity_0p2s"]["alarm"] is True


def test_warning_rejects_invalid_measurements_and_limits():
    with pytest.raises(ValueError):
        joint_warnings(np.full(7, np.nan), np.zeros(7), LIMITS)
    with pytest.raises(ValueError):
        joint_warnings(np.zeros(7), np.zeros(7), np.zeros((7, 2)))


def test_future_event_window_includes_fourth_endpoint_and_censors_incomplete():
    q = np.zeros((9, 7))
    q[4, 0] = 1.01
    flags = np.array([True, False, False, False, True, False, False, False])
    result = summarize_rule(flags, q, LIMITS)
    assert result["first_outside_endpoint"] == 4
    assert result["first_alarm_while_buffer_safe_before_first_outside"] == 0
    assert result["lead_periods_to_first_outside"] == 4
    assert result["fully_observed_safe_buffer_windows"] == {
        "alarm_event": 1,
        "alarm_no_event": 0,
        "no_alarm_event": 3,
        "no_alarm_no_event": 0,
    }
    assert result["safe_buffer_windows_censored"] == 3


def test_unsafe_alarm_is_not_retrospectively_promoted_to_early_warning():
    q = np.zeros((7, 7))
    q[1, 0], q[2, 0] = 0.96, 1.01
    flags = np.array([False, True, True, False, False, False])
    result = summarize_rule(flags, q, LIMITS)
    assert result["first_alarm_endpoint"] == 1
    assert result["first_alarm_while_buffer_safe_before_first_outside"] is None
    assert result["lead_periods_to_first_outside"] is None


def test_no_outside_case_does_not_claim_lead_or_future_safety():
    q = np.zeros((7, 7))
    flags = np.array([False, True, False, False, False, False])
    result = summarize_rule(flags, q, LIMITS)
    assert result["first_outside_endpoint"] is None
    assert result["first_alarm_while_buffer_safe_before_first_outside"] == 1
    assert result["lead_periods_to_first_outside"] is None
    assert result["fully_observed_safe_buffer_windows"]["alarm_no_event"] == 1
