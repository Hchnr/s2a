"""Branchpoint history comparisons use the prior physical endpoint interval."""

import numpy as np
import pytest

from scripts.verification.reference_ports.audit_same_state_arm_branchpoint import FIELDS, compare_history


def test_branchpoint_history_compares_controller_and_mean_object_motion():
    baseline = {name: np.array([0.0]) for name in FIELDS}
    current = {name: value.copy() for name, value in baseline.items()}
    current["controller_goal_pos"][0] = 0.005
    old = [[0, 0, 0], [0.001, 0, 0]]
    new = [[0, 0, 0], [0.002, 0, 0]]
    result = compare_history(baseline, current, old, new, dt=0.05)
    assert result["field_maximum_absolute_differences"]["controller_goal_pos"] == pytest.approx(0.005)
    assert result["object_position_difference_m"] == pytest.approx(0.001)
    assert result["previous_period_mean_velocity_difference_m_s"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="interval"):
        compare_history(baseline, current, old, new, dt=0)
