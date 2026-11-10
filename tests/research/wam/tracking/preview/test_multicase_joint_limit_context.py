"""First crossing and buffer history must use saved physical endpoints."""

import numpy as np
import pytest

from scripts.wam.tracking.preview.describe_multicase_joint_limit_context import first_limit_context


def test_first_crossing_lists_every_joint_and_latest_buffer_loss():
    limits = np.tile([-1.0, 1.0], (7, 1))
    q = np.zeros((4, 7))
    q[1, 0] = 0.97
    q[2, 0] = 0.92
    q[3, [0, 2]] = [1.01, -1.02]
    result = first_limit_context(q, limits)
    assert result["first_outside_endpoint"] == 3
    assert result["first_outside_joint_numbers"] == [1, 3]
    assert result["last_at_or_above_buffer_endpoint"] == 2
    assert result["first_below_buffer_after_last_high_endpoint"] == 3
    assert result["cycles_from_buffer_loss_to_first_outside"] == 0


def test_initially_inside_buffer_is_not_an_observed_buffer_crossing():
    limits = np.tile([-1.0, 1.0], (7, 1))
    q = np.zeros((2, 7))
    q[:, 3] = [0.97, 1.001]
    result = first_limit_context(q, limits)
    assert result["last_at_or_above_buffer_endpoint"] is None
    assert result["first_below_buffer_after_last_high_endpoint"] == 0
    assert result["cycles_from_buffer_loss_to_first_outside"] == 1


def test_exact_joint_boundary_is_not_outside_and_invalid_data_rejected():
    limits = np.tile([-1.0, 1.0], (7, 1))
    q = np.zeros((2, 7))
    q[1, 0] = 1.0
    assert first_limit_context(q, limits)["first_outside_endpoint"] is None
    q[0, 1] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        first_limit_context(q, limits)
