"""Check physically relevant direction and fixed-interface limits."""

import numpy as np
import pytest

from s2a.controllers.osc import OSCInterface
from s2a.research.joint_control.limits.emergency_retreat import jacobian_retreat, rate_to_zero


@pytest.fixture
def interface():
    return OSCInterface(-np.ones(6), np.ones(6), -np.array([0.05] * 3 + [0.5] * 3), np.array([0.05] * 3 + [0.5] * 3))


@pytest.mark.parametrize(("qvalue", "expected_side", "expected_sign"), [(0.99, "upper", -1), (-0.99, "lower", 1)])
def test_retreat_points_inward_for_the_nearest_limit(interface, qvalue, expected_side, expected_sign):
    q = np.zeros(7)
    q[6] = qvalue
    limits = np.tile([-1.0, 1.0], (7, 1))
    jacobian = np.zeros((6, 7))
    jacobian[0, 6] = 1
    result = jacobian_retreat(q, limits, jacobian, np.zeros(6), interface)
    assert result["joint_index"] == 6
    assert result["side"] == expected_side
    assert np.sign(result["signed_joint_delta_rad"]) == expected_sign
    assert np.sign(result["command"][0]) == expected_sign
    assert abs(result["command"][0]) == pytest.approx(0.25)


def test_rate_limit_applies_to_actual_previous_command(interface):
    previous = np.array([0.9, -0.7, 0.1, 0.0, 0.0, 0.0])
    result = rate_to_zero(previous, interface)
    np.testing.assert_allclose(result["command"], [0.65, -0.45, 0, 0, 0, 0], atol=1e-15, rtol=0)
    assert result["rotation_goal_may_latch"] is True
    np.testing.assert_array_equal(previous, [0.9, -0.7, 0.1, 0, 0, 0])


def test_retreat_rejects_nonfinite_state_and_invalid_limits(interface):
    q = np.zeros(7)
    limits = np.tile([-1.0, 1.0], (7, 1))
    q[0] = np.nan
    with pytest.raises(ValueError):
        jacobian_retreat(q, limits, np.zeros((6, 7)), np.zeros(6), interface)
    q[0] = 0
    limits[0] = [1, -1]
    with pytest.raises(ValueError, match="limits"):
        jacobian_retreat(q, limits, np.zeros((6, 7)), np.zeros(6), interface)
