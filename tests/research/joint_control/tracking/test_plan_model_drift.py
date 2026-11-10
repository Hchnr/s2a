"""Drift diagnostics must use completed plans and preserve signed changes."""

import numpy as np
import pytest

from s2a.research.joint_control.tracking.plan_model_drift import (
    compare_joint_endpoint,
    compare_next_pose_forecasts,
    shifted_completed_plan,
)


def test_shift_uses_the_completed_command_and_respects_rate():
    plan = np.zeros((8, 6))
    plan[:, 0] = np.arange(8) * 0.1
    period = dict(command=np.r_[plan[0], 0.0], decision=dict(refined_plan=plan))
    shifted = shifted_completed_plan(period)
    np.testing.assert_array_equal(shifted[:-1], plan[1:])
    np.testing.assert_array_equal(shifted[-1], plan[-1])
    period["command"][0] = 0.01
    with pytest.raises(ValueError, match="actual completed command"):
        shifted_completed_plan(period)


def test_signed_endpoint_and_forecast_differences_telescope():
    q = compare_joint_endpoint(np.zeros(7), np.ones(7), np.ones(7) * 0.1, np.ones(7) * 0.8)
    np.testing.assert_allclose(q["joint_position_error_rad"], 0.1)
    np.testing.assert_allclose(q["joint_velocity_error_rad_s"], -0.2)
    a, b, c, d = (np.arange(6) * scale for scale in (0.0, 0.1, 0.3, 0.2))
    result = compare_next_pose_forecasts(a, b, c, d)
    np.testing.assert_allclose(sum(result["components"].values()), d - a, atol=1e-12, rtol=0)
    assert result["component_norms"]["velocity_update"]["rotation_rad"] > 0
