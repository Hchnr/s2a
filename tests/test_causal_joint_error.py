"""Prevent current observations and mutable caller buffers entering past bounds."""

import numpy as np
import pytest
from test_configurable_tracking_terminal_plan import inputs

from s2a.causal_joint_error import CausalJointError
from s2a.past_error_joint_plan import solve_interior_plan
from s2a.velocity_braking_joint_plan import solve_interior_plan as original


def test_history_uses_only_completed_periods_and_copies_predictions():
    history = CausalJointError()
    prediction = np.zeros((25, 7))
    history.stage(10, prediction)
    prediction[:] = 100
    assert not history.snapshot()["available"]
    np.testing.assert_array_equal(history.snapshot()["position_error_rad"], 0)
    with pytest.raises(ValueError):
        history.complete(11, np.ones((25, 7)))
    history.complete(10, np.full((25, 7), 0.003))
    prior = history.snapshot()
    history.stage(11, np.zeros((25, 7)))
    history.complete(11, np.full((25, 7), 0.02))
    np.testing.assert_array_equal(prior["position_error_rad"], np.full(7, 0.003))
    np.testing.assert_array_equal(history.snapshot()["position_error_rad"], np.full(7, 0.02))
    with pytest.raises(ValueError):
        history.stage(13, np.zeros((25, 7)))


def test_zero_error_preserves_braking_plan_and_tightening_is_checked():
    args = inputs()
    baseline = original(*args)
    zero = solve_interior_plan(*args)
    assert baseline["witness_found"] and zero["witness_found"]
    np.testing.assert_array_equal(zero["planned_commands"], baseline["planned_commands"])
    error = np.full(7, 0.003)
    tightened = solve_interior_plan(*args, past_position_error_rad=error)
    assert tightened["witness_found"]
    q = tightened["rollout"]["inner_joint_position"]
    assert np.min(np.minimum(q - args[-2] - 0.05 - error, args[-1] - 0.05 - error - q)) >= -1e-10
    assert tightened["minimum_tightened_inner_slack_rad"] >= -1e-10
    with pytest.raises(ValueError):
        solve_interior_plan(*args, past_position_error_rad=-error)
