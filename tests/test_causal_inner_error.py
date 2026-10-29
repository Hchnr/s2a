"""Preserve causality and later-period tightening with inner-time error profiles."""

import numpy as np
import pytest
from test_configurable_tracking_terminal_plan import inputs

from s2a.causal_inner_error import CausalInnerError
from s2a.past_error_joint_plan import solve_interior_plan as peak_solve
from s2a.past_inner_error_joint_plan import solve_interior_plan


def test_profile_updates_after_completion_and_does_not_move_late_error_earlier():
    history = CausalInnerError()
    prediction = np.zeros((25, 7))
    history.stage(3, prediction)
    prediction[:] = 10
    before = history.snapshot()
    assert not before["available"]
    actual = np.zeros((25, 7))
    actual[0, 6], actual[-1, 6] = 0.00003, 0.004
    with pytest.raises(ValueError):
        history.complete(4, actual)
    history.complete(3, actual)
    after = history.snapshot()
    assert after["inner_position_error_rad"][0, 6] == 0.00003
    assert after["inner_position_error_rad"][-1, 6] == 0.004
    np.testing.assert_array_equal(before["inner_position_error_rad"], 0)
    after["inner_position_error_rad"][:] = 100
    assert history.snapshot()["inner_position_error_rad"][-1, 6] == 0.004


def test_constant_profile_preserves_peak_solver_and_all_later_constraints():
    args = inputs()
    peak = peak_solve(*args, past_position_error_rad=np.full(7, 0.003))
    constant = solve_interior_plan(*args, past_inner_error_rad=np.full((25, 7), 0.003))
    assert peak["witness_found"] and constant["witness_found"]
    np.testing.assert_array_equal(constant["planned_commands"], peak["planned_commands"])
    profile = np.tile(np.linspace(0.00003, 0.003, 25)[:, None], (1, 7))
    result = solve_interior_plan(*args, past_inner_error_rad=profile)
    assert result["witness_found"]
    margins = result["requested_position_margin_rad"]
    np.testing.assert_array_equal(margins[:25], 0.05 + profile)
    np.testing.assert_array_equal(margins[25:], np.full_like(margins[25:], 0.05 + 0.003))
    assert result["minimum_tightened_inner_slack_rad"] >= -1e-10
    with pytest.raises(ValueError):
        solve_interior_plan(*args, past_inner_error_rad=profile[::-1])
