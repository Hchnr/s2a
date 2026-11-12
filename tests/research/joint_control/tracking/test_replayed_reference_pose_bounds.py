import copy

import numpy as np
import pytest

from scripts.tracking.trajectory_search.audit_replayed_reference_pose_bounds import (
    completed_plan_input,
    reference_window,
)


def completed_period():
    plan = np.tile(np.arange(8, dtype=float)[:, None] * 0.05, (1, 6))
    return {
        "step": 107,
        "command": [*plan[0], 0.0],
        "decision": {"refined_plan": plan},
        "completed_feedback": {
            "policy_history": {
                "past_inner_error": {
                    "available": True,
                    "future_guarantee": False,
                    "last_completed_step": 107,
                    "total_completed_periods": 108,
                    "inner_position_error_rad": np.tile(np.arange(25)[:, None] * 0.001, (1, 7)),
                }
            }
        },
    }


def test_completed_plan_uses_only_the_preceding_dispatch_and_keeps_rate():
    period = completed_period()
    shifted, profile = completed_plan_input(period, 108)
    np.testing.assert_array_equal(shifted[:-1], period["decision"]["refined_plan"][1:])
    np.testing.assert_array_equal(shifted[-1], shifted[-2])
    assert profile.shape == (25, 7)
    shifted[0, 0] = 99
    profile[0, 0] = 99
    assert period["decision"]["refined_plan"][1, 0] == 0.05
    assert period["completed_feedback"]["policy_history"]["past_inner_error"]["inner_position_error_rad"][0, 0] == 0


def test_completed_plan_rejects_future_history_or_a_different_dispatched_command():
    period = completed_period()
    with pytest.raises(ValueError):
        completed_plan_input(period, 107)
    wrong_command = copy.deepcopy(period)
    wrong_command["command"][0] = 0.1
    with pytest.raises(ValueError):
        completed_plan_input(wrong_command, 108)
    future = copy.deepcopy(period)
    future["completed_feedback"]["policy_history"]["past_inner_error"]["last_completed_step"] = 108
    with pytest.raises(ValueError):
        completed_plan_input(future, 108)
    unbounded = copy.deepcopy(period)
    unbounded["decision"]["refined_plan"][1, 0] = 0.5
    with pytest.raises(ValueError):
        completed_plan_input(unbounded, 108)


def test_reference_window_preserves_the_generated_clock_and_rejects_future_chunk():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, 0, 3] = np.arange(40)
    selected = reference_window(poses, 88, 108)
    np.testing.assert_array_equal(selected[:, 0, 3], np.arange(20, 28))
    selected[0, 0, 3] = -1
    assert poses[20, 0, 3] == 20
    with pytest.raises(ValueError):
        reference_window(poses, 88, 121)
    with pytest.raises(ValueError):
        reference_window(poses, 88, 87)
