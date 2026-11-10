"""The stopped-state diagnostic uses only actual completed feedback."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.tracking.quality_stop_diagnostic import stopped_feedback_inputs


def periods():
    plan = np.zeros((8, 6))
    plan[:, 0] = np.arange(8) * 0.1
    result = [{"step": 43 + i} for i in range(5)]
    result[-1].update(
        {
            "actual_state": np.arange(79),
            "command": np.r_[plan[0], 0.0],
            "decision": {"refined_plan": plan},
            "completed_feedback": {
                "past_error_after_completion": {
                    "completed_periods": 5,
                    "last_completed_step": 47,
                    "available": True,
                    "future_guarantee": False,
                    "inner_position_error_rad": np.tile(np.arange(25)[:, None] / 1000, (1, 7)),
                }
            },
        }
    )
    return result


def test_shift_uses_only_last_completed_action_plan_and_history():
    actual = periods()
    saved = copy.deepcopy(actual)
    result = stopped_feedback_inputs(actual, 43)
    assert result["stop_step"] == 48 and result["completed_periods"] == 5
    np.testing.assert_array_equal(result["seed"][:-1], actual[-1]["decision"]["refined_plan"][1:])
    np.testing.assert_array_equal(result["seed"][-1], result["seed"][-2])
    result["seed"][0, 0] = -1
    np.testing.assert_array_equal(actual[-1]["decision"]["refined_plan"], saved[-1]["decision"]["refined_plan"])


def test_rejects_unexecuted_or_rate_infeasible_shifted_plan():
    actual = periods()
    actual[-1]["command"][0] = 0.1
    with pytest.raises(ValueError, match="last actual command"):
        stopped_feedback_inputs(actual, 43)
    actual = periods()
    actual[-1]["decision"]["refined_plan"][1, 0] = 0.5
    with pytest.raises(ValueError, match="box and rate"):
        stopped_feedback_inputs(actual, 43)


def test_rejects_uncompleted_or_future_error_information():
    actual = periods()
    actual[-1]["completed_feedback"]["past_error_after_completion"]["completed_periods"] = 6
    with pytest.raises(ValueError, match="past-only"):
        stopped_feedback_inputs(actual, 43)
    actual = periods()
    actual[2]["step"] = 100
    with pytest.raises(ValueError, match="Consecutive"):
        stopped_feedback_inputs(actual, 43)
