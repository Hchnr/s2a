"""The anticipatory search may consume only an already completed period."""

import copy

import numpy as np
import pytest

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.tracking.trajectory_search.run_anticipatory_causal_pose import completed_step120


class FuturePeriod(dict):
    def __getitem__(self, key):
        if key == "step":
            return 121
        raise AssertionError("Future observations cannot be read")


def branch(row):
    return {"runtime": {"continuation": {"segments": [{"continuation": {"periods": [row, FuturePeriod()]}}]}}}


def completed_row():
    state = np.zeros(79)
    return {
        "step": 120,
        "actual_state": state,
        "command": np.zeros(7),
        "actual_inner": [{"q": np.zeros(7), "qvel": np.zeros(7), "time": float(index)} for index in range(25)],
        "endpoint_receipt": {"sim_state_sha256": state_sha256(state)},
        "decision": {
            "refined_plan": np.zeros((8, 6)),
            "quality": {"verification": {"passed": True}, "first_tcp_error": np.zeros(6)},
        },
        "tcp_error": np.array([0.001, 0, 0, 0, 0, 0]),
        "completed_feedback": {
            "past_inner_error": {
                "available": True,
                "last_completed_step": 120,
                "completed_periods": 1,
                "total_completed_periods": 41,
                "future_guarantee": False,
                "inner_position_error_rad": np.zeros((25, 7)),
            }
        },
        "task_success": object(),
        "target_grasp": object(),
    }


def test_step120_only_does_not_read_future_observations_or_task_labels():
    row = completed_row()
    actual = completed_step120(branch(row), branch(copy.deepcopy(row)))
    np.testing.assert_array_equal(actual["translation_innovation"], [0.001, 0, 0])
    np.testing.assert_array_equal(actual["state"], row["actual_state"])


def test_mismatched_replay_is_rejected_before_history_is_used():
    row = completed_row()
    checker = copy.deepcopy(row)
    checker["actual_inner"][0]["q"][0] = 0.1
    with pytest.raises(AssertionError):
        completed_step120(branch(row), branch(checker))
