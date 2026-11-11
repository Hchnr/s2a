"""Reject differences in cached controller state before branching."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime.tabletop_projection_continuation import compare_branch_start


def example():
    return {
        "state_sha256": "a" * 64,
        "time_s": 6.0,
        "current_tcp": np.eye(4),
        "current_vertices_tcp": np.array([[0.01, 0.0, 0.0]]),
        "previous_command": np.zeros(7),
        "controller_goal_position": np.zeros(3),
        "controller_goal_rotation": np.eye(3),
        "gripper_motor_target": np.zeros(2),
    }


@pytest.mark.parametrize("field", ["controller_goal_position", "gripper_motor_target", "previous_command"])
def test_branch_rejects_changed_controller_or_gripper_history(field):
    baseline = example()
    actual = copy.deepcopy(baseline)
    compare_branch_start(actual, baseline)
    actual[field].flat[0] += 0.01
    with pytest.raises(AssertionError):
        compare_branch_start(actual, baseline)


def test_branch_rejects_changed_simulator_state():
    baseline = example()
    actual = copy.deepcopy(baseline)
    actual["state_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        compare_branch_start(actual, baseline)
