"""Verify the actual goal boundary and retain the entire native command prefix."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.settled_native import SettledGoalEnvironment, settled_anchor_indices, settled_command_schedule


def test_settled_windows_preserve_native_and_last_gripper():
    native = np.arange(43 * 7, dtype=float).reshape(43, 7) / 1000
    actual = settled_command_schedule(native)
    assert np.array_equal(actual[:43], native)
    assert actual.shape == (123, 7)
    assert np.all(actual[43:, :6] == 0)
    assert np.all(actual[43:, 6] == native[-1, 6])
    assert settled_anchor_indices(43) == [0, 1, 2, 3, 83]
    assert settled_anchor_indices(600) == [0, 186, 373, 560, 640]


def test_only_terminal_goals_use_fixed_native_end_measurement():
    position = np.array([0.3, -0.2, 0.5])
    controller = SimpleNamespace(eef_name="site")
    calls = []

    def goal(action, set_pos=None, set_ori=None):
        controller.goal_pos = position.copy() if set_pos is None else set_pos.copy()
        controller.goal_ori = np.eye(3) if set_ori is None else set_ori.copy()
        calls.append((action.copy(), set_pos, set_ori))

    controller.set_goal = goal
    sent = []

    def step(action):
        sent.append(action.copy())
        controller.set_goal(action[:6])
        position[0] += 0.001

    data = SimpleNamespace(get_site_xpos=lambda _: position, get_site_xmat=lambda _: np.eye(3))
    env = SimpleNamespace(
        robots=[SimpleNamespace(controller=controller)], sim=SimpleNamespace(data=data), step=step, close=lambda: None
    )
    wrapper = SettledGoalEnvironment(env, 43)
    native = np.full((43, 7), 0.1)
    native[-1, 6] = -1
    for action in settled_command_schedule(native):
        wrapper.step(action)
    assert all(call[1] is None for call in calls[:43])
    assert np.array_equal(np.stack(sent)[:43], native)
    assert all(np.array_equal(call[1], wrapper.latch[0]) for call in calls[43:])
    assert all(item["overridden"] for item in wrapper.goals[43:])
    assert np.isclose(wrapper.latch[0][0], 0.343)
    assert np.all(np.stack(sent)[43:, -1] == -1)
    with pytest.raises(ValueError, match="budget"):
        wrapper.step(np.zeros(7))
    wrapper.close()
    assert controller.set_goal is goal
