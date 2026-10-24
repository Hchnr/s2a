"""Only terminal command zero, never measured qpos, may freeze motor targets."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.artifacts import REPO


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("collect_native_motor_hold_diagnostic")


def test_only_terminal_gripper_commands_change(monkeypatch):
    module = load(monkeypatch)
    native = np.arange(43 * 7, dtype=float).reshape(43, 7) / 1000
    actual = module.motor_hold_schedule(native)
    assert np.array_equal(actual[:43], native)
    assert actual.shape == (83, 7) and np.all(actual[43:] == 0)
    assert native[-1, -1] != 0


@pytest.mark.parametrize("method", ["motor_hold", "pose_motor_hold"])
def test_actual_substeps_preserve_partial_motor_target(monkeypatch, method):
    module = load(monkeypatch)
    data = SimpleNamespace(time=0.0, ctrl=np.zeros(2))
    data.get_site_xpos = lambda _: np.array([0.3, -0.2, 0.5])
    data.get_site_xmat = lambda _: np.eye(3)
    ranges = np.array([[0, 0.04], [-0.04, 0]])
    model = SimpleNamespace(actuator_ctrlrange=ranges, actuator_name2id=lambda name: ["left", "right"].index(name))
    gripper = SimpleNamespace(current_action=np.array([0.0]), actuators=["left", "right"])
    controller = SimpleNamespace(eef_name="site")

    def set_goal(action, set_pos=None, set_ori=None):
        controller.goal_pos = data.get_site_xpos("site") if set_pos is None else set_pos.copy()
        controller.goal_ori = np.eye(3) if set_ori is None else set_ori.copy()

    def grip_action(gripper, gripper_action):
        gripper.current_action = np.clip(
            gripper.current_action + np.array([-1, 1]) * 0.01 * np.sign(gripper_action), -1, 1
        )
        data.ctrl[:] = ranges.mean(axis=1) + (ranges[:, 1] - ranges[:, 0]) * 0.5 * gripper.current_action

    controller.set_goal = set_goal
    robot = SimpleNamespace(controller=controller, gripper=gripper, grip_action=grip_action)

    def step(action):
        controller.set_goal(action[:6])
        for _ in range(25):
            robot.grip_action(gripper=gripper, gripper_action=action[-1:])
            data.time += 0.002

    env = SimpleNamespace(robots=[robot], sim=SimpleNamespace(model=model, data=data), step=step, close=lambda: None)
    wrapped = module.MotorRecordingEnvironment(env, 2, method)
    for _ in range(2):
        wrapped.step(np.array([0, 0, 0, 0, 0, 0, 1]))
    target = gripper.current_action.copy()
    assert np.allclose(target, [-0.5, 0.5])
    for _ in range(40):
        wrapped.step(np.zeros(7))
    assert len(wrapped.motor_calls) == 42 * 25
    assert np.array_equal(wrapped.motor_calls[0]["cache_before"], [0.0])
    assert all(np.array_equal(item["cache_after"], target) for item in wrapped.motor_calls[50:])
    assert all(np.array_equal(item["current_action"], target) for item in wrapped.motor_targets[2:])
    assert np.array_equal(wrapped.motor_targets[1]["actuator_ctrl"], wrapped.motor_targets[-1]["actuator_ctrl"])
    assert all(g["overridden"] is (method == "pose_motor_hold") for g in wrapped.goals[2:])
    wrapped.close()
    assert robot.grip_action is grip_action and controller.set_goal is set_goal
