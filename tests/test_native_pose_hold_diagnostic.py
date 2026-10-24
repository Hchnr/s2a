"""Validate terminal-only goal intervention without running a simulator."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.artifacts import REPO


@pytest.mark.parametrize("method", ["baseline", "latched_pose"])
def test_goal_override_starts_only_after_native_and_preserves_gripper(monkeypatch, method):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    producer = importlib.import_module("collect_native_pose_hold_diagnostic")
    audit = importlib.import_module("verify_native_pose_hold_diagnostic")
    sent = []
    site = np.array([0.3, -0.2, 0.5])

    def set_goal(action, set_pos=None, set_ori=None):
        ctrl.goal_pos = site.copy() if set_pos is None else set_pos.copy()
        ctrl.goal_ori = np.eye(3) if set_ori is None else set_ori.copy()

    ctrl = SimpleNamespace(eef_name="actual_site", set_goal=set_goal)

    def step(action):
        sent.append(action.copy())
        ctrl.set_goal(action[:6])
        site[:] += 0.001

    fake = SimpleNamespace(
        robots=[SimpleNamespace(controller=ctrl)],
        sim=SimpleNamespace(data=SimpleNamespace(get_site_xpos=lambda _: site, get_site_xmat=lambda _: np.eye(3))),
        step=step,
        close=lambda: None,
    )
    wrapped = producer.GoalRecordingEnvironment(fake, 2, method)
    native = np.array([[0.1, 0, 0, 0, 0, 0, -1], [0.2, 0, 0, 0, 0, 0, -1]])
    for command in native:
        wrapped.step(command)
    measured = np.eye(4)
    measured[:3, 3] = site
    for _ in range(40):
        wrapped.step(np.array([0, 0, 0, 0, 0, 0, -1]))
    assert np.array_equal(np.asarray(sent)[:2], native)
    assert np.all(np.asarray(sent)[:, -1] == -1)
    data = dict(sent_commands=np.asarray(sent), controller_site_pose=np.stack([np.eye(4), np.eye(4), measured]))
    audit.verify_goals(wrapped.goals, data, 2, method)
    if method == "latched_pose":
        assert all(np.array_equal(g["actual_goal_pos"], measured[:3, 3]) for g in wrapped.goals[2:])
        wrapped.goals[2]["set_pos"] = [0, 0, 0]
        with pytest.raises(ValueError, match="measured native terminal SITE"):
            audit.verify_goals(wrapped.goals, data, 2, method)
    with pytest.raises(ValueError, match="budget"):
        wrapped.step(np.zeros(7))
    wrapped.close()
    assert ctrl.set_goal is set_goal
