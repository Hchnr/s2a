"""An explicitly bounded earlier recording window uses its real step indices."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime.tabletop_projection_inner_recording import ProjectionInnerRecorder


def test_earlier_window_labels_each_period_and_refuses_budget_overrun():
    controller = SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3))
    env = SimpleNamespace(
        robots=[SimpleNamespace(controller=controller, gripper=SimpleNamespace(current_action=[0, 0]))]
    )
    vertices = np.array([[0.01, 0, 0]])
    sensor = SimpleNamespace(capture=lambda: {"tcp": np.eye(4), "vertices_tcp": vertices, "time_s": 1.0})
    samples = []

    def step():
        samples.append({"time_before": 1.0, "time_after": 1.002})

    base = SimpleNamespace(samples=samples, original=lambda: None, step=step)
    recorder = ProjectionInnerRecorder(env, sensor, "pose_bounded", first_step=118, periods=2, base=base)
    for decision_step in (118, 119):
        recorder.set_context(decision_step, np.eye(4), vertices)
        for _ in range(25):
            recorder.step()
    assert [row["step"] for row in samples] == [118] * 25 + [119] * 25
    with pytest.raises(ValueError):
        recorder.set_context(120, np.eye(4), vertices)
