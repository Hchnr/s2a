"""An earlier face-feedback window preserves actual decision indices."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime.box_face_feedback_measurement import BoxFaceInnerRecorder


def test_early_window_records_actual_steps_and_rejects_overrun():
    controller = SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3))
    env = SimpleNamespace(
        robots=[SimpleNamespace(controller=controller, gripper=SimpleNamespace(current_action=[0, 0]))]
    )
    current = dict(
        tcp=np.eye(4),
        vertices_tcp=np.array([[0.0, 0, 0]]),
        time_s=1.0,
        joint_position_m=-0.03,
        slide_axis_tcp=np.array([1.0, 0, 0]),
    )
    sensor = SimpleNamespace(capture=lambda: current, interval=np.array([-0.04, 0]))
    samples = []

    def step():
        samples.append({"time_before": 1.0, "time_after": 1.002})

    base = SimpleNamespace(samples=samples, original=lambda: None, step=step)
    recorder = BoxFaceInnerRecorder(env, sensor, first_step=118, periods=2, base=base)
    for decision_step in (118, 119):
        recorder.set_context(decision_step, np.eye(4), current)
        for _ in range(25):
            recorder.step()
    assert [row["step"] for row in samples] == [118] * 25 + [119] * 25
    with pytest.raises(ValueError):
        recorder.set_context(120, np.eye(4), current)
