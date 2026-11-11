"""Telemetry must align a decision, inner prestate and post-integration sample."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime.tabletop_projection_inner_recording import ProjectionInnerRecorder


class FakeBase:
    def __init__(self):
        self.samples = []
        self.original = lambda: None

    def step(self):
        self.samples.append({"time_before": 1.0, "time_after": 1.002})
        return "stepped"


def test_recorder_requires_decision_and_measures_at_correct_time():
    controller = SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3))
    robot = SimpleNamespace(controller=controller, gripper=SimpleNamespace(current_action=np.array([0.1, -0.1])))
    env = SimpleNamespace(robots=[robot])
    vertices = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    sensor = SimpleNamespace(capture=lambda: {"tcp": np.eye(4), "vertices_tcp": vertices.copy(), "time_s": 1.0})
    recorder = ProjectionInnerRecorder(env, sensor, "pose_bounded", base=FakeBase())
    with pytest.raises(ValueError):
        recorder.step()
    assert recorder.samples == []
    recorder.set_context(120, np.eye(4), vertices)
    assert recorder.step() == "stepped"
    row = recorder.samples[0]
    assert row["step"] == 120 and row["bounded_sample_conditions"]["all_sample_bounds_met"]
    assert row["geometry_sample_phase"] == "after_environment_forward_before_integration"
    assert row["contact_sample_phase"] == "last_inner_step_solver_cache_after_integration"
    with pytest.raises(ValueError):
        recorder.set_context(121, np.eye(4), vertices)
