"""Record current slide conditions without certifying future physical motion."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime.box_face_feedback_measurement import (
    BoxFaceInnerRecorder,
    checked_slide_topology,
    slide_sample_conditions,
)


def sample(q, vertices):
    return dict(
        tcp=np.eye(4),
        vertices_tcp=np.asarray(vertices, dtype=float),
        time_s=1.0,
        joint_position_m=q,
        slide_axis_tcp=np.array([1.0, 0, 0]),
    )


def test_slide_samples_separately_record_range_and_kinematic_residual():
    first = sample(-0.03, [[0, 0, 0], [0.01, 0, 0]])
    later = sample(-0.029, [[0.001, 0, 0], [0.011, 0, 0]])
    result = slide_sample_conditions(first, later, np.eye(4), [-0.04, 0])
    assert result["finger_joint_range_met"] and result["position_bound_met"] and result["rotation_bound_met"]
    assert result["slide_vertex_residual_m"] < 1e-15
    later["joint_position_m"] = -0.041
    assert not slide_sample_conditions(first, later, np.eye(4), [-0.04, 0])["finger_joint_range_met"]
    with pytest.raises(ValueError, match="decision"):
        slide_sample_conditions(later, first, np.eye(4), [-0.04, 0])


def test_recorded_inner_samples_obey_period_budget():
    controller = SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3))
    env = SimpleNamespace(
        robots=[SimpleNamespace(controller=controller, gripper=SimpleNamespace(current_action=[0, 0]))]
    )
    current = sample(-0.03, [[0, 0, 0]])
    sensor = SimpleNamespace(capture=lambda: current, interval=np.array([-0.04, 0]))
    samples = []

    def step():
        samples.append({"time_before": 1.0, "time_after": 1.002})

    base = SimpleNamespace(samples=samples, original=lambda: None, step=step)
    recorder = BoxFaceInnerRecorder(env, sensor, periods=1, base=base)
    recorder.set_context(120, np.eye(4), current)
    for _ in range(25):
        recorder.step()
    assert all(row["step"] == 120 and row["bounded_sample_conditions"]["finger_joint_range_met"] for row in samples)
    with pytest.raises(ValueError):
        recorder.set_context(121, np.eye(4), current)


def test_topology_rejects_a_tcp_that_also_moves_with_another_joint():
    model = SimpleNamespace(
        nbody=4,
        body_parentid=np.array([0, 0, 1, 1]),
        body_jntadr=np.array([-1, -1, 0, -1]),
        body_jntnum=np.array([0, 0, 1, 0]),
        jnt_type=np.array([2]),
        jnt_limited=np.array([1]),
        jnt_axis=np.array([[0.0, 1.0, 0.0]]),
        jnt_range=np.array([[-0.04, 0.0]]),
    )
    np.testing.assert_array_equal(checked_slide_topology(model, 2, 3, 0, 2), [-0.04, 0.0])
    model.body_jntadr[3], model.body_jntnum[3] = 0, 1
    with pytest.raises(ValueError, match="one limited unit slide"):
        checked_slide_topology(model, 2, 3, 0, 2)


def test_measurement_import_keeps_checker_policy_isolated():
    code = """\
import sys
from scripts.simulation.runtime import box_face_feedback_measurement
from scripts.simulation.runtime.run_tabletop_projection_feedback_probe import checker_isolation
checker_isolation()
assert 's2a.research.joint_control.tracking.box_face_feedback' not in sys.modules
assert 's2a.research.joint_control.tracking.box_face_reference_projection' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
