"""The face candidate preserves the fixed interface and stops before invalid actions."""

from itertools import product

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import transform
from s2a.research.joint_control.tracking.box_face_feedback import select_box_face_feedback
from s2a.research.joint_control.tracking.tabletop_projection_feedback import select_tabletop_feedback


def select(*, q=-0.03, cap=0.05, far=False):
    interface = OSCInterface(-np.ones(6), np.ones(6), -0.05 * np.ones(6), 0.05 * np.ones(6))
    vertices = np.array(list(product((-0.01, 0.01), repeat=3)))
    pose = transform(np.array([0.0, 0.0, -0.05 if not far else -1.0]), np.eye(3))
    boxes = [("table", pose, np.array([0.2, 0.2, 0.05])), ("stove", pose, np.array([0.1, 0.1, 0.02]))]
    previous = np.array([0.1, -0.1, 0, 0, 0, 0, -0.6])
    result = select_box_face_feedback(
        np.eye(4),
        np.eye(4),
        np.eye(4),
        vertices,
        [0, 1, 0],
        q,
        [-0.04, 0],
        boxes,
        previous,
        interface,
        position_uncertainty=0.01,
        rotation_uncertainty=0.05,
        additional_shape_uncertainty=0,
        clearance=0.001,
        numerical_margin=1e-9,
        max_translation=cap,
        command_step_limit=0.25,
    )
    return result, previous, interface, vertices, pose


def test_face_feedback_keeps_gripper_and_original_rate_limit():
    result, previous, *_ = select()
    assert result["newly_selected"] is True and result["projection"]["accepted"]
    assert result["command"][6] == previous[6]
    assert np.max(np.abs(result["command"][:6] - previous[:6])) <= 0.25
    np.testing.assert_array_equal(result["original_target_tcp"], np.eye(4))
    np.testing.assert_array_equal(result["selected_target_tcp"][:3, :3], np.eye(3))


def test_out_of_range_and_excessive_correction_never_dispatch():
    outside, *_ = select(q=-0.041)
    rejected, *_ = select(cap=0.001)
    assert outside["stop"]["reason"] == "finger_joint_outside_declared_range"
    assert rejected["stop"]["reason"] == "projection_correction_exceeds_limit"
    for result in (outside, rejected):
        assert result["newly_selected"] is False and "command" not in result


def test_clear_reference_matches_the_original_fixed_interface():
    result, previous, interface, vertices, pose = select(far=True)
    original = select_tabletop_feedback(
        np.eye(4),
        np.eye(4),
        np.eye(4),
        vertices,
        pose,
        [0.2, 0.2, 0.05],
        previous,
        interface,
        mode="original",
        clearance=0.001,
        numerical_margin=1e-9,
        max_translation=0.05,
        command_step_limit=0.25,
    )
    assert result["reference_translation_m"] == 0.0
    np.testing.assert_allclose(result["command"], original["command"], atol=0, rtol=0)
