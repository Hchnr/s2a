"""A positive certificate remains inside the box under pose and joint changes."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.pose_box_collision_certificate import fixed_vertex_box_certificate
from s2a.research.joint_control.tracking.slide_range_box_collision_certificate import (
    slide_range_vertex_box_certificate,
)

VERTICES = np.array(list(product((-0.01, 0.01), repeat=3)))
BOX = np.array([0.5, 0.6, 0.05])
OPTIONS = dict(
    position_tolerance=0.01, rotation_tolerance=0.05, additional_shape_uncertainty=0.0, numerical_margin=1e-9
)


def test_certificate_covers_allowed_slide_and_tcp_perturbations():
    rng = np.random.default_rng(42)
    desired = transform(np.array([0.0, 0.0, 0.0]), Rotation.from_euler("xyz", [0.1, -0.1, 0.2]).as_matrix())
    slide = np.array([1.0, 0, 0])
    result = slide_range_vertex_box_certificate(desired, VERTICES, slide, -0.03, -0.04, 0, np.eye(4), BOX, **OPTIONS)
    assert result["certified"] and not result["unproven_means_collision_free"]
    vertex = VERTICES[result["vertex_index"]]
    for _ in range(512):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * rng.uniform(0, 0.05)).as_matrix() @ desired[:3, :3]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = desired[:3, 3] + direction * rng.uniform(0, 0.01)
        q = rng.uniform(-0.04, 0)
        point = position + rotation @ (vertex + slide * (q + 0.03))
        assert np.all(np.abs(point) < BOX)


def test_fixed_joint_matches_existing_fixed_vertex_certificate():
    desired = np.eye(4)
    result = slide_range_vertex_box_certificate(
        desired, VERTICES, [1, 0, 0], -0.03, -0.03, -0.03, np.eye(4), BOX, **OPTIONS
    )
    old = fixed_vertex_box_certificate(
        VERTICES, desired, np.eye(4), BOX, position_tolerance=0.01, rotation_tolerance=0.05, margin=1e-9
    )
    assert result["certified"] == old["certified"]
    assert result["interior_slack_m"] == pytest.approx(old["certificate_slack_m"], abs=1e-12)


def test_outside_measured_joint_can_still_bound_allowed_target_range():
    result = slide_range_vertex_box_certificate(
        np.eye(4), VERTICES, [1, 0, 0], -0.041, -0.04, 0, np.eye(4), BOX, **OPTIONS
    )
    assert result["certified"] and not result["measured_joint_in_target_interval"]
    assert result["joint_target_interval_m"] == [-0.04, 0.0]


def test_unproven_is_not_clearance_and_invalid_inputs_are_rejected():
    thin = slide_range_vertex_box_certificate(
        np.eye(4), VERTICES, [1, 0, 0], -0.03, -0.04, 0, np.eye(4), [0.5, 0.6, 0.001], **OPTIONS
    )
    assert not thin["certified"] and not thin["unproven_means_collision_free"]
    with pytest.raises(ValueError, match="ordered target interval"):
        slide_range_vertex_box_certificate(np.eye(4), VERTICES, [1, 0, 0], -0.03, 0, -0.04, np.eye(4), BOX, **OPTIONS)
    with pytest.raises(ValueError, match="unit length"):
        slide_range_vertex_box_certificate(np.eye(4), VERTICES, [2, 0, 0], -0.03, -0.04, 0, np.eye(4), BOX, **OPTIONS)
