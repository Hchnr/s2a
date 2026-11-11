"""Verify directional joint and rotation bounds against material-point motion."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.directional_slide_box_projection import (
    project_directional_slide_boxes_reference,
)
from s2a.research.joint_control.tracking.parallel_box_reference_projection import project_parallel_boxes_reference

VERTICES = np.array(list(product((-0.01, 0.01), repeat=3)))
OPTIONS = dict(
    position_uncertainty=0.01,
    rotation_uncertainty=0.05,
    additional_shape_uncertainty=0.002,
    clearance=0.001,
    numerical_margin=1e-9,
    max_translation=0.2,
)


def boxes():
    return [
        ("table", transform(np.array([0.0, 0.0, -0.02]), np.eye(3)), np.array([0.2, 0.2, 0.02])),
        ("stove", transform(np.array([0.0, 0.0, 0.012]), np.eye(3)), np.array([0.1, 0.1, 0.0005])),
    ]


def test_directional_envelope_covers_joint_rotation_position_and_residual():
    rng = np.random.default_rng(42)
    desired = transform(np.array([0.0, 0.0, 0.0]), Rotation.from_euler("xyz", [0.2, -0.1, 0.1]).as_matrix())
    slide = np.array([0.0, 1.0, 0.0])
    result = project_directional_slide_boxes_reference(desired, VERTICES, slide, -0.03, -0.04, 0.0, boxes(), **OPTIONS)
    assert result["accepted"]
    projected = result["projected_tcp"]
    for _ in range(512):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * rng.uniform(0, 0.05)).as_matrix() @ projected[:3, :3]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = projected[:3, 3] + direction * rng.uniform(0, 0.01)
        q = rng.uniform(-0.04, 0.0)
        residual = rng.normal(size=VERTICES.shape)
        residual /= np.linalg.norm(residual, axis=1)[:, None]
        residual *= rng.uniform(0, 0.002, size=(len(VERTICES), 1))
        actual = (VERTICES + slide * (q + 0.03) + residual) @ rotation.T + position
        for _, pose, half_size in boxes():
            local = (actual - pose[:3, 3]) @ pose[:3, :3]
            assert np.min(local[:, 2]) >= half_size[2] + 0.001 - 1e-12


def test_directional_bound_never_exceeds_spherical_bound_on_the_same_states():
    rng = np.random.default_rng(7)
    for _ in range(50):
        rotation = Rotation.random(random_state=rng).as_matrix()
        desired = transform(rng.uniform(-0.03, 0.03, size=3), rotation)
        slide = rng.normal(size=3)
        slide /= np.linalg.norm(slide)
        q = rng.uniform(-0.04, 0.0)
        directed = project_directional_slide_boxes_reference(
            desired, VERTICES, slide, q, -0.04, 0.0, boxes(), **OPTIONS
        )
        spherical = project_parallel_boxes_reference(
            desired,
            VERTICES,
            boxes(),
            position_uncertainty=OPTIONS["position_uncertainty"],
            rotation_uncertainty=OPTIONS["rotation_uncertainty"],
            shape_uncertainty=max(q + 0.04, -q) + OPTIONS["additional_shape_uncertainty"],
            clearance=OPTIONS["clearance"],
            numerical_margin=OPTIONS["numerical_margin"],
            max_translation=OPTIONS["max_translation"],
        )
        for name in ("table", "stove"):
            assert (
                directed["boxes"][name]["required_translation_m"]
                <= spherical["boxes"][name]["required_translation_m"] + 1e-12
            )


def test_future_slide_can_create_a_box_footprint_overlap():
    box = [("box", np.eye(4), np.array([0.1, 0.1, 0.02]))]
    desired = transform(np.array([0.12, 0, 0]), np.eye(3))
    result = project_directional_slide_boxes_reference(
        desired,
        VERTICES,
        np.array([-1.0, 0, 0]),
        0.0,
        0.0,
        0.04,
        box,
        **{**OPTIONS, "position_uncertainty": 0.0, "rotation_uncertainty": 0.0, "additional_shape_uncertainty": 0.0},
    )
    assert result["boxes"]["box"]["uncertainty_expanded_xy_overlap"]
    assert result["required_translation_m"] > 0.0


def test_directional_bound_rejects_out_of_range_or_unsupported_geometry():
    desired = np.eye(4)
    with pytest.raises(ValueError, match="ordered interval"):
        project_directional_slide_boxes_reference(desired, VERTICES, [0, 1, 0], -0.041, -0.04, 0, boxes(), **OPTIONS)
    with pytest.raises(ValueError, match="unit length"):
        project_directional_slide_boxes_reference(desired, VERTICES, [0, 2, 0], -0.03, -0.04, 0, boxes(), **OPTIONS)
    tilted = boxes()
    tilted[1] = ("stove", transform(np.zeros(3), Rotation.from_euler("x", 0.1).as_matrix()), np.ones(3))
    with pytest.raises(ValueError, match="normals"):
        project_directional_slide_boxes_reference(desired, VERTICES, [0, 1, 0], -0.03, -0.04, 0, tilted, **OPTIONS)
    refused = project_directional_slide_boxes_reference(
        desired, VERTICES, [0, 1, 0], -0.03, -0.04, 0, boxes(), **{**OPTIONS, "max_translation": 0.001}
    )
    assert not refused["accepted"] and refused["projected_tcp"] is None
