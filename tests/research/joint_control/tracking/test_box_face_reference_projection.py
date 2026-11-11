"""Check conditional multidirectional projections and their restrictions."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.box_face_reference_projection import project_box_faces_reference
from s2a.research.joint_control.tracking.directional_slide_box_projection import (
    project_directional_slide_boxes_reference,
)

VERTICES = np.array(list(product((-0.005, 0.005), repeat=3)))
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
        ("table", transform(np.array([0.0, 0.0, -0.02]), np.eye(3)), np.array([0.3, 0.3, 0.02])),
        ("stove", transform(np.array([0.0, 0.0, 0.02]), np.eye(3)), np.array([0.05, 0.05, 0.01])),
    ]


def test_sideways_can_avoid_stove_with_less_translation_than_vertical():
    target = transform(np.array([0.054, 0, 0.02]), np.eye(3))
    opts = {**OPTIONS, "position_uncertainty": 0.0, "rotation_uncertainty": 0.0, "additional_shape_uncertainty": 0.0}
    lateral = project_box_faces_reference(target, VERTICES, [0, 1, 0], 0.0, 0.0, 0.0, boxes(), **opts)
    vertical = project_directional_slide_boxes_reference(target, VERTICES, [0, 1, 0], 0.0, 0.0, 0.0, boxes(), **opts)
    assert lateral["accepted"] and lateral["face_combinations_checked"] == 36
    assert lateral["required_translation_m"] == pytest.approx(0.002000001)
    assert lateral["required_translation_m"] < vertical["required_translation_m"]
    np.testing.assert_allclose(lateral["translation_world_m"], [0.002000001, 0, 0], atol=1e-12)


def test_selected_faces_cover_random_joint_and_pose_perturbations():
    rng = np.random.default_rng(42)
    target = transform(np.array([0.04, 0, 0.02]), Rotation.from_euler("xyz", [0.1, -0.1, 0.2]).as_matrix())
    slide = np.array([0.0, 1.0, 0.0])
    result = project_box_faces_reference(target, VERTICES, slide, -0.03, -0.04, 0, boxes(), **OPTIONS)
    assert result["accepted"]
    projected = result["projected_tcp"]
    for _ in range(512):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * rng.uniform(0, 0.05)).as_matrix() @ projected[:3, :3]
        shift = rng.normal(size=3)
        shift /= np.linalg.norm(shift)
        position = projected[:3, 3] + shift * rng.uniform(0, 0.01)
        residual = rng.normal(size=VERTICES.shape)
        residual /= np.linalg.norm(residual, axis=1)[:, None]
        residual *= rng.uniform(0, 0.002, size=(len(VERTICES), 1))
        q = rng.uniform(-0.04, 0.0)
        actual = (VERTICES + slide * (q + 0.03) + residual) @ rotation.T + position
        for (_, pose, size), face in zip(boxes(), result["selected_faces"], strict=True):
            local = (actual - pose[:3, 3]) @ pose[:3, :3]
            values = face["side"] * local[:, face["axis"]]
            assert np.min(values) >= size[face["axis"]] + 0.001 - 1e-12


def test_multiface_is_no_worse_than_registered_vertical_candidate():
    rng = np.random.default_rng(7)
    for _ in range(50):
        target = transform(rng.uniform(-0.04, 0.04, size=3), Rotation.random(random_state=rng).as_matrix())
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        q = rng.uniform(-0.04, 0.0)
        result = project_box_faces_reference(target, VERTICES, axis, q, -0.04, 0.0, boxes(), **OPTIONS)
        vertical = project_directional_slide_boxes_reference(target, VERTICES, axis, q, -0.04, 0.0, boxes(), **OPTIONS)
        assert result["required_translation_m"] <= vertical["required_translation_m"] + 1e-12


def test_bad_axes_out_of_range_and_excess_correction_are_refused():
    target = np.eye(4)
    with pytest.raises(ValueError, match="ordered interval"):
        project_box_faces_reference(target, VERTICES, [0, 1, 0], -0.041, -0.04, 0, boxes(), **OPTIONS)
    tilted = boxes()
    tilted[1] = ("stove", transform(np.zeros(3), Rotation.from_euler("z", 0.1).as_matrix()), np.ones(3))
    with pytest.raises(ValueError, match="axes"):
        project_box_faces_reference(target, VERTICES, [0, 1, 0], -0.03, -0.04, 0, tilted, **OPTIONS)
    refused = project_box_faces_reference(
        target, VERTICES, [0, 1, 0], -0.03, -0.04, 0, boxes(), **{**OPTIONS, "max_translation": 0.001}
    )
    assert not refused["accepted"] and refused["projected_tcp"] is None
