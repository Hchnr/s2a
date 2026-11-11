"""Exercise the box and kinematic assumptions used by the static study."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.parallel_box_reference_projection import (
    project_parallel_boxes_reference,
    single_slide_shape_bound,
)


def project(boxes, **overrides):
    options = dict(
        position_uncertainty=0.01,
        rotation_uncertainty=0.05,
        shape_uncertainty=0.004,
        clearance=0.001,
        numerical_margin=1e-9,
        max_translation=0.15,
    )
    options.update(overrides)
    desired = transform(np.array([0.0, 0.0, 0.0]), np.eye(3))
    vertices = np.array(list(product((-0.01, 0.01), repeat=3)))
    return project_parallel_boxes_reference(desired, vertices, boxes, **options)


def box(name, center, size=(0.2, 0.2, 0.02), rotation=None):
    return name, transform(np.asarray(center), np.eye(3) if rotation is None else rotation), np.asarray(size)


def test_parallel_boxes_take_maximum_lift_and_preserve_conditional_clearance():
    boxes = [box("table", [0, 0, -0.02]), box("stove", [0, 0, 0.012], (0.1, 0.1, 0.0005))]
    result = project(boxes)
    assert result["accepted"] and result["conditional_clearance_certified"]
    assert result["required_translation_m"] == max(row["required_translation_m"] for row in result["boxes"].values())
    assert result["boxes"]["stove"]["required_translation_m"] > result["boxes"]["table"]["required_translation_m"]
    projected = result["projected_tcp"]
    vertices = np.array(list(product((-0.01, 0.01), repeat=3)))
    rng = np.random.default_rng(42)
    for _ in range(128):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * rng.uniform(0, 0.05)).as_matrix() @ projected[:3, :3]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = projected[:3, 3] + direction * rng.uniform(0, 0.01)
        change = rng.normal(size=vertices.shape)
        change /= np.linalg.norm(change, axis=1)[:, None]
        change *= rng.uniform(0, 0.004, size=(len(vertices), 1))
        actual = (vertices + change) @ rotation.T + position
        for _, pose, size in boxes:
            local = (actual - pose[:3, 3]) @ pose[:3, :3]
            assert np.min(local[:, 2]) >= size[2] + 0.001 - 1e-12


def test_nonoverlapping_box_does_not_force_lift_and_cap_refuses_combined_target():
    table = box("table", [0, 0, -0.02])
    remote = box("stove", [0.5, 0, 0.1])
    one, two = project([table]), project([table, remote])
    assert two["boxes"]["stove"]["uncertainty_expanded_xy_overlap"] is False
    assert one["required_translation_m"] == two["required_translation_m"]
    refused = project([table, box("stove", [0, 0, 0.04])], max_translation=0.01)
    assert not refused["accepted"] and refused["projected_tcp"] is None
    assert not refused["conditional_clearance_certified"]


def test_inconsistent_normals_and_ambiguous_box_names_are_rejected():
    table = box("table", [0, 0, -0.02])
    tilted = box("stove", [0, 0, 0], rotation=Rotation.from_euler("x", 0.1).as_matrix())
    with pytest.raises(ValueError, match="normals"):
        project([table, tilted])
    with pytest.raises(ValueError, match="names"):
        project([table, table])
    with pytest.raises(ValueError, match="At least"):
        project([])


def test_slide_range_bound_is_conditional_and_rejects_outside_positions():
    assert single_slide_shape_bound(-0.03, -0.04, 0.0) == pytest.approx(0.03)
    assert single_slide_shape_bound(-0.04, -0.04, 0.0) == pytest.approx(0.04)
    for invalid in (0.001, -0.041, np.nan, True):
        with pytest.raises(ValueError):
            single_slide_shape_bound(invalid, -0.04, 0.0)
    with pytest.raises(ValueError):
        single_slide_shape_bound(0.0, 1.0, -1.0)
