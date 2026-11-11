"""Reference projection preserves the stated conditional geometry and bounds."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.tabletop_reference_projection import (
    collision_vertices_in_tcp,
    project_tabletop_reference,
)


def project(desired, vertices, table=None, **overrides):
    options = {
        "position_uncertainty": 0.01,
        "rotation_uncertainty": 0.05,
        "shape_uncertainty": 0.0,
        "clearance": 0.001,
        "numerical_margin": 1e-9,
        "max_translation": 0.2,
    }
    options.update(overrides)
    table = transform(np.array([0.0, 0.0, -0.05]), np.eye(3)) if table is None else table
    return project_tabletop_reference(desired, vertices, table, np.array([0.2, 0.2, 0.05]), **options)


def test_projection_preserves_rotation_and_rejects_excessive_correction():
    desired = transform(np.array([0.0, 0.0, 0.0]), Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix())
    vertices = np.array(list(product((-0.01, 0.01), repeat=3)))
    before = desired.copy()
    result = project(desired, vertices)
    assert result["accepted"] is True and result["status"] == "projected"
    assert result["projected_height_lower_bound_m"] >= 0.001
    np.testing.assert_array_equal(desired, before)
    np.testing.assert_array_equal(result["projected_tcp"][:3, :3], desired[:3, :3])
    np.testing.assert_allclose(
        result["projected_tcp"][:3, 3] - desired[:3, 3], [0, 0, result["required_translation_m"]], atol=1e-12
    )
    refused = project(desired, vertices, max_translation=0.001)
    assert refused["accepted"] is False and refused["status"] == "correction_exceeds_limit"
    assert refused["projected_tcp"] is None and refused["conditional_table_clearance_certified"] is False


def test_expanded_footprint_accounts_for_uncertainty_and_clearance():
    vertices = np.array(list(product((-0.01, 0.01), repeat=3)))
    far = transform(np.array([0.23, 0, 0]), np.eye(3))
    assert project(far, vertices, position_uncertainty=0.005)["status"] == "unchanged"
    assert project(far, vertices, position_uncertainty=0.03)["status"] == "projected"
    near_side = transform(np.array([0.2105, 0, 0]), np.eye(3))
    assert project(near_side, vertices, position_uncertainty=0, rotation_uncertainty=0)["status"] == "projected"


def test_conditional_clearance_under_bounded_pose_and_shape_changes():
    rng = np.random.default_rng(42)
    vertices = np.array(list(product((-0.015, 0.015), repeat=3)))
    desired = transform(np.array([0.0, 0.0, 0.0]), Rotation.from_euler("xyz", [0.25, -0.1, 0.2]).as_matrix())
    result = project(desired, vertices, position_uncertainty=0.01, rotation_uncertainty=0.05, shape_uncertainty=0.004)
    assert result["accepted"] and result["conditional_table_clearance_certified"]
    projected = result["projected_tcp"]
    for _ in range(512):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * rng.uniform(0, 0.05)).as_matrix() @ projected[:3, :3]
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = projected[:3, 3] + direction * rng.uniform(0, 0.01)
        changes = rng.normal(size=vertices.shape)
        changes /= np.linalg.norm(changes, axis=1)[:, None]
        changes *= rng.uniform(0, 0.004, size=(len(vertices), 1))
        actual = (vertices + changes) @ rotation.T + position
        assert np.min(actual[:, 2]) >= 0.001 - 1e-12


def test_common_rigid_transform_and_current_tcp_geometry_conversion():
    tcp = transform(np.array([0.1, -0.2, 0.3]), Rotation.from_euler("xyz", [0.4, -0.2, 0.3]).as_matrix())
    geom = transform(np.array([-0.1, 0.05, 0.2]), Rotation.from_euler("xyz", [-0.3, 0.1, 0.2]).as_matrix())
    verts_geom = np.array([[0, 0, 0], [0.01, -0.02, 0.03]])
    vertices = collision_vertices_in_tcp(verts_geom, tcp, geom)
    np.testing.assert_allclose(
        vertices @ tcp[:3, :3].T + tcp[:3, 3], verts_geom @ geom[:3, :3].T + geom[:3, 3], atol=1e-12
    )
    table = transform(np.array([0.0, 0.0, -0.05]), np.eye(3))
    first = project(tcp, vertices, table)
    common = transform(np.array([0.2, 0.1, -0.4]), Rotation.from_euler("xyz", [0.3, 0.2, -0.1]).as_matrix())
    second = project(common @ tcp, vertices, common @ table)
    assert first["status"] == second["status"]
    np.testing.assert_allclose(first["required_translation_m"], second["required_translation_m"], atol=1e-12)
    if first["accepted"]:
        np.testing.assert_allclose(common @ first["projected_tcp"], second["projected_tcp"], atol=1e-12)


def test_invalid_uncertainty_or_geometry_is_rejected():
    with pytest.raises(ValueError):
        project(np.eye(4), np.zeros((0, 3)))
    with pytest.raises(ValueError):
        project(np.eye(4), [[0, 0, 0]], rotation_uncertainty=np.pi + 0.1)
    with pytest.raises(ValueError):
        project(np.eye(4), [[0, 0, 0]], shape_uncertainty=-0.1)
