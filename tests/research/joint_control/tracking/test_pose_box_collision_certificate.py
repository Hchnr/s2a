"""Exercise the SE(3) bound and oriented-box witness, including near misses."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.pose_box_collision_certificate import fixed_vertex_box_certificate


def certify(points, tcp=None, box=None, *, p=0.01, theta=0.05, margin=1e-9):
    return fixed_vertex_box_certificate(
        np.asarray(points, dtype=float),
        np.eye(4) if tcp is None else tcp,
        np.eye(4) if box is None else box,
        np.ones(3),
        position_tolerance=p,
        rotation_tolerance=theta,
        margin=margin,
    )


def test_strict_inside_witness_and_inconclusive_near_boundary():
    good = certify([[0, 0, 0]])
    assert good["certified"] is True and good["inside_margin_m"] == 1
    assert good["unproven_means_collision_free"] is False
    assert certify([[0.995, 0, 0]])["certified"] is False
    assert certify([[1.1, 0, 0]])["certified"] is False
    assert certify([[0.875, 0, 0]], p=0.125, theta=0, margin=0)["certified"] is False


def test_certificate_is_invariant_under_common_rigid_transform():
    tcp = transform(np.array([0.1, 0.2, 0.3]), Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix())
    box = transform(np.array([0.2, 0.15, 0.25]), Rotation.from_euler("xyz", [-0.3, 0.2, 0.1]).as_matrix())
    points = np.array([[0.2, 0.15, 0.25], [0.4, 0.3, 0.1]])
    first = certify(points, tcp, box)
    common = transform(np.array([-0.3, 0.1, 0.7]), Rotation.from_euler("xyz", [0.5, -0.4, 0.3]).as_matrix())
    moved = points @ common[:3, :3].T + common[:3, 3]
    second = certify(moved, common @ tcp, common @ box)
    assert first["certified"] == second["certified"] and first["vertex_index"] == second["vertex_index"]
    np.testing.assert_allclose(first["certificate_slack_m"], second["certificate_slack_m"], atol=1e-12, rtol=0)


def test_certified_material_point_stays_inside_for_bounded_pose_perturbations():
    rng = np.random.default_rng(42)
    tcp = transform(np.array([0.12, -0.08, 0.2]), Rotation.from_euler("xyz", [0.4, 0.2, -0.1]).as_matrix())
    box = transform(np.array([0.03, 0.05, 0.08]), Rotation.from_euler("xyz", [-0.3, 0.1, 0.2]).as_matrix())
    points = np.array([[0.05, 0.07, 0.10], [0.3, -0.1, 0.2]])
    result = certify(points, tcp, box, p=0.05, theta=0.25)
    assert result["certified"] is True
    point = points[result["vertex_index"]]
    local_tcp = tcp[:3, :3].T @ (point - tcp[:3, 3])
    for _ in range(512):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        delta_rotation = Rotation.from_rotvec(axis * rng.uniform(0, 0.25)).as_matrix()
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = tcp[:3, 3] + direction * rng.uniform(0, 0.05)
        moved = position + delta_rotation @ tcp[:3, :3] @ local_tcp
        box_local = box[:3, :3].T @ (moved - box[:3, 3])
        assert np.all(np.abs(box_local) < 1)


def test_invalid_geometry_or_tolerances_are_rejected():
    with pytest.raises(ValueError):
        certify(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        certify([[0, 0, 0]], theta=np.pi + 0.1)
