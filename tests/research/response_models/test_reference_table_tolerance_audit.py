"""The independent scalar audit must agree with valid witnesses and reject changes."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import inverse, transform
from s2a.research.joint_control.tracking.pose_box_collision_certificate import fixed_vertex_box_certificate
from scripts.verification.reference_ports import audit_reference_table_tolerance as audit


def test_independent_target_finger_transform_handles_noncommuting_rotations():
    site = transform(np.array([0.1, -0.2, 0.3]), Rotation.from_euler("xyz", [0.4, -0.1, 0.2]).as_matrix())
    finger_local = transform(np.array([0.03, 0.04, -0.06]), Rotation.from_euler("xyz", [0.2, 0.1, -0.3]).as_matrix())
    site_from_tcp = transform(np.array([0, 0, 0.1]), Rotation.from_euler("z", 0.5).as_matrix())
    target = transform(np.array([-0.2, 0.1, 0.4]), Rotation.from_euler("xyz", [-0.2, 0.4, 0.3]).as_matrix())
    actual = audit.target_finger_pose(site, site @ finger_local, target, site_from_tcp)
    np.testing.assert_allclose(actual, target @ inverse(site_from_tcp) @ finger_local, atol=1e-12, rtol=0)


def test_scalar_audit_reconstructs_witness_and_rejects_tampering():
    finger = transform(np.array([0.02, -0.01, 0.03]), Rotation.from_euler("xyz", [0.3, 0.4, -0.2]).as_matrix())
    tcp = transform(np.array([-0.03, 0.04, 0.06]), Rotation.from_euler("xyz", [0.1, -0.3, 0.2]).as_matrix())
    table = transform(np.array([0.01, 0.01, 0.01]), Rotation.from_euler("xyz", [-0.1, 0.2, 0.3]).as_matrix())
    vertices = np.array([[0.0, 0.0, 0.0], [0.2, 0.15, -0.1], [-0.1, 0.1, 0.3]])
    world = vertices @ finger[:3, :3].T + finger[:3, 3]
    saved = fixed_vertex_box_certificate(
        world, tcp, table, np.ones(3), position_tolerance=0.01, rotation_tolerance=0.05, margin=1e-9
    )
    scores = audit.independent_scores(vertices, finger, tcp, table, np.ones(3), 0.01, 0.05)
    assert audit.verify_certificate(scores, saved, margin=1e-9)["certified"] is True
    damaged = copy.deepcopy(saved)
    damaged["tcp_radius_m"] += 0.001
    with pytest.raises(AssertionError):
        audit.verify_certificate(scores, damaged, margin=1e-9)


def test_original_study_and_independent_budget_are_source_bound():
    inputs = audit.load_inputs()["reference_table_tolerance_audit"]
    assert inputs["config"]["steps"] == list(range(120, 127))
    assert inputs["config"]["max_forward_queries"] == 7
    assert inputs["original"]["stats"]["physics_steps_attempted"] == 0
