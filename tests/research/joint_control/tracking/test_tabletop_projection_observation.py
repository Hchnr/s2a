"""Measured shape and pose conditions use the actual sample without hidden slack."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.tabletop_projection_observation import (
    measured_vertices_in_tcp,
    observed_bound_conditions,
)


def test_current_vertex_reconstruction_survives_common_frame_change():
    vertices = np.array([[0.01, 0.02, -0.03], [-0.01, 0.0, 0.02]])
    tcp = transform(np.array([0.1, 0.2, 0.3]), Rotation.from_euler("xyz", [0.1, 0.2, -0.3]).as_matrix())
    geom = transform(np.array([-0.2, 0.1, 0.3]), Rotation.from_euler("xyz", [-0.2, 0.4, 0.1]).as_matrix())
    common = transform(np.array([0.4, -0.1, 0.2]), Rotation.from_euler("xyz", [0.3, 0.2, 0.1]).as_matrix())
    actual = measured_vertices_in_tcp(vertices, tcp, geom)
    transformed = measured_vertices_in_tcp(vertices, common @ tcp, common @ geom)
    np.testing.assert_allclose(actual, transformed, atol=1e-12, rtol=0)


def test_zero_shape_assumption_does_not_hide_even_a_small_observed_deformation():
    vertices = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    moved = vertices.copy()
    moved[0, 0] += 1e-10
    result = observed_bound_conditions(
        vertices, moved, np.eye(4), np.eye(4), position_bound=0.01, rotation_bound=0.05, shape_bound=0
    )
    assert result["position_bound_met"] and result["rotation_bound_met"]
    assert not result["shape_bound_met"] and not result["all_sample_bounds_met"]
    assert result["max_material_vertex_displacement_tcp_m"] > 0
