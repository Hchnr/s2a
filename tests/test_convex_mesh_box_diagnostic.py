"""SAT detects separation/overlap consistently under common rigid transforms."""

from itertools import product

import numpy as np

from s2a.convex_mesh_box_diagnostic import axis_clearance


def test_known_box_gaps_and_common_rigid_transform():
    vertices = np.asarray(list(product((-0.5, 0.5), repeat=3)))
    pose = np.eye(4)
    size = np.full(3, 0.5)
    for offset, expected in [(1.25, 0.25), (1.0, 0.0), (0.75, -0.25)]:
        moved = vertices + [offset, 0, 0]
        result = axis_clearance(moved, pose, size)
        np.testing.assert_allclose(result["maximum_axis_gap_m"], expected, atol=1e-12, rtol=0)
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        transformed = pose.copy()
        transformed[:3, :3] = rotation
        transformed[:3, 3] = [0.2, -0.3, 0.4]
        result = axis_clearance(moved @ rotation.T + transformed[:3, 3], transformed, size)
        np.testing.assert_allclose(result["maximum_axis_gap_m"], expected, atol=1e-12, rtol=0)
