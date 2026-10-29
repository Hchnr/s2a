"""Offline convex-hull/box SAT; signed axis gap is not Euclidean distance."""

from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from s2a.geometry import finite


def axis_clearance(vertices, box_pose, half_size):
    """Positive means a separating axis; negative means convex hull overlap.

    This tests the convex hull of the supplied world vertices, not an arbitrary
    nonconvex visual mesh. Values are projection gaps along unit SAT axes.
    """
    vertices = finite(vertices)
    pose, size = finite(box_pose, (4, 4)), finite(half_size, (3,))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or np.any(size <= 0):
        raise ValueError("A full dimensional point cloud and positive box required")
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-10, rtol=0)
    hull = ConvexHull(vertices)
    box_axes = pose[:3, :3].T
    edges = {tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) for face in hull.simplices for i in range(3)}
    directions = np.asarray([vertices[j] - vertices[i] for i, j in sorted(edges)])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    cross = np.cross(directions[:, None, :], box_axes[None, :, :]).reshape(-1, 3)
    axes = np.vstack((hull.equations[:, :3], box_axes, cross))
    norm = np.linalg.norm(axes, axis=1)
    axes = axes[norm > 1e-10] / norm[norm > 1e-10, None]
    corners = np.asarray(list(product((-1, 1), repeat=3))) * size
    corners = corners @ pose[:3, :3].T + pose[:3, 3]
    a, b = vertices @ axes.T, corners @ axes.T
    gaps = np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0))
    index = int(np.argmax(gaps))
    return dict(maximum_axis_gap_m=float(gaps[index]), separating_axis=axes[index], tested_axes=len(axes))
