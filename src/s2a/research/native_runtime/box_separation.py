"""Offline separation-axis checks for two finite oriented boxes."""

import numpy as np


def box_separation(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    """Return the greatest signed projection gap over all 15 OBB separating axes.

    A strictly positive gap proves separation. A strictly negative maximum means
    overlap on every nondegenerate axis; near-zero results need numerical care.
    """
    ca, ra, ha, cb, rb, hb = (
        np.asarray(v, dtype=float) for v in (center_a, rotation_a, half_a, center_b, rotation_b, half_b)
    )
    if (
        ca.shape != (3,)
        or cb.shape != (3,)
        or ha.shape != (3,)
        or hb.shape != (3,)
        or ra.shape != (3, 3)
        or rb.shape != (3, 3)
        or not all(np.isfinite(v).all() for v in (ca, ra, ha, cb, rb, hb))
        or np.any(ha <= 0)
        or np.any(hb <= 0)
    ):
        raise ValueError("Two finite boxes with positive half sizes are required")
    for rotation in (ra, rb):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(
            np.linalg.det(rotation), 1, atol=1e-10, rtol=0
        ):
            raise ValueError("Box rotations must be proper orthogonal matrices")
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    best_gap, best_axis, tested = -np.inf, None, 0
    for raw in axes:
        norm = float(np.linalg.norm(raw))
        if norm <= 1e-12:
            continue
        axis = raw / norm
        gap = float(abs((cb - ca) @ axis) - (np.abs(ra.T @ axis) @ ha + np.abs(rb.T @ axis) @ hb))
        tested += 1
        if gap > best_gap:
            best_gap, best_axis = gap, axis.copy()
    return dict(maximum_gap_m=best_gap, witness_axis_world=best_axis, tested_axes=tested)
