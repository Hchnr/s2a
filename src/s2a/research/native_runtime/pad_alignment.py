"""One-dimensional projections of saved pad and object box geometry."""

import numpy as np


def project_box(center, rotation, half_size, origin, axis):
    """Return the exact interval of an oriented box along a unit world axis."""
    center, rotation, half_size, origin, axis = (
        np.asarray(value, dtype=float) for value in (center, rotation, half_size, origin, axis)
    )
    if (
        center.shape != (3,)
        or rotation.shape != (3, 3)
        or half_size.shape != (3,)
        or origin.shape != (3,)
        or axis.shape != (3,)
        or not all(np.isfinite(v).all() for v in (center, rotation, half_size, origin, axis))
        or np.any(half_size <= 0)
        or not np.isclose(np.linalg.norm(axis), 1, atol=1e-10, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Finite oriented box, proper rotation and unit projection axis required")
    midpoint = float(np.dot(center - origin, axis))
    radius = float(np.dot(np.abs(rotation.T @ axis), half_size))
    return [midpoint - radius, midpoint + radius]


def alignment(left, right, body, site_position, site_rotation):
    """Describe origin-to-pad geometry, without inferring contact or graspability."""
    left, right, body, site_position, site_rotation = (
        np.asarray(v, dtype=float) for v in (left, right, body, site_position, site_rotation)
    )
    if (
        left.shape != (3,)
        or right.shape != (3,)
        or body.shape != (3,)
        or site_position.shape != (3,)
        or site_rotation.shape != (3, 3)
        or not all(np.isfinite(v).all() for v in (left, right, body, site_position, site_rotation))
        or not np.allclose(site_rotation.T @ site_rotation, np.eye(3), atol=1e-10, rtol=0)
        or not np.isclose(np.linalg.det(site_rotation), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Finite pad, body and proper site geometry required")
    midpoint = (left + right) / 2
    separation = float(np.linalg.norm(right - left))
    if separation <= 1e-12:
        raise ValueError("Distinct pad centers are required to define a jaw axis")
    axis = (right - left) / separation
    vector = body - midpoint
    along = float(np.dot(vector, axis))
    return dict(
        left_pad_world_m=left,
        right_pad_world_m=right,
        pad_midpoint_world_m=midpoint,
        jaw_axis_world=axis,
        pad_center_separation_m=separation,
        body_origin_world_m=body,
        body_from_pad_midpoint_world_m=vector,
        body_axis_projection_m=along,
        body_axis_perpendicular_m=float(np.linalg.norm(vector - along * axis)),
        site_position_world_m=site_position,
        site_rotation_world=site_rotation,
        body_from_site_in_site_m=site_rotation.T @ (body - site_position),
    )
