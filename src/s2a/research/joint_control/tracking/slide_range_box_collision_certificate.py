"""Sufficient collision certificate for a material vertex on one slide."""

from numbers import Real

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.tabletop_reference_projection import _nonnegative, _strict_pose


def slide_range_vertex_box_certificate(
    desired_tcp,
    vertices_tcp,
    slide_axis_tcp,
    measured_joint,
    joint_lower,
    joint_upper,
    box_pose,
    box_half_size,
    *,
    position_tolerance,
    rotation_tolerance,
    additional_shape_uncertainty,
    numerical_margin,
):
    """Prove one material vertex stays strictly inside a fixed box.

    The supplied vertices and unit slide axis describe current TCP-relative
    geometry at measured_joint. Under the verified single-slide relation, any
    allowed target joint position adds axis*(joint-measured_joint), plus the
    stated bounded residual. measured_joint need not be in the target range.
    A positive result covers every supplied TCP pose and joint condition for
    that one vertex; a negative result is inconclusive, not a clearance proof.
    """
    desired, box = _strict_pose(desired_tcp), _strict_pose(box_pose)
    vertices, axis, size = finite(vertices_tcp), finite(slide_axis_tcp, (3,)), finite(box_half_size, (3,))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or np.any(size <= 0):
        raise ValueError("Finite material vertices and a positive box are required")
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-12, rtol=0):
        raise ValueError("The TCP-frame slide axis must have unit length")
    values = (measured_joint, joint_lower, joint_upper)
    if (
        any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) for value in values)
        or joint_lower > joint_upper
    ):
        raise ValueError("A finite measured joint and ordered target interval are required")
    position = _nonnegative(position_tolerance, "position_tolerance")
    rotation = _nonnegative(rotation_tolerance, "rotation_tolerance")
    shape = _nonnegative(additional_shape_uncertainty, "additional_shape_uncertainty")
    numerical = _nonnegative(numerical_margin, "numerical_margin")
    if rotation > np.pi:
        raise ValueError("rotation_tolerance must be at most pi")

    delta_min, delta_max = float(joint_lower - measured_joint), float(joint_upper - measured_joint)
    axis_error = 2 * np.sin(rotation / 2)
    world = vertices @ desired[:3, :3].T + desired[:3, 3]
    local = (world - box[:3, 3]) @ box[:3, :3]
    nominal_axis = (desired[:3, :3] @ axis) @ box[:3, :3]
    axis_lower = np.clip(nominal_axis - axis_error, -1.0, 1.0)
    axis_upper = np.clip(nominal_axis + axis_error, -1.0, 1.0)
    possibilities = np.array(
        [axis_lower * delta_min, axis_lower * delta_max, axis_upper * delta_min, axis_upper * delta_max]
    )
    slide_lower, slide_upper = np.min(possibilities, axis=0), np.max(possibilities, axis=0)
    vertex_error = position + axis_error * np.linalg.norm(vertices, axis=1) + shape
    lower = local + slide_lower[None, :] - vertex_error[:, None]
    upper = local + slide_upper[None, :] + vertex_error[:, None]
    slack = np.min(np.minimum(lower + size[None, :], size[None, :] - upper), axis=1)
    index = int(np.argmax(slack))
    scale = max(1.0, float(np.max(np.abs(world))), float(np.max(np.abs(box[:3, 3]))), float(np.max(size)))
    guard = max(numerical, 64 * np.finfo(float).eps * scale)
    return {
        "certified": bool(slack[index] > guard),
        "vertex_index": index,
        "vertex_tcp": vertices[index].copy(),
        "box_local_lower_bound_m": lower[index].copy(),
        "box_local_upper_bound_m": upper[index].copy(),
        "interior_slack_m": float(slack[index]),
        "numerical_guard_m": float(guard),
        "joint_target_interval_m": [float(joint_lower), float(joint_upper)],
        "measured_joint_m": float(measured_joint),
        "measured_joint_in_target_interval": bool(joint_lower <= measured_joint <= joint_upper),
        "vertices_checked": len(vertices),
        "unproven_means_collision_free": False,
        "scope": "one specified material vertex and fixed box under supplied pose, slide and shape bounds only",
    }
