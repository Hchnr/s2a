"""Minimum conditional translation among fixed-box axis separation choices."""

from itertools import product

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.parallel_box_reference_projection import single_slide_shape_bound
from s2a.research.joint_control.tracking.tabletop_reference_projection import _nonnegative, _strict_pose

FACES = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))


def project_box_faces_reference(
    desired_tcp,
    vertices_tcp,
    slide_axis_tcp,
    current_joint,
    joint_lower,
    joint_upper,
    boxes,
    *,
    position_uncertainty,
    rotation_uncertainty,
    additional_shape_uncertainty,
    clearance,
    numerical_margin,
    max_translation,
):
    """Find the shortest translation that separates each conservative box envelope.

    All fixed boxes must have exactly matching oriented coordinate axes. For
    each box one of six face halfspaces must separate the entire uncertainty
    envelope. Enumerating their combinations gives the minimum translation
    only within this conservative family, not a general minimum collision-free
    motion. No path or unlisted geometry is certified. The supplied unit slide
    must be independently verified on the TCP-relative kinematic chain.
    """
    desired, vertices, axis = _strict_pose(desired_tcp), finite(vertices_tcp), finite(slide_axis_tcp, (3,))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("A nonempty finite set of material vertices is required")
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-12, rtol=0):
        raise ValueError("The TCP-frame slide axis must have unit length")
    slide_bound = single_slide_shape_bound(current_joint, joint_lower, joint_upper)
    position = _nonnegative(position_uncertainty, "position_uncertainty")
    rotation = _nonnegative(rotation_uncertainty, "rotation_uncertainty")
    residual = _nonnegative(additional_shape_uncertainty, "additional_shape_uncertainty")
    clearance = _nonnegative(clearance, "clearance")
    numerical = _nonnegative(numerical_margin, "numerical_margin")
    limit = _nonnegative(max_translation, "max_translation")
    if rotation > np.pi:
        raise ValueError("rotation_uncertainty must be at most pi")
    if not 1 <= len(boxes) <= 4:
        raise ValueError("One to four fixed boxes are required")
    names = [box[0] for box in boxes]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Distinct nonempty box names are required")
    poses = [_strict_pose(box[1]) for box in boxes]
    axes = poses[0][:3, :3]
    if any(not np.array_equal(axes, pose[:3, :3]) for pose in poses[1:]):
        raise ValueError("All fixed box axes must be exactly identical")

    delta_min, delta_max = float(joint_lower - current_joint), float(joint_upper - current_joint)
    axis_error = 2 * np.sin(rotation / 2)
    world = vertices @ desired[:3, :3].T + desired[:3, 3]
    nominal_axis = (desired[:3, :3] @ axis) @ axes
    axis_lower = np.clip(nominal_axis - axis_error, -1.0, 1.0)
    axis_upper = np.clip(nominal_axis + axis_error, -1.0, 1.0)
    possibilities = np.array(
        [axis_lower * delta_min, axis_lower * delta_max, axis_upper * delta_min, axis_upper * delta_max]
    )
    slide_lower, slide_upper = np.min(possibilities, axis=0), np.max(possibilities, axis=0)
    vertex_error = position + axis_error * np.linalg.norm(vertices, axis=1) + residual
    envelopes = []
    for (name, _pose, half_size), pose in zip(boxes, poses, strict=True):
        size = finite(half_size, (3,))
        if np.any(size <= 0):
            raise ValueError("Each fixed box must have strictly positive half-size")
        local = (world - pose[:3, 3]) @ axes
        lower = np.min(local + slide_lower[None, :] - vertex_error[:, None], axis=0)
        upper = np.max(local + slide_upper[None, :] + vertex_error[:, None], axis=0)
        scale = max(1.0, float(np.max(np.abs(world))), float(np.max(np.abs(pose[:3, 3]))), float(np.max(size)))
        guard = max(numerical, 64 * np.finfo(float).eps * scale)
        envelopes.append({"name": name, "lower": lower, "upper": upper, "size": size, "guard": guard})

    best, combinations = None, 0
    for faces in product(FACES, repeat=len(envelopes)):
        combinations += 1
        lower, upper = np.full(3, -np.inf), np.full(3, np.inf)
        for envelope, (dimension, side) in zip(envelopes, faces, strict=True):
            extent = envelope["size"][dimension] + clearance + envelope["guard"]
            if side > 0:
                lower[dimension] = max(lower[dimension], extent - envelope["lower"][dimension])
            else:
                upper[dimension] = min(upper[dimension], -extent - envelope["upper"][dimension])
        if np.any(lower > upper):
            continue
        offset = np.maximum(lower, np.minimum(upper, np.zeros(3)))
        distance = float(np.linalg.norm(offset))
        if not np.isfinite(distance):
            raise ArithmeticError("Finite box separation must yield a finite translation")
        if best is None or distance < best[0]:
            best = (distance, offset, faces)
    if best is None:
        raise ArithmeticError("No consistent face separation combination found")
    distance, offset, faces = best
    accepted = distance <= limit
    world_offset = axes @ offset
    projected = None
    if accepted:
        projected = desired.copy()
        projected[:3, 3] += world_offset
        actual_local_offset = (projected[:3, 3] - desired[:3, 3]) @ axes
        for envelope, (dimension, side) in zip(envelopes, faces, strict=True):
            gap = (
                envelope["lower"][dimension] + actual_local_offset[dimension] - envelope["size"][dimension]
                if side > 0
                else -envelope["upper"][dimension] - actual_local_offset[dimension] - envelope["size"][dimension]
            )
            if gap < clearance:
                raise ArithmeticError("The selected face failed its own conditional clearance bound")
    return {
        "accepted": bool(accepted),
        "status": "correction_exceeds_limit" if not accepted else ("projected" if distance else "unchanged"),
        "projected_tcp": projected,
        "required_translation_m": distance,
        "translation_world_m": world_offset,
        "selected_faces": [
            {"box": envelope["name"], "axis": dimension, "side": side}
            for envelope, (dimension, side) in zip(envelopes, faces, strict=True)
        ],
        "face_combinations_checked": combinations,
        "joint_range_displacement_bound_m": slide_bound,
        "conditional_clearance_certified": bool(accepted),
        "scope": "specified vertices, one verified slide and identically oriented fixed boxes under supplied bounds only",
    }
