"""Conditional fixed-box projection for one TCP-relative prismatic joint."""

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.parallel_box_reference_projection import single_slide_shape_bound
from s2a.research.joint_control.tracking.tabletop_reference_projection import _nonnegative, _strict_pose


def project_directional_slide_boxes_reference(
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
    """Bound one slide's direction and TCP pose around the returned reference.

    Every material vertex at another allowed joint value is assumed to equal
    its measured TCP-frame vertex plus the unit slide axis times the joint
    displacement, with an additional independently bounded residual. The
    caller must verify this kinematic relation and enforce or observe the
    stated joint, TCP and fixed-box conditions. The clearance result covers
    only the supplied vertices and boxes at a pose, not the continuous path.
    """
    desired = _strict_pose(desired_tcp)
    vertices, axis = finite(vertices_tcp), finite(slide_axis_tcp, (3,))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("A nonempty finite set of material vertices is required")
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-12, rtol=0):
        raise ValueError("The TCP-frame slide axis must have unit length")
    range_bound = single_slide_shape_bound(current_joint, joint_lower, joint_upper)
    position = _nonnegative(position_uncertainty, "position_uncertainty")
    rotation = _nonnegative(rotation_uncertainty, "rotation_uncertainty")
    residual = _nonnegative(additional_shape_uncertainty, "additional_shape_uncertainty")
    clearance = _nonnegative(clearance, "clearance")
    numerical = _nonnegative(numerical_margin, "numerical_margin")
    limit = _nonnegative(max_translation, "max_translation")
    if rotation > np.pi:
        raise ValueError("rotation_uncertainty must be at most pi")
    if not boxes:
        raise ValueError("At least one fixed box is required")
    names = [box[0] for box in boxes]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Distinct nonempty box names are required")
    poses = [_strict_pose(box[1]) for box in boxes]
    normal = poses[0][:3, 2].copy()
    if any(not np.array_equal(normal, pose[:3, 2]) for pose in poses[1:]):
        raise ValueError("Box face normals must be exactly equal and oriented in the same direction")

    delta_min, delta_max = float(joint_lower - current_joint), float(joint_upper - current_joint)
    axis_error = 2 * np.sin(rotation / 2)
    world = vertices @ desired[:3, :3].T + desired[:3, 3]
    axis_world = desired[:3, :3] @ axis
    vertex_error = position + axis_error * np.linalg.norm(vertices, axis=1) + residual
    results = {}
    for (name, _pose, half_size), pose in zip(boxes, poses, strict=True):
        size = finite(half_size, (3,))
        if np.any(size <= 0):
            raise ValueError("Each fixed box must have strictly positive half-size")
        local = (world - pose[:3, 3]) @ pose[:3, :3]
        nominal_axis = axis_world @ pose[:3, :3]
        axis_lower = np.clip(nominal_axis - axis_error, -1.0, 1.0)
        axis_upper = np.clip(nominal_axis + axis_error, -1.0, 1.0)
        possibilities = np.array(
            [axis_lower * delta_min, axis_lower * delta_max, axis_upper * delta_min, axis_upper * delta_max]
        )
        slide_lower, slide_upper = np.min(possibilities, axis=0), np.max(possibilities, axis=0)
        lower = np.min(local + slide_lower[None, :] - vertex_error[:, None], axis=0)
        upper = np.max(local + slide_upper[None, :] + vertex_error[:, None], axis=0)
        scale = max(1.0, float(np.max(np.abs(world))), float(np.max(np.abs(pose[:3, 3]))), float(np.max(size)))
        guard = max(numerical, 64 * np.finfo(float).eps * scale)
        expanded = size[:2] + clearance + guard
        overlaps = bool(np.all(lower[:2] <= expanded) and np.all(upper[:2] >= -expanded))
        original_height = float(lower[2] - size[2])
        required = max(0.0, clearance + guard - original_height) if overlaps else 0.0
        results[name] = {
            "uncertainty_expanded_xy_overlap": overlaps,
            "box_local_slide_axis_nominal": nominal_axis,
            "box_local_slide_axis_lower": axis_lower,
            "box_local_slide_axis_upper": axis_upper,
            "original_height_lower_bound_m": original_height,
            "required_translation_m": float(required),
            "numerical_guard_m": guard,
        }
    required = max(result["required_translation_m"] for result in results.values())
    accepted = required <= limit
    projected = None
    if accepted:
        projected = desired.copy()
        projected[:3, 3] += required * normal
    return {
        "accepted": bool(accepted),
        "status": "correction_exceeds_limit" if not accepted else ("projected" if required else "unchanged"),
        "projected_tcp": projected,
        "required_translation_m": float(required),
        "boxes": results,
        "slide_delta_interval_m": [delta_min, delta_max],
        "joint_range_displacement_bound_m": range_bound,
        "additional_shape_uncertainty_m": residual,
        "conditional_clearance_certified": bool(accepted),
        "scope": "specified vertices, one verified slide and parallel fixed boxes under supplied bounds only",
    }
