"""Conditional one-direction projection against parallel fixed box surfaces."""

from numbers import Real

import numpy as np

from s2a.research.joint_control.tracking.tabletop_reference_projection import project_tabletop_reference


def single_slide_shape_bound(position, lower, upper):
    """Bound one material vertex's translation while a unit slide stays in range.

    The caller must independently verify the kinematic chain contains exactly
    this slide between the measured TCP and the geometry. Joint limits are an
    explicit condition; this function does not enforce them in a simulator or
    robot and does not establish a velocity or continuous-time bound.
    """
    values = (position, lower, upper)
    if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) for value in values):
        raise ValueError("The current slide coordinate and interval must be finite scalars")
    if lower > upper or not lower <= position <= upper:
        raise ValueError("The current slide coordinate must lie in the ordered interval")
    return float(max(position - lower, upper - position))


def project_parallel_boxes_reference(
    desired_tcp,
    vertices_tcp,
    boxes,
    *,
    position_uncertainty,
    rotation_uncertainty,
    shape_uncertainty,
    clearance,
    numerical_margin,
    max_translation,
):
    """Project above fixed boxes sharing exactly one oriented face normal.

    Each box is ``(name, pose, half_size)``. The maximum of each box's
    individually sufficient lift also suffices for all boxes because movement
    along their common normal leaves each horizontal footprint unchanged. It
    is minimal only within this particular conservative envelope and direction.
    The same conditional bounds and limitations as the single-box projection
    apply; no continuous path or unlisted collision geometry is certified.
    """
    if not boxes:
        raise ValueError("At least one fixed box is required")
    names = [box[0] for box in boxes]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Distinct nonempty box names are required")
    poses = [np.asarray(box[1], dtype=float) for box in boxes]
    if any(pose.shape != (4, 4) for pose in poses):
        raise ValueError("Each fixed box requires one rigid pose")
    normal = poses[0][:3, 2].copy()
    if any(not np.array_equal(normal, pose[:3, 2]) for pose in poses[1:]):
        raise ValueError("Box face normals must be exactly equal and oriented in the same direction")
    options = dict(
        position_uncertainty=position_uncertainty,
        rotation_uncertainty=rotation_uncertainty,
        shape_uncertainty=shape_uncertainty,
        clearance=clearance,
        numerical_margin=numerical_margin,
        max_translation=max_translation,
    )
    results = {
        name: project_tabletop_reference(desired_tcp, vertices_tcp, pose, size, **options)
        for (name, _pose, size), pose in zip(boxes, poses, strict=True)
    }
    required = max(result["required_translation_m"] for result in results.values())
    accepted = all(result["accepted"] for result in results.values())
    projected = None
    if accepted:
        projected = np.asarray(desired_tcp, dtype=float).copy()
        projected[:3, 3] += required * normal
    return {
        "accepted": accepted,
        "status": "correction_exceeds_limit" if not accepted else ("projected" if required else "unchanged"),
        "projected_tcp": projected,
        "required_translation_m": float(required),
        "boxes": results,
        "conditional_clearance_certified": accepted,
        "shape_uncertainty_m": float(shape_uncertainty),
        "scope": "specified vertices and parallel fixed boxes under supplied bounds only",
    }
