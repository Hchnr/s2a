"""Opt-in tabletop reference projection for a measured rigid collision geometry."""

from numbers import Real

import numpy as np

from s2a.geometry import finite, inverse, validate_transform


def _strict_pose(value):
    pose = validate_transform(np.asarray(value, dtype=float))
    if pose.shape != (4, 4):
        raise ValueError("One rigid transform is required")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0
    ):
        raise ValueError("A numerically orthonormal proper rotation is required")
    return pose


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(value)


def collision_vertices_in_tcp(vertices_geom, measured_tcp, measured_geom):
    """Express current collision vertices in the measured TCP frame.

    The caller supplies the current measured robot/gripper configuration and
    fixed robot geometry; no future state or task label is consumed.
    """
    vertices = finite(vertices_geom)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("A nonempty finite set of collision vertices is required")
    tcp_from_geom = inverse(_strict_pose(measured_tcp)) @ _strict_pose(measured_geom)
    return vertices @ tcp_from_geom[:3, :3].T + tcp_from_geom[:3, 3]


def project_tabletop_reference(
    desired_tcp,
    vertices_tcp,
    table_pose,
    table_half_size,
    *,
    position_uncertainty,
    rotation_uncertainty,
    shape_uncertainty,
    clearance,
    numerical_margin,
    max_translation,
):
    """Project one reference above a calibrated table, subject to an explicit cap.

    The conditional envelope assumes each actual material vertex moves by at
    most shape_uncertainty in the TCP frame and the actual TCP remains within
    the supplied translation and SO(3) bounds of the returned target. It covers
    only these vertices and this fixed table box, not path, other geometry,
    control tracking, perception error, or physical safety in general.
    """
    desired, table = _strict_pose(desired_tcp), _strict_pose(table_pose)
    vertices, size = finite(vertices_tcp), finite(table_half_size, (3,))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or np.any(size <= 0):
        raise ValueError("Finite collision vertices and a positive table box are required")
    position = _nonnegative(position_uncertainty, "position_uncertainty")
    rotation = _nonnegative(rotation_uncertainty, "rotation_uncertainty")
    shape = _nonnegative(shape_uncertainty, "shape_uncertainty")
    clearance = _nonnegative(clearance, "clearance")
    numerical = _nonnegative(numerical_margin, "numerical_margin")
    limit = _nonnegative(max_translation, "max_translation")
    if rotation > np.pi:
        raise ValueError("rotation_uncertainty must be at most pi")

    world = vertices @ desired[:3, :3].T + desired[:3, 3]
    local = (world - table[:3, 3]) @ table[:3, :3]
    displacement = position + 2 * np.linalg.norm(vertices, axis=1) * np.sin(rotation / 2) + shape
    lower_xy = np.min(local[:, :2] - displacement[:, None], axis=0)
    upper_xy = np.max(local[:, :2] + displacement[:, None], axis=0)
    scale = max(1.0, float(np.max(np.abs(world))), float(np.max(np.abs(table[:3, 3]))), float(np.max(size)))
    guard = max(numerical, 64 * np.finfo(float).eps * scale)
    expanded_half_size = size[:2] + clearance + guard
    overlaps = bool(np.all(lower_xy <= expanded_half_size) and np.all(upper_xy >= -expanded_half_size))
    original_lower = float(np.min(local[:, 2] - displacement) - size[2])
    required = max(0.0, clearance + guard - original_lower) if overlaps else 0.0
    accepted = required <= limit
    projected = None
    projected_lower = None
    if accepted:
        projected = desired.copy()
        projected[:3, 3] += required * table[:3, 2]
        projected_world = vertices @ projected[:3, :3].T + projected[:3, 3]
        projected_local = (projected_world - table[:3, 3]) @ table[:3, :3]
        projected_lower = float(np.min(projected_local[:, 2] - displacement) - size[2])
        if overlaps and projected_lower < clearance:
            raise ArithmeticError("The projected reference failed its own conditional clearance bound")
    return {
        "accepted": bool(accepted),
        "status": "correction_exceeds_limit" if not accepted else ("projected" if required > 0 else "unchanged"),
        "projected_tcp": projected,
        "required_translation_m": float(required),
        "reference_rotation_change_rad": 0.0,
        "uncertainty_expanded_xy_overlap": overlaps,
        "original_height_lower_bound_m": original_lower,
        "projected_height_lower_bound_m": projected_lower,
        "maximum_vertex_displacement_bound_m": float(np.max(displacement)),
        "numerical_guard_m": guard,
        "conditional_table_clearance_certified": bool(accepted),
        "shape_uncertainty_m": shape,
        "scope": "specified collision vertices and fixed table box under supplied bounds only",
    }
