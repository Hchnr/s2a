"""Conservative fixed-shape collision certificate over an SE(3) tolerance ball."""

import numpy as np

from s2a.geometry import finite, validate_transform


def fixed_vertex_box_certificate(
    vertices_world, tcp_pose, box_pose, half_size, *, position_tolerance, rotation_tolerance, margin
):
    """Certify that one rigid material point remains inside a fixed oriented box.

    Vertices are material points of the fixed-shape collision geometry at the
    desired TCP pose. The box is fixed. The allowed measured TCP has Euclidean
    position error <= position_tolerance and SO(3) geodesic error <=
    rotation_tolerance. A negative result means only that this sufficient
    certificate did not prove collision; it is not a free-space witness.
    """
    vertices = finite(vertices_world)
    tcp, box = (
        validate_transform(np.asarray(tcp_pose, dtype=float)),
        validate_transform(np.asarray(box_pose, dtype=float)),
    )
    size = finite(half_size, (3,))
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or tcp.shape != (4, 4)
        or box.shape != (4, 4)
        or np.any(size <= 0)
        or not np.isfinite(position_tolerance)
        or position_tolerance < 0
        or not np.isfinite(rotation_tolerance)
        or not 0 <= rotation_tolerance <= np.pi
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise ValueError("Finite rigid vertices, one positive box, and valid nonnegative pose tolerances are required")
    for rotation in (tcp[:3, :3], box[:3, :3]):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0
        ):
            raise ValueError("Certificate transforms must use numerically orthonormal proper rotations")
    local = (vertices - box[:3, 3]) @ box[:3, :3]
    face_margins = size[None, :] - np.abs(local)
    inside_margin = np.min(face_margins, axis=1)
    radius = np.linalg.norm(vertices - tcp[:3, 3], axis=1)
    motion_bound = position_tolerance + 2 * radius * np.sin(rotation_tolerance / 2)
    slack = inside_margin - motion_bound
    index = int(np.argmax(slack))
    scale = max(
        1.0,
        float(np.max(np.abs(vertices))),
        float(np.max(np.abs(tcp[:3, 3]))),
        float(np.max(np.abs(box[:3, 3]))),
        float(np.max(size)),
        float(np.max(motion_bound)),
    )
    numeric_guard = max(float(margin), 64 * np.finfo(float).eps * scale)
    return {
        "certified": bool(slack[index] > numeric_guard),
        "vertex_index": index,
        "vertex_world": vertices[index].copy(),
        "vertex_table_local": local[index].copy(),
        "face_margins_m": face_margins[index].copy(),
        "inside_margin_m": float(inside_margin[index]),
        "tcp_radius_m": float(radius[index]),
        "motion_bound_m": float(motion_bound[index]),
        "certificate_slack_m": float(slack[index]),
        "required_margin_m": numeric_guard,
        "vertices_checked": len(vertices),
        "fixed_gripper_configuration": True,
        "unproven_means_collision_free": False,
    }
