"""Independent geometry and per-sample bookkeeping for tabletop feedback probes."""

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform


def measured_vertices_in_tcp(vertices_geom, tcp_pose, geom_pose):
    """Reconstruct material vertices from current rigid kinematic transforms."""
    vertices = finite(vertices_geom)
    tcp, geom = (validate_transform(np.asarray(value, dtype=float)) for value in (tcp_pose, geom_pose))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or tcp.shape != (4, 4) or geom.shape != (4, 4):
        raise ValueError("One nonempty measured mesh and two current poses are required")
    world = vertices @ geom[:3, :3].T + geom[:3, 3]
    return (world - tcp[:3, 3]) @ tcp[:3, :3]


def observed_bound_conditions(
    reference_vertices_tcp,
    current_vertices_tcp,
    selected_tcp,
    current_tcp,
    *,
    position_bound,
    rotation_bound,
    shape_bound,
):
    """Score one measured sample; do not infer unobserved continuous behavior."""
    reference, current = finite(reference_vertices_tcp), finite(current_vertices_tcp)
    selected, measured = (validate_transform(np.asarray(value, dtype=float)) for value in (selected_tcp, current_tcp))
    bounds = np.asarray([position_bound, rotation_bound, shape_bound], dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or not len(reference)
        or current.shape != reference.shape
        or selected.shape != (4, 4)
        or measured.shape != (4, 4)
        or not np.isfinite(bounds).all()
        or np.any(bounds < 0)
    ):
        raise ValueError("Corresponding finite vertices, poses and nonnegative sample bounds are required")
    error = pose_error(selected, measured)
    displacement = float(np.max(np.linalg.norm(current - reference, axis=1)))
    position_error, rotation_error = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
    checks = {
        "position_bound_met": bool(position_error <= bounds[0]),
        "rotation_bound_met": bool(rotation_error <= bounds[1]),
        "shape_bound_met": bool(displacement <= bounds[2]),
    }
    return {
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "max_material_vertex_displacement_tcp_m": displacement,
        **checks,
        "all_sample_bounds_met": all(checks.values()),
        "scope": "this measured sample only; no continuous path or other collision bodies",
    }
