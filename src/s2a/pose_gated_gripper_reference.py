"""Experimental causal opening schedule from already generated TCP references."""

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform


def align_closure(
    current_tcp,
    generated_tcp,
    openings,
    *,
    execute_steps,
    closing_threshold,
    position_tolerance_m,
    rotation_tolerance_rad,
):
    """Advance only a deferred first closure whose pose is already reached.

    This changes desired gripper timing, never arm references. No object state,
    success labels, future observations or subsequent generated chunks are read.
    The caller owns motor mapping and must separately qualify physical behavior.
    """
    current = validate_transform(current_tcp)
    poses = validate_transform(generated_tcp)
    values = finite(openings).copy()
    if poses.ndim != 3 or values.shape != (len(poses),) or not len(poses):
        raise ValueError("One opening per generated TCP target required")
    if not 1 <= execute_steps <= len(poses) or not 0 < closing_threshold < 1:
        raise ValueError("Explicit supported execution window and closure threshold required")
    if (
        not np.isfinite([position_tolerance_m, rotation_tolerance_rad]).all()
        or min(position_tolerance_m, rotation_tolerance_rad) <= 0
    ):
        raise ValueError("Positive finite geometric tolerances required")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Normalized opening references required")
    closing = np.flatnonzero(values < closing_threshold)
    selected = None
    error = None
    if len(closing) and closing[0] >= execute_steps:
        index = int(closing[0])
        residual = pose_error(poses[index], current)
        error = dict(position_m=float(np.linalg.norm(residual[:3])), rotation_rad=float(np.linalg.norm(residual[3:])))
        if error["position_m"] <= position_tolerance_m and error["rotation_rad"] <= rotation_tolerance_rad:
            selected = index
            values[:execute_steps] = np.minimum(values[:execute_steps], values[index])
    return dict(openings=values, selected_index=selected, first_deferred_closure_pose_error=error)
