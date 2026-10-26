"""220 current measured SITE state and already generated references for219 planning."""

import numpy as np

from s2a.geometry import validate_transform
from s2a.rate_constrained_preview_active_set import plan_translation
from s2a.reference_linear_velocity import reference_linear_velocity


def plan_current_translation(data, generated_site_poses, index, anchor_site, contract):
    generated = validate_transform(np.asarray(generated_site_poses, dtype=float))
    anchor = validate_transform(np.asarray(anchor_site, dtype=float))
    if generated.ndim != 3 or anchor.shape != (4, 4) or type(index) is not int or not 0 <= index < len(generated):
        raise ValueError("Available generated SITE trajectory, current integer cursor and query anchor required")
    pose = validate_transform(np.asarray(data["x_site_pose"], dtype=float))
    velocity = np.asarray(data["x_site_twist"], dtype=float)
    previous = np.asarray(data["x_previous_command"], dtype=float)
    available = np.asarray(data["x_previous_command_available"])
    completed = np.asarray(data["x_response_residual"], dtype=float)
    if (
        pose.shape != (1, 4, 4)
        or velocity.shape != (1, 6)
        or previous.shape != (1, 7)
        or completed.shape != (1, 12)
        or available.shape != (1,)
        or not np.isin(available, [False, True]).all()
        or not all(np.isfinite(v).all() for v in (velocity, previous, completed))
    ):
        raise ValueError("One finite actual endpoint, previous command and completed innovation required")
    if not bool(available[0]) and (np.any(previous != 0) or np.any(completed != 0)):
        raise ValueError("Episode reset requires zero previous command and completed innovation")
    horizon = min(6, len(generated) - index)
    references = generated[index : index + horizon, :3, 3].copy()
    velocities = np.array(
        [reference_linear_velocity(generated, index + j, anchor, 0.05)["velocity"] for j in range(horizon)]
    )
    innovation = np.vstack([completed[0, :3], completed[0, 6:9] / 0.05])
    result = plan_translation(
        pose[0, :3, 3], velocity[0, :3], references, velocities, previous[0, :3], contract, innovation=innovation
    )
    lo, hi, outlo, outhi = (
        np.asarray(contract[key], dtype=float)[:3] for key in ("input_min", "input_max", "output_min", "output_max")
    )
    result["physical_delta"] = outlo + (result["command"] - lo) * (outhi - outlo) / (hi - lo)
    result["reference_positions"] = references
    result["reference_velocities"] = velocities
    result["generated_indices"] = list(range(index, index + horizon))
    result["completed_innovation"] = innovation
    return result
