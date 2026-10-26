"""216 translation commands with the fixed 20Hz executed-command rate limit."""

import numpy as np

from s2a.late_clipped_arm_inverse import command_prediction


def translation_command(data, desired_site, *, gripper_command, parameters):
    """Use measured prior commands; callers commit the returned actual command.

    This function does not advance history or predict a future measured endpoint.
    Rotation and gripper policies remain the caller's explicitly fixed policies.
    """
    count = len(data["x_site_pose"])
    previous = np.asarray(data["x_previous_command"], dtype=float)
    available = np.asarray(data["x_previous_command_available"])
    lo, hi, outlo, outhi = (
        np.asarray(parameters["translation_" + k], dtype=float)
        for k in ("input_min", "input_max", "output_min", "output_max")
    )
    if (
        previous.shape != (count, 7)
        or available.shape != (count,)
        or not np.isfinite(previous).all()
        or not np.isin(available, [False, True]).all()
        or np.any(previous[:, :3] < lo)
        or np.any(previous[:, :3] > hi)
        or np.any(previous[~available.astype(bool)] != 0)
    ):
        raise ValueError("Actual feasible previous commands or explicit zero episode reset required")
    if parameters["dt"] != 0.05:
        raise ValueError("Fixed20Hz command interval required")
    prediction = command_prediction(data, desired_site, gripper_command=gripper_command, parameters=parameters)
    lower = np.maximum(lo, previous[:, :3] - 0.25)
    upper = np.minimum(hi, previous[:, :3] + 0.25)
    command = np.minimum(upper, np.maximum(lower, prediction["command"]))
    physical = outlo + (command - lo) * (outhi - outlo) / (hi - lo)
    return {
        **prediction,
        "unconstrained_rate_command": prediction["command"].copy(),
        "command": command,
        "physical_delta": physical,
        "rate_limited": np.any(command != prediction["command"], axis=1),
        "lower": lower,
        "upper": upper,
    }
