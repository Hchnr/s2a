"""112 explicit per-trajectory causal measured-wrench changes."""

import numpy as np

from s2a.joint_wrench_condition_forward import design as previous_design
from s2a.joint_wrench_condition_forward import nominal, raw_condition, targets, with_history

__all__ = ["attach_wrench_change", "design", "predict", "nominal", "raw_condition", "targets", "with_history"]


def attach_wrench_change(data):
    wrench = np.asarray(data["x_current_wrench_world"], dtype=float)
    if wrench.ndim != 2 or wrench.shape[1] != 6 or not np.isfinite(wrench).all() or len(wrench) == 0:
        raise ValueError("Finite nonempty Nx6 current wrench required")
    available = np.asarray(data["x_previous_command_available"], dtype=bool)
    if available.shape != (len(wrench),) or available[0] or not available[1:].all():
        raise ValueError("Complete single-trajectory prefix required before wrench differencing")
    output = dict(data)
    output["x_wrench_change_world"] = np.vstack((np.zeros((1, 6)), wrench[1:] - wrench[:-1]))
    return output


def design(data, parameters):
    previous = previous_design(data, parameters)
    if "x_wrench_change_world" not in data:
        raise ValueError("Explicit causal wrench change required; construct full prefix before slicing")
    change = np.asarray(data["x_wrench_change_world"], dtype=float)
    if change.shape != (len(previous), 6) or not np.isfinite(change).all():
        raise ValueError("Finite Nx6 causal wrench change required")
    phi = np.c_[previous[:, :282], change / np.asarray(parameters["state_scale"])[83:89]]
    action = (np.asarray(data["action_goal_delta"])[:, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
        parameters["translation_scale"]
    )
    return np.c_[phi, *(phi * action[:, i, None] for i in range(3))]


def predict(data, parameters):
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (1152, 12) or not np.isfinite(coefficients).all():
        raise ValueError("Finite1152x12 coefficients required")
    result = nominal(data, parameters) + design(data, parameters) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite causal wrench-change prediction")
    return result
