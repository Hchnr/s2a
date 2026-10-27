"""090 append causal observed response innovation; translation action stays affine."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.state_conditioned_arm_forward import nominal, targets
from s2a.state_conditioned_arm_forward import raw_condition as base_condition

__all__ = ["with_history", "raw_condition", "design", "predict", "nominal", "targets"]


def with_history(data, parameters):
    pose = np.asarray(data["x_site_pose"], dtype=float)
    twist = np.asarray(data["x_site_twist"], dtype=float)
    available = np.asarray(data["x_previous_command_available"])
    n = len(pose)
    if pose.shape != (n, 4, 4) or twist.shape != (n, 6) or available.shape != (n,) or not n:
        raise ValueError("Current/past episode endpoint arrays required")
    if not np.isin(available, [0, 1]).all() or bool(available[0]):
        raise ValueError("Build history from an explicit episode prefix, not an isolated anchor")
    if not np.isfinite(pose).all() or not np.isfinite(twist).all():
        raise ValueError("Finite observed endpoints required")
    previous_prediction = nominal(data, parameters)
    value = np.zeros((n, 12))
    for i in range(1, n):
        if not available[i]:
            continue
        prediction = previous_prediction[i - 1]
        value[i, :3] = (pose[i, :3, 3] - pose[i - 1, :3, 3]) - prediction[:3]
        relative = pose[i, :3, :3] @ pose[i - 1, :3, :3].T
        value[i, 3:6] = Rotation.from_matrix(relative @ Rotation.from_rotvec(prediction[3:6]).as_matrix().T).as_rotvec()
        value[i, 6:] = 0.05 * twist[i] - prediction[6:]
    result = dict(data)
    result["x_response_residual"] = value
    return result


def raw_condition(data):
    base = base_condition(data)
    history = np.asarray(data["x_response_residual"], dtype=float)
    if history.shape != (len(base), 12) or not np.isfinite(history).all():
        raise ValueError("Explicit causal12 response history required")
    return np.concatenate([base, history], axis=1)


def design(data, parameters):
    raw = raw_condition(data)
    mean = np.asarray(parameters["state_mean"])
    scale = np.asarray(parameters["state_scale"])
    amean = np.asarray(parameters["translation_mean"])
    ascale = np.asarray(parameters["translation_scale"])
    if mean.shape != (83,) or scale.shape != (83,) or amean.shape != (3,) or ascale.shape != (3,):
        raise ValueError("Fixed83/3 statistics required")
    if not all(np.isfinite(v).all() for v in (mean, scale, amean, ascale)) or np.any(scale <= 0) or np.any(ascale <= 0):
        raise ValueError("Finite positive fit scales required")
    phi = np.c_[np.ones(len(raw)), (raw - mean) / scale]
    action = (np.asarray(data["action_goal_delta"])[:, :3] - amean) / ascale
    result = np.c_[phi, *(phi * action[:, i, None] for i in range(3))]
    assert result.shape == (len(raw), 336)
    return result


def predict(data, parameters):
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (336, 12) or not np.isfinite(coefficients).all():
        raise ValueError("Finite336x12 coefficients required")
    value = nominal(data, parameters) + design(data, parameters) @ coefficients
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite forward prediction")
    return value
