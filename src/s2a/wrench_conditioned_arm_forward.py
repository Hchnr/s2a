"""104 append explicit current measured world wrench, preserving100 features."""

import numpy as np

from s2a.nonlinear_conditioned_arm_forward import design as original_design
from s2a.nonlinear_conditioned_arm_forward import nominal, targets, with_history
from s2a.nonlinear_conditioned_arm_forward import raw_condition as original_condition

__all__ = ["raw_condition", "design", "predict", "nominal", "targets", "with_history"]


def current_wrench(data, count):
    if "x_current_wrench_world" not in data:
        raise ValueError("Current measured world-at-control-SITE wrench required")
    value = np.asarray(data["x_current_wrench_world"], dtype=float)
    if value.shape != (count, 6) or not np.isfinite(value).all():
        raise ValueError("Finite current Nx6 measured wrench required")
    return value


def raw_condition(data):
    original = original_condition(data)
    return np.c_[original, current_wrench(data, len(original))]


def design(data, parameters):
    mean, scale = (np.asarray(parameters[k], dtype=float) for k in ("state_mean", "state_scale"))
    if (
        mean.shape != (89,)
        or scale.shape != (89,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Fixed finite89 fit statistics and positive scales required")
    base = dict(parameters, state_mean=mean[:83], state_scale=scale[:83])
    previous = original_design(data, base)
    wrench = current_wrench(data, len(previous))
    phi = np.c_[previous[:, :212], (wrench - mean[83:]) / scale[83:]]
    action = (np.asarray(data["action_goal_delta"])[:, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
        parameters["translation_scale"]
    )
    value = np.c_[phi, *(phi * action[:, i, None] for i in range(3))]
    assert value.shape == (len(previous), 872)
    return value


def predict(data, parameters):
    coef = np.asarray(parameters["coefficients"], dtype=float)
    if coef.shape != (872, 12) or not np.isfinite(coef).all():
        raise ValueError("Finite872x12 coefficients required")
    value = nominal(data, parameters) + design(data, parameters) @ coef
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite wrench-conditioned prediction")
    return value
