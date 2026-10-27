"""100 fixed nonlinear causal conditions with affine translation action."""

import numpy as np

from s2a.history_conditioned_arm_forward import design as linear_design
from s2a.history_conditioned_arm_forward import nominal, raw_condition, targets, with_history

__all__ = ["design", "predict", "nominal", "raw_condition", "targets", "with_history"]


def design(data, parameters):
    linear = linear_design(data, parameters)
    z = linear[:, 1:84]
    frequencies = np.asarray(parameters["frequencies"], dtype=float)
    if frequencies.shape != (64, 83) or not np.isfinite(frequencies).all():
        raise ValueError("Fixed finite64x83 frequencies required")
    phase = z @ frequencies.T
    phi = np.c_[linear[:, :84], np.cos(phase) / 8, np.sin(phase) / 8]
    action = (np.asarray(data["action_goal_delta"])[:, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
        parameters["translation_scale"]
    )
    value = np.c_[phi, *(phi * action[:, i, None] for i in range(3))]
    assert value.shape == (len(z), 848)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite nonlinear design")
    return value


def predict(data, parameters):
    coef = np.asarray(parameters["coefficients"], dtype=float)
    if coef.shape != (848, 12) or not np.isfinite(coef).all():
        raise ValueError("Finite848x12 coefficients required")
    value = nominal(data, parameters) + design(data, parameters) @ coef
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite prediction")
    return value
