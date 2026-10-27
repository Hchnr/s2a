"""109 preserve107 basis and append fixed joint state/wrench Fourier features."""

import numpy as np

from s2a.wrench_conditioned_arm_forward import design as previous_design
from s2a.wrench_conditioned_arm_forward import nominal, raw_condition, targets, with_history

__all__ = ["design", "predict", "nominal", "raw_condition", "targets", "with_history"]


def design(data, parameters):
    prior = previous_design(data, parameters)
    z = (raw_condition(data) - np.asarray(parameters["state_mean"])) / np.asarray(parameters["state_scale"])
    frequencies = np.asarray(parameters["joint_frequencies"], dtype=float)
    if frequencies.shape != (32, 89) or not np.isfinite(frequencies).all():
        raise ValueError("Fixed finite32x89 joint frequencies required")
    phase = z @ frequencies.T
    phi = np.c_[prior[:, :218], np.cos(phase) / np.sqrt(32), np.sin(phase) / np.sqrt(32)]
    action = (np.asarray(data["action_goal_delta"])[:, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
        parameters["translation_scale"]
    )
    result = np.c_[phi, *(phi * action[:, i, None] for i in range(3))]
    assert result.shape == (len(z), 1128)
    return result


def predict(data, parameters):
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (1128, 12) or not np.isfinite(coefficients).all():
        raise ValueError("Finite1128x12 coefficients required")
    result = nominal(data, parameters) + design(data, parameters) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite joint-wrench prediction")
    return result
