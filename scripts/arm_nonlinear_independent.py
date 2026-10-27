"""100 independent scalar Fourier columns atop independent83-condition history."""

import numpy as np
from arm_history_independent import INPUTS, augment, conditions, nominal_prediction

__all__ = ["INPUTS", "augment", "conditions", "nominal_prediction", "feature_matrix", "prediction"]


def feature_matrix(data, parameters):
    z = (conditions(data) - np.asarray(parameters["state_mean"])) / np.asarray(parameters["state_scale"])
    w = np.asarray(parameters["frequencies"])
    assert w.shape == (64, 83) and np.isfinite(w).all()
    columns = [np.ones(len(z)), *[z[:, i] for i in range(83)]]
    columns += [np.cos(np.sum(z * w[i], axis=1)) / np.sqrt(64) for i in range(64)]
    columns += [np.sin(np.sum(z * w[i], axis=1)) / np.sqrt(64) for i in range(64)]
    phi = np.column_stack(columns)
    a = (data["action_goal_delta"][:, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
        parameters["translation_scale"]
    )
    return np.column_stack([*columns, *[phi[:, i] * a[:, j] for j in range(3) for i in range(212)]])


def prediction(data, parameters):
    return nominal_prediction(data, parameters) + feature_matrix(data, parameters) @ np.asarray(
        parameters["coefficients"]
    )
