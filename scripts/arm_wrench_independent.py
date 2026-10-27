"""104 independent feature assembly preserving the old Fourier input contract."""

import numpy as np
from arm_nonlinear_independent import INPUTS as ORIGINAL_INPUTS
from arm_nonlinear_independent import augment, nominal_prediction
from arm_nonlinear_independent import conditions as original_conditions
from arm_nonlinear_independent import feature_matrix as original_features

INPUTS = (*ORIGINAL_INPUTS, "x_current_wrench_world")
__all__ = ["INPUTS", "augment", "nominal_prediction", "conditions", "feature_matrix", "prediction"]


def conditions(data):
    return np.column_stack([original_conditions(data), np.asarray(data["x_current_wrench_world"])])


def feature_matrix(data, parameters):
    mean, scale = np.asarray(parameters["state_mean"]), np.asarray(parameters["state_scale"])
    assert mean.shape == scale.shape == (89,)
    old = original_features(data, dict(parameters, state_mean=mean[:83], state_scale=scale[:83]))
    normalized = (np.asarray(data["x_current_wrench_world"]) - mean[83:]) / scale[83:]
    phi = np.column_stack([*[old[:, i] for i in range(212)], *[normalized[:, i] for i in range(6)]])
    action = (np.asarray(data["action_goal_delta"])[:, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
        parameters["translation_scale"]
    )
    return np.column_stack(
        [*[phi[:, i] for i in range(218)], *[phi[:, i] * action[:, a] for a in range(3) for i in range(218)]]
    )


def prediction(data, parameters):
    return nominal_prediction(data, parameters) + feature_matrix(data, parameters) @ np.asarray(
        parameters["coefficients"]
    )
