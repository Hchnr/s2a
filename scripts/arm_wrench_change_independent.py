"""112 independent feature columns; explicit full-prefix wrench-change input."""

import numpy as np
from arm_joint_wrench_independent import INPUTS as PREVIOUS_INPUTS
from arm_joint_wrench_independent import augment, conditions, nominal_prediction
from arm_joint_wrench_independent import feature_matrix as previous_features

INPUTS = (*PREVIOUS_INPUTS, "x_wrench_change_world")
__all__ = ["INPUTS", "augment", "conditions", "nominal_prediction", "feature_matrix", "prediction"]


def feature_matrix(data, parameters):
    old = previous_features(data, parameters)
    change = np.asarray(data["x_wrench_change_world"])
    assert change.shape == (len(old), 6) and np.isfinite(change).all()
    phi = np.column_stack([old[:, :282], change / np.asarray(parameters["state_scale"])[83:89]])
    action = (data["action_goal_delta"][:, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
        parameters["translation_scale"]
    )
    return np.column_stack(
        [*[phi[:, k] for k in range(288)], *[phi[:, k] * action[:, a] for a in range(3) for k in range(288)]]
    )


def prediction(data, parameters):
    return nominal_prediction(data, parameters) + feature_matrix(data, parameters) @ np.asarray(
        parameters["coefficients"]
    )
