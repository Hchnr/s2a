"""109 independent scalar joint phases and block assembly."""

import math

import numpy as np
from arm_wrench_independent import INPUTS, augment, conditions, nominal_prediction
from arm_wrench_independent import feature_matrix as previous_features

__all__ = ["INPUTS", "augment", "conditions", "nominal_prediction", "feature_matrix", "prediction"]


def feature_matrix(data, parameters):
    old = previous_features(data, parameters)
    raw = conditions(data)
    mean, scale = np.asarray(parameters["state_mean"]), np.asarray(parameters["state_scale"])
    frequencies = np.asarray(parameters["joint_frequencies"])
    assert frequencies.shape == (32, 89)
    rows = []
    for index, value in enumerate(raw):
        z = (value - mean) / scale
        phases = [sum(float(w[k]) * float(z[k]) for k in range(89)) for w in frequencies]
        phi = list(old[index, :218])
        phi.extend(math.cos(v) / math.sqrt(32) for v in phases)
        phi.extend(math.sin(v) / math.sqrt(32) for v in phases)
        a = (data["action_goal_delta"][index, :3] - np.asarray(parameters["translation_mean"])) / np.asarray(
            parameters["translation_scale"]
        )
        rows.append(phi + [v * a[axis] for axis in range(3) for v in phi])
    return np.asarray(rows)


def prediction(data, parameters):
    return nominal_prediction(data, parameters) + feature_matrix(data, parameters) @ np.asarray(
        parameters["coefficients"]
    )
