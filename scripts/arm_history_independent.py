"""090 independent absolute-rotation history and scalar conditional features."""

import numpy as np
from arm_forward_independent import INPUTS as BASE_INPUTS
from arm_forward_independent import conditions as base_conditions
from arm_forward_independent import nominal_prediction
from scipy.spatial.transform import Rotation

INPUTS = (*BASE_INPUTS, "x_response_residual")


def augment(data, parameters):
    count = len(data["x_site_pose"])
    available = np.asarray(data["x_previous_command_available"])
    assert count and available.shape == (count,) and np.isin(available, [0, 1]).all() and not available[0]
    nominal = nominal_prediction(data, parameters)
    residual = np.zeros((count, 12))
    for k in range(1, count):
        if not available[k]:
            continue
        before = np.asarray(data["x_site_pose"][k - 1])
        now = np.asarray(data["x_site_pose"][k])
        expected = nominal[k - 1]
        predicted_position = before[:3, 3] + expected[:3]
        predicted_rotation = Rotation.from_rotvec(expected[3:6]).as_matrix() @ before[:3, :3]
        residual[k] = np.r_[
            now[:3, 3] - predicted_position,
            Rotation.from_matrix(now[:3, :3] @ predicted_rotation.T).as_rotvec(),
            0.05 * np.asarray(data["x_site_twist"][k]) - expected[6:],
        ]
    result = dict(data)
    result["x_response_residual"] = residual
    return result


def conditions(data):
    value = np.c_[base_conditions(data), np.asarray(data["x_response_residual"])]
    assert value.shape[1] == 83 and np.isfinite(value).all()
    return value


def feature_matrix(data, parameters):
    raw = conditions(data)
    phi = np.c_[np.ones(len(raw)), (raw - np.array(parameters["state_mean"])) / np.array(parameters["state_scale"])]
    action = (np.asarray(data["action_goal_delta"])[:, :3] - np.array(parameters["translation_mean"])) / np.array(
        parameters["translation_scale"]
    )
    return np.concatenate([phi, *[phi * action[:, a : a + 1] for a in range(3)]], axis=1)


def prediction(data, parameters):
    return nominal_prediction(data, parameters) + feature_matrix(data, parameters) @ np.asarray(
        parameters["coefficients"]
    )
