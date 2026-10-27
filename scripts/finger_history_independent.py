"""198 independent scalar extension of audited current-condition features."""

import numpy as np
from arm_joint_wrench_independent import INPUTS as CURRENT_INPUTS
from arm_joint_wrench_independent import feature_matrix as current_features

INPUTS = (*CURRENT_INPUTS, "x_finger_contact_history")


def feature_matrix(data, parameters):
    old = current_features(data, parameters)
    history = np.asarray(data["x_finger_contact_history"])
    mean = parameters["finger_history_mean"]
    scale = parameters["finger_history_scale"]
    assert history.shape == (len(old), 39)
    result = []
    for i, row in enumerate(old):
        standardized = [(float(history[i, j]) - float(mean[j])) / float(scale[j]) for j in range(39)]
        values = []
        for block in range(4):
            segment = row[282 * block : 282 * (block + 1)]
            values.extend(segment.tolist())
            values.extend(float(segment[0]) * v for v in standardized)
        result.append(values)
    return np.asarray(result)
