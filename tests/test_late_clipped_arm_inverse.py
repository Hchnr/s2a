import numpy as np
from test_direct_arm_inverse import parameters
from test_direct_arm_inverse_inputs import prepared

from s2a.direct_arm_inverse import command_prediction as early_prediction
from s2a.late_clipped_arm_inverse import command_prediction
from s2a.late_clipped_arm_inverse_runtime import translation_command


def test_far_target_does_not_lose_physical_authority_before_adding_residual():
    data, desired = prepared()
    p = parameters()
    desired[:, :3, 3] = [0.1, -0.1, 0.1]
    p["coefficients"][84:87] = -100 * np.eye(3)
    early = early_prediction(data, desired, gripper_command=np.zeros(2), parameters=p)
    late = command_prediction(data, desired, gripper_command=np.zeros(2), parameters=p)
    np.testing.assert_array_equal(early["command"], [[-1, 1, -1]] * 2)
    np.testing.assert_array_equal(late["command"], [[1, -1, 1]] * 2)
    runtime = translation_command(data, desired, gripper_command=np.zeros(2), parameters=p)
    np.testing.assert_array_equal(runtime["command"], [[0.25, -0.25, 0.25]] * 2)
    np.testing.assert_allclose(runtime["physical_delta"], runtime["command"] * 0.05, atol=1e-15)


def test_unsaturated_baseline_keeps_identical_frozen_predictions():
    data, desired = prepared()
    p = parameters()
    desired[:, :3, 3] = 0.0001
    p["coefficients"][0] = [0.01, -0.02, 0.03]
    early = early_prediction(data, desired, gripper_command=np.zeros(2), parameters=p)
    late = command_prediction(data, desired, gripper_command=np.zeros(2), parameters=p)
    for key in early:
        np.testing.assert_allclose(late[key], early[key], atol=1e-15)
