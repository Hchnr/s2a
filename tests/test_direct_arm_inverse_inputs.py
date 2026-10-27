import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_state_conditioned_arm_forward import sample

from s2a.direct_arm_inverse_inputs import inverse_condition


def prepared():
    data = sample()
    data["x_response_residual"] = np.zeros((2, 12))
    data["x_current_wrench_world"] = np.zeros((2, 6))
    target = data["x_site_pose"].copy()
    target[:, :3, 3] = [0.01, -0.02, 0.03]
    target[:, :3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.05]).as_matrix()
    return data, target


def test_target_actions_and_future_labels_are_excluded_from_inverse_input():
    data, target = prepared()
    grip = np.array([1.0, 0.0])
    original = copy.deepcopy(data)
    before = inverse_condition(data, target, gripper_command=grip)
    changed = copy.deepcopy(data)
    changed["action_command"][:] = np.nan
    changed["action_goal_delta"][:] = np.nan
    changed["y_motion"] = np.full((2, 3), np.nan)
    changed["y_twist"] = np.full((2, 6), 1e9)
    np.testing.assert_array_equal(inverse_condition(changed, target, gripper_command=grip), before)
    for key in original:
        np.testing.assert_array_equal(data[key], original[key])


def test_explicit_generated_pose_uses_world_motion_and_so3_convention():
    data, target = prepared()
    value = inverse_condition(data, target, gripper_command=np.zeros(2))
    np.testing.assert_allclose(value[:, -6:-3], [[0.01, -0.02, 0.03]] * 2, atol=1e-15)
    np.testing.assert_allclose(value[:, -3:], [[0.2, -0.1, 0.05]] * 2, atol=1e-15)
    changed = target.copy()
    changed[0, 0, 3] += 0.001
    after = inverse_condition(data, changed, gripper_command=np.zeros(2))
    np.testing.assert_array_equal(value[1], after[1])
    assert after[0, -6] - value[0, -6] == pytest.approx(0.001)


def test_current_gripper_candidate_is_explicit_and_invalid_input_rejected():
    data, target = prepared()
    close = inverse_condition(data, target, gripper_command=np.ones(2))
    open_ = inverse_condition(data, target, gripper_command=-np.ones(2))
    assert not np.array_equal(close, open_)
    with pytest.raises(ValueError, match="gripper"):
        inverse_condition(data, target, gripper_command=np.ones(3))
