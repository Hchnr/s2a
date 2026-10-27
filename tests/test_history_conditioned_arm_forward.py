import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.history_conditioned_arm_forward import design, nominal, predict, with_history


def contract():
    return dict(dt=0.05, kp=np.ones(6) * 150, kd=np.ones(6) * 2 * np.sqrt(150))


def sequence(n=4):
    pose = np.repeat(np.eye(4)[None], n, axis=0)
    return dict(
        x_site_pose=pose,
        x_site_twist=np.zeros((n, 6)),
        action_goal_delta=np.full((n, 6), 0.01),
        x_previous_command_available=np.arange(n) > 0,
    )


def test_innovation_uses_actual_previous_goal_and_world_group_rotation_error():
    data = sequence(2)
    p = contract()
    data["x_site_pose"][0, :3, :3] = Rotation.from_rotvec([0, 0, 1.2]).as_matrix()
    old = nominal(data, p)[0]
    dp = np.array([0.001, -0.002, 0.003])
    dr = np.array([0.02, -0.03, 0.01])
    dv = np.arange(6) * 0.01
    data["x_site_pose"][1, :3, 3] = old[:3] + dp
    data["x_site_pose"][1, :3, :3] = (
        Rotation.from_rotvec(dr).as_matrix()
        @ Rotation.from_rotvec(old[3:6]).as_matrix()
        @ data["x_site_pose"][0, :3, :3]
    )
    data["x_site_twist"][1] = old[6:] / 0.05 + dv
    value = with_history(data, p)["x_response_residual"]
    np.testing.assert_allclose(value[1], np.r_[dp, dr, 0.05 * dv], atol=1e-14)
    np.testing.assert_array_equal(value[0], np.zeros(12))


def test_future_observations_actions_and_labels_cannot_change_current_history():
    data = sequence()
    before = with_history(data, contract())["x_response_residual"]
    later = copy.deepcopy(data)
    later["x_site_pose"][3, :3, 3] = [1, 2, 3]
    later["x_site_twist"][3] = 3
    later["action_goal_delta"][2:] = 0.2
    later["y_twist"] = np.full((4, 6), np.nan)
    after = with_history(later, contract())["x_response_residual"]
    np.testing.assert_array_equal(before[:3], after[:3])
    assert "x_response_residual" not in data


def test_explicit_reset_isolates_episodes_and_isolated_anchor_cannot_silently_zero_history():
    data = sequence()
    data["x_previous_command_available"][2] = False
    np.testing.assert_array_equal(with_history(data, contract())["x_response_residual"][2], np.zeros(12))
    cut = {k: v[1:2].copy() for k, v in data.items()}
    with pytest.raises(ValueError):
        with_history(cut, contract())


def test_prepared_anchor_keeps_history_and_translation_affinity():
    data = sequence(2)
    data.update(
        x_joint_position=np.ones((2, 7)) * 0.2,
        x_joint_velocity=np.zeros((2, 7)),
        x_finger_qpos=np.zeros((2, 2)),
        x_finger_qvel=np.zeros((2, 2)),
        x_motor_before=np.zeros((2, 2)),
        action_command=np.zeros((2, 7)),
        x_previous_command=np.zeros((2, 7)),
        x_previous_goal_pos=np.zeros((2, 3)),
        x_previous_goal_ori=np.repeat(np.eye(3)[None], 2, axis=0),
    )
    complete = with_history(data, contract())
    a = {k: v[1:2].copy() for k, v in complete.items()}
    b = copy.deepcopy(a)
    a["action_goal_delta"][0, :3] = [-0.02, 0.01, 0.02]
    b["action_goal_delta"][0, :3] = [0.01, -0.01, -0.01]
    middle = copy.deepcopy(a)
    middle["action_goal_delta"][0, :3] = 0.6 * a["action_goal_delta"][0, :3] + 0.4 * b["action_goal_delta"][0, :3]
    p = dict(
        contract(),
        state_mean=np.zeros(83),
        state_scale=np.ones(83),
        translation_mean=np.zeros(3),
        translation_scale=np.ones(3) * 0.05,
        coefficients=np.random.default_rng(42).normal(size=(336, 12)) * 0.001,
    )
    np.testing.assert_allclose(design(middle, p), 0.6 * design(a, p) + 0.4 * design(b, p), atol=1e-12)
    np.testing.assert_allclose(predict(middle, p), 0.6 * predict(a, p) + 0.4 * predict(b, p), atol=1e-12)
    assert np.linalg.norm(middle["x_response_residual"]) > 0
