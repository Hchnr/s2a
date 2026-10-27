import copy

import numpy as np

from s2a.state_conditioned_arm_forward import design, nominal, predict, raw_condition


def sample():
    n = 2
    pose = np.repeat(np.eye(4)[None], n, axis=0)
    data = dict(
        x_joint_position=np.full((n, 7), 0.2),
        x_joint_velocity=np.full((n, 7), 0.1),
        x_site_pose=pose,
        x_site_twist=np.full((n, 6), 0.02),
        x_finger_qpos=np.tile([0.02, -0.02], (n, 1)),
        x_finger_qvel=np.zeros((n, 2)),
        x_motor_before=np.zeros((n, 2)),
        action_command=np.zeros((n, 7)),
        x_previous_command=np.zeros((n, 7)),
        x_previous_goal_pos=np.zeros((n, 3)),
        x_previous_goal_ori=pose[:, :3, :3].copy(),
        action_goal_delta=np.full((n, 6), 0.01),
        x_previous_command_available=np.array([False, True]),
    )
    return data


def parameters():
    return dict(
        state_mean=np.zeros(71),
        state_scale=np.ones(71),
        translation_mean=np.zeros(3),
        translation_scale=np.ones(3) * 0.05,
        kp=np.ones(6) * 150,
        kd=np.ones(6) * 2 * np.sqrt(150),
        dt=0.05,
        coefficients=np.random.default_rng(42).normal(size=(288, 12)) * 0.001,
    )


def test_future_labels_and_episode_metadata_are_not_forward_inputs():
    data = sample()
    before = raw_condition(data)
    data.update(
        y_motion=np.full((2, 3), np.nan),
        y_twist=np.full((2, 6), 1e9),
        task_id=99,
        phase="release",
        time=np.ones(2) * 1e9,
    )
    np.testing.assert_array_equal(raw_condition(data), before)
    assert before.shape == (2, 71)


def test_translation_action_affinity_holds_with_coupled_residual_and_unchanged_condition():
    a = sample()
    b = copy.deepcopy(a)
    a["action_goal_delta"][:, :3] = [-0.02, 0.01, 0.03]
    b["action_goal_delta"][:, :3] = [0.01, -0.01, -0.02]
    middle = copy.deepcopy(a)
    alpha = 0.37
    middle["action_goal_delta"][:, :3] = (1 - alpha) * a["action_goal_delta"][:, :3] + alpha * b["action_goal_delta"][
        :, :3
    ]
    p = parameters()
    np.testing.assert_array_equal(raw_condition(a), raw_condition(b))
    np.testing.assert_allclose(design(middle, p), (1 - alpha) * design(a, p) + alpha * design(b, p), atol=1e-12)
    np.testing.assert_allclose(predict(middle, p), (1 - alpha) * predict(a, p) + alpha * predict(b, p), atol=1e-12)


def test_prospective_motor_feature_uses_actual_current_command_and_fixed_substep_rules():
    data = sample()
    data["x_motor_before"][:] = [[-0.97, 0.97], [-1, 1]]
    data["action_command"][:, -1] = [1, -1]
    condition = raw_condition(data)
    np.testing.assert_array_equal(condition[:, 47:49], data["x_motor_before"])
    np.testing.assert_allclose(condition[:, 49:51], [[-1, 1], [-0.75, 0.75]], atol=1e-15)
    data["action_command"][:, -1] = 0
    np.testing.assert_array_equal(raw_condition(data)[:, 49:51], data["x_motor_before"])


def test_zero_residual_matches_independent_critical_pd_solution_in_motion_and_velocity_units():
    data = sample()
    p = parameters()
    p["coefficients"] = np.zeros((288, 12))
    a = np.sqrt(150)
    dt = 0.05
    decay = np.exp(-a * dt)
    b = decay * dt
    r = 1 - decay * (1 + a * dt)
    expected = np.c_[
        r * data["action_goal_delta"] + b * data["x_site_twist"],
        dt * (150 * b * data["action_goal_delta"] + decay * (1 - a * dt) * data["x_site_twist"]),
    ]
    np.testing.assert_allclose(nominal(data, p), expected, atol=1e-14)
    np.testing.assert_allclose(predict(data, p), expected, atol=1e-14)
