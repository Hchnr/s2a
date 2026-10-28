"""Reset-anchor consistency, native latch rejection and fixed-face derivatives."""

import numpy as np
import pytest

from s2a.current_coupled_response import build_response
from s2a.geometry import pose_error
from s2a.joint_limit_preview import build_joint_preview, rollout_joint_preview
from s2a.joint_limit_response import build_joint_response, predict_joint_response


def random_problem():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + 0.03 * a.T @ a
    j = np.c_[np.eye(6), rng.normal(size=6) * 0.1]
    q = rng.normal(size=7) * 0.1
    initial = q + rng.normal(size=7) * 0.1
    velocity = rng.normal(size=7) * 0.02
    commands = rng.normal(size=(4, 6)) * 0.02
    kp, kd = np.ones(6) * 100, np.ones(6) * 20
    model = build_joint_preview(mass, j, kp, kd, q, initial, null_kp=10.0, dt=0.002)
    return model, mass, j, kp, kd, q, initial, velocity, commands


def test_each_period_matches_independently_rebuilt_joint_anchor():
    model, mass, j, kp, kd, q, initial, velocity, commands = random_problem()
    result = rollout_joint_preview(
        model, velocity, commands, np.eye(4), -np.ones(7) * 2, np.ones(7) * 2, enforce_limits=False
    )
    for k, delta in enumerate(commands):
        rebuilt = build_joint_response(mass, j, kp, kd, q, initial, null_kp=10.0, dt=0.002)
        expected = predict_joint_response(
            rebuilt, velocity, delta, -np.ones(7) * 2, np.ones(7) * 2, substeps=25, enforce_limits=False
        )
        np.testing.assert_allclose(result["joint_position"][k], expected["joint_position"], atol=1e-11, rtol=0)
        np.testing.assert_allclose(result["joint_velocity"][k], expected["joint_velocity"], atol=1e-11, rtol=0)
        q, velocity = expected["joint_position"], expected["joint_velocity"]


def test_goal_resets_match_original_free_four_period_task_rollout():
    mass, j = np.eye(7), np.c_[np.eye(6), np.zeros(6)]
    model = build_joint_preview(
        mass, j, np.ones(6) * 100, np.ones(6) * 20, np.zeros(7), np.zeros(7), null_kp=10.0, dt=0.002
    )
    command = np.array([0.02, 0, 0, 1e-9, 0, 0])
    actual = rollout_joint_preview(
        model, np.zeros(7), np.tile(command, (4, 1)), np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False
    )
    task = build_response(mass, j, np.ones(6) * 100, np.ones(6) * 20, np.eye(3))
    state = np.zeros(12)
    for k in range(4):
        state = (
            np.r_[state[:6], np.zeros(6)]
            + task["initial_velocity_response"] @ state[6:]
            + task["goal_forcing_response"] @ command
        )
        np.testing.assert_allclose(actual["pose"][k, :3, 3], state[:3], atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual["twist"][k], state[6:], atol=1e-12, rtol=0)


@pytest.mark.parametrize("limited", [False, True])
def test_full_pose_and_joint_derivatives_match_finite_differences(limited):
    model, mass, j, kp, kd, q, initial, velocity, commands = random_problem()
    commands = commands[:2]
    lower, upper = -np.ones(7), np.ones(7)
    if limited:
        # Strict outward forcing at an upper face keeps the active set stable.
        q[0] = 1.0
        model = build_joint_preview(mass, j, kp, kd, q, initial, null_kp=10.0, dt=0.002)
        velocity[0] = 0.02
        commands[:, 0] = 0.1
    base = rollout_joint_preview(model, velocity, commands, np.eye(4), lower, upper, enforce_limits=limited)
    epsilon = 1e-6
    for column in range(commands.size):
        plus, minus = commands.copy(), commands.copy()
        plus.flat[column] += epsilon
        minus.flat[column] -= epsilon
        a = rollout_joint_preview(model, velocity, plus, np.eye(4), lower, upper, enforce_limits=limited)
        b = rollout_joint_preview(model, velocity, minus, np.eye(4), lower, upper, enforce_limits=limited)
        for k in range(2):
            numeric = pose_error(a["pose"][k], b["pose"][k]) / (2 * epsilon)
            np.testing.assert_allclose(numeric[:3], base["position_jacobian"][k, :, column], atol=2e-6, rtol=1e-5)
            np.testing.assert_allclose(numeric[3:], base["rotation_jacobian"][k, :, column], atol=2e-6, rtol=1e-5)
            for value, derivative in (
                ("joint_position", "joint_position_jacobian"),
                ("joint_velocity", "joint_velocity_jacobian"),
                ("twist", "twist_jacobian"),
            ):
                np.testing.assert_allclose(
                    (a[value][k] - b[value][k]) / (2 * epsilon), base[derivative][k, :, column], atol=2e-6, rtol=1e-5
                )
    np.testing.assert_array_equal(base["position_jacobian"][0, :, 6:], np.zeros((3, 6)))
    if limited:
        assert np.max(base["joint_position"][:, 0]) <= 1.0


def test_zero_rotation_latch_is_not_silently_modeled_as_reset():
    model, _, _, _, _, _, _, velocity, _ = random_problem()
    with pytest.raises(ValueError, match="latch mode is unsupported"):
        rollout_joint_preview(
            model, velocity, np.zeros((1, 6)), np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True
        )


def test_later_period_can_release_a_previously_active_limit():
    q = np.zeros(7)
    q[0] = 1.0
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    deltas = np.array([[0.1, 0, 0, 1e-9, 0, 0], [-0.1, 0, 0, 1e-9, 0, 0]])
    actual = rollout_joint_preview(model, np.zeros(7), deltas, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True)
    assert actual["joint_position"][0, 0] == pytest.approx(1.0)
    assert abs(actual["joint_velocity"][0, 0]) < 1e-10
    assert actual["joint_position"][1, 0] < 0.999
    assert actual["joint_velocity"][1, 0] < 0
