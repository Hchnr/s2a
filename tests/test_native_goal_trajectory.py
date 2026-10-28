"""Cross-period state and goal history must come from the rollout itself."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.native_goal_joint_response import native_goal_rotation, rollout_native_goal_joint_response
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface


def _dynamics(q, velocity):
    pose = np.eye(4)
    pose[:3, 3] = q[:3]
    pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
    return dict(
        mass=np.diag(1 + q * q),
        jacobian=np.c_[np.eye(6), np.zeros(6)],
        pose=pose,
        bias=0.1 * q,
        damping=np.full(7, 0.02),
        passive_extra=np.zeros(7),
    )


def _arguments(commands):
    q = np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.15, 0.4])
    scale = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
    return dict(
        q=q,
        velocity=np.array([0.02, 0.0, -0.01, 0.0, 0.0, 0.1, 0.03]),
        commands=np.asarray(commands),
        interface=OSCInterface(-np.ones(6), np.ones(6), -scale, scale),
        initial_joint=q.copy(),
        kp=np.full(6, 20.0),
        kd=np.full(6, 4.0),
        dynamics=_dynamics,
        torque_lower=np.full(7, -20.0),
        torque_upper=np.full(7, 20.0),
        latched_rotation_goal=Rotation.from_rotvec([0.0, 0.0, -0.3]).as_matrix(),
    )


def test_cross_period_state_velocity_and_zero_rotation_latch_are_predicted_recursively():
    arguments = _arguments([[0.2, 0, 0, 0, 0, 0.1], [-0.1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -0.08]])
    originals = {key: arguments[key].copy() for key in ("q", "velocity", "commands", "latched_rotation_goal")}
    calls = []

    def dynamics(q, velocity):
        calls.append((q.copy(), velocity.copy()))
        response = _dynamics(q, velocity)
        q[:] = 900.0
        velocity[:] = -900.0
        return response

    arguments["dynamics"] = dynamics
    actual = rollout_native_goal_trajectory(**arguments)
    assert len(calls) == 3 * 26
    assert actual["inner_joint_position"].shape == actual["inner_joint_velocity"].shape == (75, 7)
    assert actual["joint_position"].shape == actual["joint_velocity"].shape == (3, 7)
    assert actual["pose"].shape == actual["goals"].shape == (3, 4, 4)
    assert actual["twist"].shape == (3, 6)
    for period, detail in enumerate(actual["periods"]):
        q_start = originals["q"] if period == 0 else actual["joint_position"][period - 1]
        v_start = originals["velocity"] if period == 0 else actual["joint_velocity"][period - 1]
        observed = calls[26 * period : 26 * (period + 1)]
        np.testing.assert_array_equal(
            np.asarray([item[0] for item in observed]), np.vstack((q_start, detail["joint_position"]))
        )
        np.testing.assert_array_equal(
            np.asarray([item[1] for item in observed]), np.vstack((v_start, detail["joint_velocity"]))
        )
        np.testing.assert_array_equal(
            actual["inner_joint_position"][25 * period : 25 * (period + 1)], detail["joint_position"]
        )
        np.testing.assert_array_equal(
            actual["inner_joint_velocity"][25 * period : 25 * (period + 1)], detail["joint_velocity"]
        )
        np.testing.assert_array_equal(actual["joint_position"][period], detail["joint_position"][-1])
        np.testing.assert_array_equal(actual["joint_velocity"][period], detail["joint_velocity"][-1])
        delta = arguments["interface"].decode(originals["commands"][period])
        np.testing.assert_array_equal(actual["goals"][period, :3, 3], q_start[:3] + delta[:3])
        assert detail["update_matrices"] is True and detail["dynamics_queries"] == 26
    initial_rotation = _dynamics(originals["q"], originals["velocity"])["pose"][:3, :3]
    np.testing.assert_array_equal(
        actual["goals"][0, :3, :3], native_goal_rotation(np.array([0, 0, 0.05])) @ initial_rotation
    )
    np.testing.assert_array_equal(actual["goals"][1, :3, :3], actual["goals"][0, :3, :3])
    assert not np.allclose(actual["goals"][1, :3, :3], actual["pose"][0, :3, :3])
    delta = arguments["interface"].decode(originals["commands"][2])
    np.testing.assert_array_equal(
        actual["goals"][2, :3, :3], native_goal_rotation(delta[3:]) @ actual["pose"][1, :3, :3]
    )
    assert not np.array_equal(actual["joint_velocity"][0], originals["velocity"])
    for key, original in originals.items():
        np.testing.assert_array_equal(arguments[key], original)


def test_first_zero_rotation_uses_explicit_current_controller_history():
    arguments = _arguments(np.zeros((2, 6)))
    actual = rollout_native_goal_trajectory(**arguments)
    for goal in actual["goals"]:
        np.testing.assert_array_equal(goal[:3, :3], arguments["latched_rotation_goal"])
    np.testing.assert_array_equal(actual["goals"][0, :3, 3], arguments["q"][:3])
    np.testing.assert_array_equal(actual["goals"][1, :3, 3], actual["pose"][0, :3, 3])
    assert not np.allclose(actual["goals"][0, :3, :3], actual["pose"][0, :3, :3])


def test_one_period_matches_original_native_goal_predictor():
    arguments = _arguments([[0.1, -0.2, 0.3, 0.0, 0.0, 0.1]])
    actual = rollout_native_goal_trajectory(**arguments)
    expected = rollout_native_goal_joint_response(
        arguments["q"],
        arguments["velocity"],
        arguments["interface"].decode(arguments["commands"][0]),
        arguments["initial_joint"],
        arguments["kp"],
        arguments["kd"],
        arguments["dynamics"],
        arguments["torque_lower"],
        arguments["torque_upper"],
        dt=0.002,
        substeps=25,
        update_matrices=True,
        latched_rotation_goal=arguments["latched_rotation_goal"],
    )
    assert set(actual["periods"][0]) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual["periods"][0][key], value, err_msg=key)
    np.testing.assert_array_equal(actual["pose"][0], expected["endpoint_pose"])
    np.testing.assert_array_equal(actual["twist"][0], expected["endpoint_twist"])


def test_invalid_horizon_commands_or_initial_goal_fail_before_any_prediction():
    def forbidden(q, velocity):
        raise AssertionError("Invalid initial input must not trigger any dynamics query")

    for commands in (
        np.zeros((0, 6)),
        np.zeros((17, 6)),
        np.zeros((2, 7)),
        np.full((1, 6), np.nan),
        np.full((1, 6), 1.01),
    ):
        arguments = _arguments(commands)
        arguments["dynamics"] = forbidden
        with np.testing.assert_raises(ValueError):
            rollout_native_goal_trajectory(**arguments)
    arguments = _arguments(np.zeros((1, 6)))
    arguments["dynamics"] = forbidden
    arguments["latched_rotation_goal"] = np.full((3, 3), np.nan)
    with np.testing.assert_raises(ValueError):
        rollout_native_goal_trajectory(**arguments)
