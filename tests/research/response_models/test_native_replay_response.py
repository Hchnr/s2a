"""Causal recurrence and saturation checks for the separate response variant."""

import numpy as np
from numpy.testing import assert_array_equal

from s2a.controllers.osc import OSCInterface
from s2a.research.response_models.forward.native_replay_joint_response import rollout_replay_order_joint_response
from s2a.research.trajectory_search.replay_order_trajectory import rollout_replay_order_trajectory


def dynamics(q, velocity):
    pose = np.eye(4)
    pose[:3, 3] = q[:3]
    return dict(
        mass=np.eye(7),
        jacobian=np.eye(6, 7),
        pose=pose,
        bias=np.zeros(7),
        damping=np.ones(7),
    )


def test_clipped_implicit_damping_recurrence_matches_scalar_equation():
    delta = np.array([1.0, 0, 0, 0, 0, 0])
    result = rollout_replay_order_joint_response(
        np.zeros(7),
        np.zeros(7),
        delta,
        np.zeros(7),
        np.ones(6),
        np.ones(6),
        dynamics,
        -np.ones(7) * 0.01,
        np.ones(7) * 0.01,
        dt=0.002,
        substeps=25,
        update_matrices=True,
        latched_rotation_goal=np.eye(3),
    )
    q, velocity = 0.0, 0.0
    for i in range(25):
        velocity = (velocity + 0.002 * 0.01) / (1 + 0.002)
        q += 0.002 * velocity
        assert result["joint_position"][i, 0] == q
        assert result["joint_velocity"][i, 0] == velocity
    assert result["clipped"][:, 0].all()
    assert not result["clipped"][:, 1:].any()


def test_next_period_queries_predicted_state_and_keeps_causal_latch():
    calls = []

    def query(q, velocity):
        calls.append((q.copy(), velocity.copy()))
        return dynamics(q, velocity)

    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    commands = np.zeros((2, 6))
    commands[:, 0] = 0.1
    latch = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = rollout_replay_order_trajectory(
        np.zeros(7),
        np.zeros(7),
        commands,
        interface,
        np.zeros(7),
        np.ones(6),
        np.ones(6),
        query,
        -np.ones(7) * 10,
        np.ones(7) * 10,
        latched_rotation_goal=latch,
    )
    assert len(calls) == 52
    assert_array_equal(calls[26][0], result["joint_position"][0])
    assert_array_equal(calls[26][1], result["joint_velocity"][0])
    assert_array_equal(result["goals"][:, :3, :3], np.stack([latch, latch]))
    # The fixed affine interface rounds 0.1 during decoding; the goal uses
    # that decoded displacement, rather than an assumed identity mapping.
    assert result["goals"][1, 0, 3] == result["joint_position"][0, 0] + interface.decode(commands[1])[0]
