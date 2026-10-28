"""Exact TCP composition and safeguards against an attractive rotation tradeoff."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error
from s2a.joint_limit_pose_guard import guard_command, predicted_tcp


def test_tcp_offset_is_rotated_after_noncommuting_world_innovations():
    site, tool = np.eye(4), np.eye(4)
    tool[:3, 3] = [0.1, 0.02, 0]
    response, bias = np.zeros(12), np.zeros(12)
    response[3:6], bias[3:6] = [0.4, 0, 0], [0, 0.3, 0]
    actual = predicted_tcp(site, tool, response, bias)
    expected = (Rotation.from_rotvec(bias[3:6]) * Rotation.from_rotvec(response[3:6])).as_matrix()
    np.testing.assert_allclose(actual[:3, :3], expected, atol=1e-12)
    np.testing.assert_allclose(actual[:3, 3], expected @ tool[:3, 3], atol=1e-12)
    assert np.linalg.norm(expected - Rotation.from_rotvec(bias[3:6] + response[3:6]).as_matrix()) > 0.01


def test_pose_guard_finds_position_improvement_without_accepting_rotation_trade():
    seed, proposal = np.zeros(6), np.zeros(6)
    proposal[:2] = [0.2, 0.2]
    desired = np.eye(4)
    desired[0, 3] = 0.2

    def response(command):
        value = np.zeros(12)
        value[0], value[5] = command[0], command[1]
        return value

    # This score prefers a rotational excursion; the pose guard must still reject it.
    def score(value):
        return (value[0] - 0.2) ** 2 + (value[5] - 0.2) ** 2

    result = guard_command(
        seed,
        proposal,
        -np.ones(6) * 0.25,
        np.ones(6) * 0.25,
        response,
        score,
        np.eye(4),
        np.eye(4),
        desired,
        np.zeros(12),
    )
    assert result["proposal_pose_feasible"] is False
    assert result["command"][0] > 0.1
    assert abs(result["command"][1]) < 1e-10
    assert result["candidate_objective"] < result["seed_objective"]
    seed_error = pose_error(desired, predicted_tcp(np.eye(4), np.eye(4), response(seed), np.zeros(12)))
    assert np.linalg.norm(result["chosen_tcp_error"][:3]) <= np.linalg.norm(seed_error[:3]) + 1e-10
    assert np.linalg.norm(result["chosen_tcp_error"][3:]) <= np.linalg.norm(seed_error[3:]) + 1e-10
