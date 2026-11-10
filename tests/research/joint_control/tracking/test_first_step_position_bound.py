"""The first-command outer bound must hold for exact frozen TCP positions."""

from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.joint_control.tracking.first_step_position_bound import first_step_position_bound


def example(*, lever=0.0):
    site, tool = np.eye(4), np.eye(4)
    tool[:3, 3] = [0.0, 0.0, lever]
    jacobian = np.c_[np.eye(6), np.zeros(6)]
    displacement = np.zeros(7)
    command_jacobian = np.zeros((7, 6))
    command_jacobian[:6] = np.eye(6)
    seed = previous = np.zeros(6)
    lower, upper = -np.ones(6) * 0.1, np.ones(6) * 0.1
    target = np.array([0.2, 0.0, lever])
    directions = np.vstack((np.eye(3), -np.eye(3), np.zeros(3)))
    return (site, tool, jacobian, displacement, command_jacobian, seed, previous, lower, upper, target, directions)


def exact_position(args, command):
    site, tool, jacobian, displacement, command_jacobian, seed, *_ = args
    movement = jacobian @ (displacement + command_jacobian @ (command - seed))
    return site[:3, 3] + movement[:3] + Rotation.from_rotvec(movement[3:]).apply(site[:3, :3] @ tool[:3, 3])


def test_zero_lever_gives_known_unreachable_position_bound():
    args = example()
    bound = first_step_position_bound(*args)
    assert bound["lower_bound_m"] == pytest.approx(0.1 - 1e-10)
    assert bound["rotation_remainder_bound_m"] == 0.0
    assert bound["directions"][-1]["reason"] == "zero_direction"
    assert np.linalg.norm(args[9] - exact_position(args, np.array([0.1, 0, 0, 0, 0, 0]))) == pytest.approx(0.1)


def test_rotated_lever_bound_holds_for_corners_and_interior_samples():
    args = list(example(lever=0.17))
    args[0][:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
    args[3][:6] = [0.01, -0.02, 0.03, 0.2, -0.1, 0.15]
    args[4][:6] += np.random.default_rng(42).normal(0, 0.04, (6, 6))
    args[10] = np.vstack((np.eye(3), -np.eye(3), args[9] - exact_position(args, args[5])))
    bound = first_step_position_bound(*args)
    corners = np.asarray(list(product(*zip(bound["command_lower"], bound["command_upper"], strict=True))))
    interior = np.random.default_rng(42).uniform(bound["command_lower"], bound["command_upper"], (1000, 6))
    for command in np.vstack((corners, interior)):
        assert np.linalg.norm(args[9] - exact_position(args, command)) >= bound["lower_bound_m"] - 1e-12


def test_rejects_infeasible_seed_and_empty_directions():
    args = list(example())
    args[5] = np.ones(6)
    with pytest.raises(ValueError, match="Seed"):
        first_step_position_bound(*args)
    args = list(example())
    args[10] = np.zeros((1, 3))
    with pytest.raises(ValueError, match="nonzero direction"):
        first_step_position_bound(*args)
