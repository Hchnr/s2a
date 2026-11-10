"""Support bounds must retain feasible commands and detect combined conflicts."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.joint_control.inverse.full_coupled_pose_inverse import hat
from s2a.research.joint_control.tracking.stopped_pose_safety_bound import (
    error_directions,
    first_pose_linearization,
    safety_scopes,
    support_upper_bound,
    weighted_pose_bounds,
)


def test_corrected_dual_bounds_constrained_support():
    scope = dict(
        lower=np.zeros(2),
        upper=np.full(2, 2.0),
        seed=np.zeros(2),
        a=np.array([[1.0, 1.0], [1.0, 0.0]]),
        b=np.array([1.0, 0.7]),
    )
    result = support_upper_bound(np.array([1.0, 2.0]), scope)
    assert result["support_upper"] == pytest.approx(2.0, abs=1e-10)
    assert result["support_upper"] >= result["primal_support"]
    for point in np.random.default_rng(42).uniform(0, 1, (1000, 2)):
        if np.all(scope["a"] @ point <= scope["b"]):
            assert np.array([1.0, 2.0]) @ point <= result["support_upper"]


def test_joint_pose_bound_detects_conflict_missed_by_individual_caps():
    site, tool, target = np.eye(4), np.eye(4), np.eye(4)
    target[0, 3] = 0.022
    jacobian, command_jac = np.zeros((6, 7)), np.zeros((7, 6))
    jacobian[0, 0], jacobian[5, 0], command_jac[0, 0] = 0.044, 0.22, 1
    lower, upper = np.full(6, -0.25), np.full(6, 0.25)
    lower[0], upper[0] = 0, 0.5
    seed = np.zeros(6)
    linear = first_pose_linearization(site, tool, jacobian, np.zeros(7), command_jac, seed, lower, upper, target)
    scope = dict(lower=lower, upper=upper, seed=seed, a=np.empty((0, 6)), b=np.empty(0))
    n, d = np.array([1.0, 0, 0]), -hat(np.array([0.0, 0, 1.0])) / np.sqrt(2)
    result = weighted_pose_bounds(linear, n, d, {"box": scope}, [0, 0.5, 1])["scopes"]["box"]
    bounds = [row["lower_bound"] for row in result["rows"]]
    assert bounds[0] < 1 and bounds[2] < 1
    assert bounds[1] > 1
    # Translation alone needs u >= .2727; orientation alone needs u <= .2273.
    for command in np.linspace(0, 0.5, 501):
        position = abs(0.022 - 0.044 * command)
        angle = 0.22 * command
        assert position > 0.01 or angle > 0.05


def test_existing_feasible_pose_is_never_excluded_by_the_box():
    site, tool, target = np.eye(4), np.eye(4), np.eye(4)
    target[0, 3] = 0.009
    jacobian, command_jac = np.zeros((6, 7)), np.zeros((7, 6))
    jacobian[0, 0], jacobian[5, 0], command_jac[0, 0] = 0.044, 0.22, 1
    lower, upper = np.full(6, -0.25), np.full(6, 0.25)
    seed = np.zeros(6)
    linear = first_pose_linearization(site, tool, jacobian, np.zeros(7), command_jac, seed, lower, upper, target)
    scope = dict(lower=lower, upper=upper, seed=seed, a=np.empty((0, 6)), b=np.empty(0))
    result = weighted_pose_bounds(linear, [1, 0, 0], hat([0, 0, 1]) / np.sqrt(2), {"box": scope}, [0, 0.5, 1])
    assert result["scopes"]["box"]["best_lower_bound"] <= 1


def test_first_period_projection_does_not_drop_future_coefficient():
    joint_a = np.zeros((1456, 48))
    joint_a[0, 0], joint_a[0, 6] = 1.0, 0.5
    lo, hi = -np.ones(48), np.ones(48)
    domain = dict(
        lo=lo,
        hi=hi,
        rate=np.eye(48) - np.eye(48, k=-6),
        center=np.zeros(48),
        joint_a=joint_a,
        joint_lo=-np.ones(1456),
        joint_hi=np.r_[0.0, np.ones(1455)],
    )
    scope = safety_scopes(domain, np.zeros((8, 6)))["first_interval_safety"]
    point = np.r_[0.25, np.zeros(5)]
    assert np.all(scope["a"] @ point <= scope["b"])


def test_pose_bound_holds_for_exact_rotated_tcp_samples():
    rng = np.random.default_rng(42)
    site, tool, target = np.eye(4), np.eye(4), np.eye(4)
    site[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    tool[:3, 3] = [0.04, -0.02, 0.17]
    tool[:3, :3] = Rotation.from_rotvec([-0.1, 0.2, 0.05]).as_matrix()
    target[:3, 3] = [0.02, 0.01, 0.21]
    target[:3, :3] = Rotation.from_rotvec([0.2, 0.1, -0.1]).as_matrix()
    jacobian = rng.normal(0, 0.2, (6, 7))
    command_jac = rng.normal(0, 0.1, (7, 6))
    displacement = rng.normal(0, 0.1, 7)
    seed, lower, upper = np.zeros(6), -np.ones(6) * 0.25, np.ones(6) * 0.25
    linear = first_pose_linearization(site, tool, jacobian, displacement, command_jac, seed, lower, upper, target)
    n, d = error_directions(target[:3, :3], [0.01, -0.02, 0.03, 0.03, 0.04, -0.02])
    scope = dict(lower=lower, upper=upper, seed=seed, a=np.empty((0, 6)), b=np.empty(0))
    results = weighted_pose_bounds(linear, n, d, {"box": scope}, [0, 0.25, 0.5, 0.75, 1])
    rotation_cap = results["rotation_chord_cap"]
    offset = site[:3, :3] @ tool[:3, 3]
    for command in rng.uniform(lower, upper, (300, 6)):
        movement = jacobian @ (displacement + command_jac @ command)
        rotation = Rotation.from_rotvec(movement[3:]).as_matrix()
        position = site[:3, 3] + movement[:3] + rotation @ offset
        tcp_rotation = rotation @ site[:3, :3] @ tool[:3, :3]
        p = n @ (target[:3, 3] - position) / 0.01
        r = np.sum(d * (target[:3, :3] - tcp_rotation)) / rotation_cap
        for row in results["scopes"]["box"]["rows"]:
            assert row["weight"] * p + (1 - row["weight"]) * r >= row["lower_bound"] - 1e-10
