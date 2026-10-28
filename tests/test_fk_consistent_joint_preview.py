"""Noncommuting rotations, TCP offsets and velocity dependence need one FK model."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.fk_consistent_joint_preview import fk_consistent_rollout
from s2a.geometry import transform
from s2a.joint_limit_preview_objective import preview_objective


def site_fk(q):
    rz = Rotation.from_rotvec([0, 0, q[1]]).as_matrix()
    ry = Rotation.from_rotvec([0, q[2], 0]).as_matrix()
    rx = Rotation.from_rotvec([q[3], 0, 0]).as_matrix()
    rotation = rz @ ry @ rx
    offset = rotation @ np.array([0.3, 0.1, 0.2])
    axes = np.column_stack(([0, 0, 1], rz @ [0, 1, 0], rz @ ry @ [1, 0, 0]))
    jac = np.zeros((6, 4))
    jac[0, 0] = 1
    jac[:3, 1:] = np.cross(axes.T, offset).T
    jac[3:, 1:] = axes
    return transform(np.array([q[0], 0, 0]) + offset, rotation), jac


def setup():
    rng = np.random.default_rng(42)
    qjac = rng.normal(scale=0.07, size=(2, 4, 12))
    vjac = rng.normal(scale=0.13, size=(2, 4, 12))
    q = np.array([[0.1, 0.3, -0.2, 0.4], [0.12, 0.25, -0.35, 0.5]])
    velocity = rng.normal(scale=0.3, size=(2, 4))

    def rollout(x):
        return dict(
            joint_position=q + np.einsum("hnw,w->hn", qjac, x),
            joint_velocity=velocity + np.einsum("hnw,w->hn", vjac, x),
            joint_position_jacobian=qjac,
            joint_velocity_jacobian=vjac,
            pose=np.tile(np.eye(4), (2, 1, 1)),
        )

    tool = transform([0.04, -0.03, 0.08], Rotation.from_rotvec([0.1, 0.2, -0.1]).as_matrix())
    targets = np.asarray([site_fk(row + [0.02, -0.04, 0.05, 0.03])[0] @ tool for row in q])
    twists = rng.normal(scale=0.2, size=(2, 6))
    return rollout, tool, targets, twists


def test_full_pose_twist_residual_and_objective_gradient_matches_finite_difference():
    rollout, tool, targets, twists = setup()
    x = np.linspace(-0.1, 0.1, 12)

    def score(value):
        return preview_objective(fk_consistent_rollout(rollout(value), site_fk), tool, targets, twists)

    analytic = score(x)
    numeric_jac, numeric_gradient = [], []
    for column in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        a, b = score(plus), score(minus)
        numeric_jac.append((a["residual"] - b["residual"]) / 2e-6)
        numeric_gradient.append((a["objective"] - b["objective"]) / 2e-6)
    np.testing.assert_allclose(analytic["jacobian"], np.asarray(numeric_jac).T, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(analytic["gradient"], numeric_gradient, atol=2e-6, rtol=1e-5)


def test_cartesian_fields_follow_predicted_joints_without_mutating_joint_response():
    rollout, _, _, _ = setup()
    original = rollout(np.zeros(12))
    saved = {key: value.copy() for key, value in original.items()}
    result = fk_consistent_rollout(original, site_fk)
    for key, before in saved.items():
        np.testing.assert_array_equal(original[key], before)
    for index, q in enumerate(original["joint_position"]):
        pose, jac = site_fk(q)
        np.testing.assert_array_equal(result["pose"][index], pose)
        np.testing.assert_array_equal(result["twist"][index], jac @ original["joint_velocity"][index])
    assert np.max(abs(result["pose"] - original["pose"])) > 0.1
    np.testing.assert_array_equal(result["joint_position_jacobian"], original["joint_position_jacobian"])
