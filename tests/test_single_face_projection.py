"""Single-face shortcut preserves coupled QP solutions and rollout derivatives."""

import numpy as np

from s2a.fast_long_joint_preview import rollout_joint_preview as fast_rollout
from s2a.joint_limit_preview import _project as general_projection
from s2a.long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.single_face_projection import project


def test_coupled_box_and_multiface_fallback_match_general_qp():
    rng = np.random.default_rng(42)
    for _ in range(40):
        a = rng.normal(size=(7, 7))
        mass = np.eye(7) + a.T @ a
        inverse = np.linalg.inv(mass)
        constraints = np.vstack((np.eye(7), -np.eye(7)))
        bounds = np.ones(14)
        for outside in (1, 3):
            value = rng.uniform(-0.1, 0.1, 7)
            value[:outside] = 1.1
            initial = np.clip(value, -1, 1)
            actual = project(mass, inverse, value, constraints, bounds, initial)
            expected = general_projection(mass, value, constraints, bounds, initial)
            for k in (0, 1):
                np.testing.assert_allclose(actual[k], expected[k], atol=1e-11, rtol=0)
            assert actual[2] == expected[2]


def test_full_eight_period_release_preserves_states_and_derivatives():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + 0.01 * a.T @ a
    q = np.zeros(7)
    q[0] = 1
    model = build_joint_preview(
        mass,
        np.c_[np.eye(6), rng.normal(size=6) * 0.1],
        np.ones(6) * 100,
        np.ones(6) * 20,
        q,
        q,
        null_kp=10.0,
        dt=0.002,
    )
    delta = np.zeros((8, 6))
    delta[:, 0] = [0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1]
    delta[:, 3] = 1e-9
    velocity = np.zeros(7)
    velocity[0] = 0.02
    args = (model, velocity, delta, np.eye(4), -np.ones(7), np.ones(7))
    fast = fast_rollout(*args, enforce_limits=True)
    expected = rollout_joint_preview(*args, enforce_limits=True)
    for key in (
        "pose",
        "twist",
        "joint_position",
        "joint_velocity",
        "joint_position_jacobian",
        "joint_velocity_jacobian",
        "position_jacobian",
        "rotation_jacobian",
        "twist_jacobian",
    ):
        np.testing.assert_allclose(fast[key], expected[key], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(fast["weak_projection_steps"], expected["weak_projection_steps"])
