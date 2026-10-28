"""Inner-step affine structure and explicit model-witness classification."""

import numpy as np

from s2a.interior_joint_plan import solve_interior_plan
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface


def test_every_inner_joint_position_is_affine_in_all_physical_commands():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + 0.02 * a.T @ a
    q = rng.normal(size=7) * 0.1
    model = build_joint_preview(
        mass,
        np.c_[np.eye(6), rng.normal(size=6) * 0.1],
        np.ones(6) * 100,
        np.ones(6) * 20,
        q,
        q + 0.1,
        null_kp=10.0,
        dt=0.002,
    )
    delta = rng.normal(size=(8, 6)) * 0.01
    change = rng.normal(size=(8, 6)) * 0.01
    args = (model, np.ones(7) * 0.02)
    first = rollout_joint_preview(*args, delta, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)
    second = rollout_joint_preview(*args, delta + change, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)
    expected = first["inner_joint_position"] + np.einsum(
        "snw,w->sn", first["inner_joint_position_jacobian"], change.ravel()
    )
    assert expected.shape == (200, 7)
    np.testing.assert_allclose(second["inner_joint_position"], expected, atol=1e-11, rtol=0)


def test_feasible_interior_witness_is_checked_at_all_inner_steps_and_bad_initial_is_distinct():
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    seed = np.zeros((8, 6))
    seed[:, 0], seed[:, 3] = 0.05, 1e-9
    for q0 in (0.94, 0.96):
        q = np.zeros(7)
        q[0] = q0
        model = build_joint_preview(
            np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
        )
        target = rollout_joint_preview(
            model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False
        )
        result = solve_interior_plan(
            model,
            np.zeros(7),
            seed,
            np.zeros(6),
            interface,
            np.eye(4),
            np.eye(4),
            target["pose"],
            target["twist"],
            -np.ones(7),
            np.ones(7),
        )
        if q0 == 0.94:
            assert result["witness_found"]
            assert result["status"] == "model_witness"
            assert result["minimum_inner_margin_rad"] >= 0.05 - 1e-10
            assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12
        else:
            assert result["status"] == "initial_outside_requested_interior"
            assert not result["witness_found"]


def test_linear_solver_no_witness_is_separate_from_initial_margin_failure():
    q = np.zeros(7)
    q[0] = 0.94
    velocity = np.zeros(7)
    velocity[0] = 5.0
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 3] = 1e-9
    target = rollout_joint_preview(model, velocity, seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    result = solve_interior_plan(
        model,
        velocity,
        seed,
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        target["pose"],
        target["twist"],
        -np.ones(7),
        np.ones(7),
    )
    assert result["initial_margin_rad"] > 0.05
    assert not result["witness_found"]
    assert result["status"] == "linear_solver_no_witness"
    assert result["linear_status"] == 2
