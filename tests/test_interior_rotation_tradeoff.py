"""First-rotation objective derivative and absence of future-action influence."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import s2a.interior_rotation_tradeoff as solver
from s2a.interior_joint_preview import build_joint_preview
from s2a.osc import OSCInterface


def test_exact_so3_objective_gradient_and_later_actions_do_not_change_first_error(monkeypatch):
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + 0.01 * a.T @ a
    q = np.zeros(7)
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
    seed = np.ones((8, 6)) * 0.02
    target = np.tile(np.eye(4), (8, 1, 1))
    target[:, :3, :3] = Rotation.from_rotvec([0.02, -0.01, 0.015]).as_matrix()
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))

    def observe(function, start, **kwargs):
        _, gradient = function(start)
        for column in (0, 3, 5, 24):
            plus, minus = start.copy(), start.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            numeric = (function(plus)[0] - function(minus)[0]) / 2e-6
            np.testing.assert_allclose(numeric, gradient[column], atol=1e-9, rtol=1e-5)
        np.testing.assert_array_equal(gradient[6:], np.zeros(42))
        return SimpleNamespace(success=False, nit=0, message="gradient diagnostic only")

    monkeypatch.setattr(solver, "minimize", observe)
    result = solver.solve_interior_plan(
        model,
        np.ones(7) * 0.01,
        seed,
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        target,
        np.zeros((8, 6)),
        -np.ones(7),
        np.ones(7),
        initial_guess=seed,
    )
    assert result["witness_found"]
    assert not result["optimizer_success"]
    np.testing.assert_allclose(result["objective"], 0.5 * result["first_rotation_error_rad"] ** 2, atol=1e-15, rtol=0)
    assert result["minimum_inner_margin_rad"] >= 0.05 - 1e-10
