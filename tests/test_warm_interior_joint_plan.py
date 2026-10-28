"""A causal feasible warm candidate survives an unsuccessful later search."""

from types import SimpleNamespace

import numpy as np

import s2a.warm_interior_joint_plan as solver
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface


def test_feasible_warm_start_is_retained_without_claiming_optimizer_success(monkeypatch):
    q = np.zeros(7)
    q[0] = 0.94
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 0] = 0.05
    seed[:, 3] = 1e-9
    warm = np.zeros_like(seed)
    warm[:, 3] = 1e-9
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)

    def stopped(*args, **kwargs):
        np.testing.assert_array_equal(args[1], warm.ravel())
        return SimpleNamespace(success=False, nit=0, message="synthetic interrupted search")

    monkeypatch.setattr(solver, "minimize", stopped)
    result = solver.solve_interior_plan(
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
        initial_guess=warm,
    )
    assert result["witness_found"]
    assert not result["optimizer_success"]
    assert result["initial_guess_used"]
    np.testing.assert_array_equal(result["planned_commands"], warm)
    assert result["minimum_inner_margin_rad"] >= 0.05 - 1e-10
    np.testing.assert_allclose(result["first_pose_caps"], [0.01, 0.05], atol=1e-12, rtol=0)
