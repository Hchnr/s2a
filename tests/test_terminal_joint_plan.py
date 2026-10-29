"""Predictive braking applies the correct joint-boundary velocity direction."""

import numpy as np
import pytest

from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface
from s2a.strict_terminal_joint_plan import solve_interior_plan as strict_solve
from s2a.terminal_joint_plan import solve_interior_plan


@pytest.mark.parametrize("solver", [solve_interior_plan, strict_solve])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_terminal_velocity_is_nonoutward_at_the_relevant_bound(sign, solver):
    q = np.zeros(7)
    q[0] = sign * 0.9
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 0] = sign * 0.1
    seed[:, 3] = 1e-9
    lo, hi = -np.ones(7), np.ones(7)
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), lo, hi, enforce_limits=False)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    result = solver(
        model, np.zeros(7), seed, np.zeros(6), interface, np.eye(4), np.eye(4), target["pose"], target["twist"], lo, hi
    )
    assert result["witness_found"]
    assert result["minimum_inner_margin_rad"] >= 0.05 - 1e-10
    assert sign * result["rollout"]["joint_velocity"][-1, 0] <= 1e-10
    assert len(result["terminal_normals"]) == 1
    np.testing.assert_allclose(result["terminal_normals"][0], np.eye(7)[0] * sign, atol=1e-12, rtol=0)
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12


def test_terminal_velocity_affine_map_matches_independent_action_perturbation():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + 0.01 * a.T @ a
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
    original = rollout_joint_preview(*args, delta, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)
    varied = rollout_joint_preview(*args, delta + change, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)
    expected = original["joint_velocity"][-1] + original["joint_velocity_jacobian"][-1] @ change.ravel()
    np.testing.assert_allclose(varied["joint_velocity"][-1], expected, atol=1e-11, rtol=0)
