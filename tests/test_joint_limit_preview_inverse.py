"""Full-plan recovery under box/rate constraints and weak-face fallback."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.geometry import pose_error
from s2a.joint_limit_preview import build_joint_preview, rollout_joint_preview
from s2a.joint_limit_preview_feasible_inverse import refine_joint_plan as feasible_refine_joint_plan
from s2a.joint_limit_preview_inverse import refine_joint_plan
from s2a.joint_limit_preview_norm_inverse import refine_joint_plan as norm_refine_joint_plan
from s2a.osc import OSCInterface


def problem():
    q = np.zeros(7)
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    return model, interface


@pytest.mark.parametrize("solver", [refine_joint_plan, norm_refine_joint_plan, feasible_refine_joint_plan])
def test_known_four_period_plan_recovery_and_command_rate(solver):
    model, interface = problem()
    known = np.zeros((4, 6))
    known[:, 0], known[:, 3] = [0.1, 0.15, 0.1, 0.05], 1e-9
    target = rollout_joint_preview(model, np.zeros(7), known, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True)
    seed = np.zeros((4, 6))
    seed[:, 3] = 1e-9
    result = solver(
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
    assert result["objective"] < result["seed_objective"] * 1e-6
    np.testing.assert_allclose(result["planned_commands"][:, 0], known[:, 0], atol=1e-3, rtol=0)
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12
    assert result["response_evaluations"] <= 80


@pytest.mark.parametrize("solver", [refine_joint_plan, norm_refine_joint_plan, feasible_refine_joint_plan])
def test_weak_face_returns_original_seed_without_claiming_smooth_optimization(solver):
    model, interface = problem()
    q = np.zeros(7)
    q[6] = 1.0
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((4, 6))
    seed[:, 0], seed[:, 3] = 0.05, 1e-9
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True)
    result = solver(
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
    np.testing.assert_array_equal(result["planned_commands"], seed)
    assert result["optimizer_success"] is False
    assert result["optimizer_message"] == "weak_face_seed"


def test_final_backtracking_rejects_infeasible_endpoint_without_loosening_pose_guard(monkeypatch):
    import s2a.joint_limit_preview_feasible_inverse as implementation

    model, interface = problem()
    seed = np.array([[0, 0, 0, 0.1, 0, 0]])
    target_command = np.array([[0.1, 0, 0, 0.05, 0, 0]])
    target = rollout_joint_preview(
        model, np.zeros(7), target_command, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True
    )
    outside = np.array([[0.1, 0, 0, -1e-7, 0, 0]])
    monkeypatch.setattr(
        implementation,
        "minimize",
        lambda *a, **kw: SimpleNamespace(x=outside.ravel(), success=False, message="Iteration limit reached", nit=20),
    )
    result = feasible_refine_joint_plan(
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
    assert result["accepted"] is True
    assert 0 < result["feasibility_step"] < 1
    assert result["objective"] < result["seed_objective"]
    initial = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=True)
    a = pose_error(target["pose"][0], initial["pose"][0])
    b = pose_error(target["pose"][0], result["rollout"]["pose"][0])
    assert np.linalg.norm(b[:3]) <= np.linalg.norm(a[:3]) + 1e-10
    assert np.linalg.norm(b[3:]) <= np.linalg.norm(a[3:]) + 1e-10
