"""Innovation constraints preserve SO(3), original gates, and response limits."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.controllers.osc import OSCInterface
from s2a.research.trajectory_search import innovation_capped_quality_solver as solver


def test_corrected_rotation_jacobian_matches_noncommuting_central_differences():
    target = Rotation.from_euler("xyz", [0.2, -0.4, 0.3])
    predicted = Rotation.from_euler("xyz", [-0.3, 0.15, 0.25])
    discrepancy = Rotation.from_euler("xyz", [0.1, 0.07, -0.08]).as_matrix()
    rotation_jac = np.array([[0.3, -0.2, 0.1], [0.1, 0.4, -0.3], [-0.2, 0.1, 0.5]])
    position_jac = np.array([[0.1, 0.2, 0], [-0.2, 0.1, 0.1], [0, -0.1, 0.2]])
    translation = np.array([0.002, -0.004, 0.003])

    def raw(x):
        pose = Rotation.from_rotvec(rotation_jac @ x) * predicted
        return np.r_[np.array([0.01, -0.02, 0.03]) + position_jac @ x, (target * pose.inv()).as_rotvec()]

    origin, epsilon = np.zeros(3), 1e-6
    raw_jac = np.column_stack(
        [
            (raw(origin + np.eye(3)[i] * epsilon) - raw(origin - np.eye(3)[i] * epsilon)) / (2 * epsilon)
            for i in range(3)
        ]
    )
    analytic = solver.corrected_error_jacobian(raw(origin), raw_jac, rotation_jac, translation, discrepancy)
    numeric = np.column_stack(
        [
            (
                solver.corrected_error(raw(origin + np.eye(3)[i] * epsilon), translation, discrepancy)
                - solver.corrected_error(raw(origin - np.eye(3)[i] * epsilon), translation, discrepancy)
            )
            / (2 * epsilon)
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-7)


def fixture():
    plan = np.zeros((8, 6))
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    poses = np.tile(np.eye(4), (8, 1, 1))
    rollout = {
        "pose": poses.copy(),
        "inner_joint_position": np.zeros((200, 7)),
        "joint_position": np.zeros((8, 7)),
        "joint_velocity": np.zeros((8, 7)),
    }
    original = {
        "witness_found": True,
        "status": "model_witness",
        "planned_commands": plan.copy(),
        "first_tcp_error": np.zeros(6),
        "rollout": rollout,
        "response_evaluations": 1,
        "terminal_normals": np.empty((0, 7)),
    }
    args = (
        object(),
        np.zeros(7),
        plan,
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        poses,
        np.zeros((8, 6)),
        -np.ones(7),
        np.ones(7),
    )
    return args, original, rollout


def run_with_cache(monkeypatch, args, original, rollout, plan, translation, *, count=1):
    cache = {}
    for index in range(count):
        row = plan.copy()
        row[-1, -1] = index * 1e-8
        cache[tuple(row.ravel())] = {"rollout": rollout, "score": {"objective": float(index)}}
    original["response_evaluations"] = count
    monkeypatch.setattr(solver, "_search_with_innovation", lambda *_args, **_kwargs: (original, {"cache": cache}))
    return solver.solve_interior_plan(
        *args,
        translation_innovation=translation,
        rotation_discrepancy=np.eye(3),
        past_inner_error_rad=np.zeros((25, 7)),
    )


def test_original_witness_is_not_accepted_if_innovation_gate_fails(monkeypatch):
    args, original, rollout = fixture()
    result = run_with_cache(monkeypatch, args, original, rollout, np.zeros((8, 6)), np.array([0.02, 0, 0]))
    assert result["original_solver_status"] == "model_witness"
    assert result["witness_found"] is False and "planned_commands" not in result
    assert result["strict_candidates"] == 0


def test_a_candidate_passing_both_original_and_innovation_gates_is_returned(monkeypatch):
    args, original, rollout = fixture()
    result = run_with_cache(monkeypatch, args, original, rollout, np.zeros((8, 6)), np.zeros(3))
    assert result["witness_found"] is True
    assert result["status"] == "innovation_model_witness"
    assert result["requires_independent_audit"] is True
    assert result["future_innovation_guarantee"] is False


def test_near_rate_candidate_requires_one_rechecked_model_response(monkeypatch):
    args, original, rollout = fixture()
    plan = np.zeros((8, 6))
    plan[0, 0] = 0.250000000006
    monkeypatch.setattr(solver.quality, "rollout_joint_preview", lambda *_args, **_kwargs: rollout)
    result = run_with_cache(monkeypatch, args, original, rollout, plan, np.zeros(3))
    assert result["witness_found"] is True
    assert result["response_evaluations"] == 2
    assert result["repair"]["accepted"] is True
    assert result["repair"]["maximum_command_change"] < 1e-8


def test_full_budget_prevents_an_extra_repair_response(monkeypatch):
    args, original, rollout = fixture()
    plan = np.zeros((8, 6))
    plan[0, 0] = 0.250000000006
    monkeypatch.setattr(
        solver.quality, "rollout_joint_preview", lambda *_args, **_kwargs: pytest.fail("extra response")
    )
    result = run_with_cache(monkeypatch, args, original, rollout, plan, np.zeros(3), count=160)
    assert result["witness_found"] is False
    assert result["response_evaluations"] == 160
