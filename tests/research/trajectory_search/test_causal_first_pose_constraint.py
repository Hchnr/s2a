"""Causal first-pose constraints must guide search without weakening safety."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.joint_control.preview.restoration_horizon_feedback import RestorationHorizonSolverAudit
from s2a.research.trajectory_search.causal_first_pose_constraint import CausalFirstPoseConstraint
from s2a.research.trajectory_search.staged_first_feasibility import solve_staged_first_feasibility
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_corrected_constraint_gradient_matches_noncommuting_rotation_differences():
    target = Rotation.from_euler("xyz", [0.2, -0.4, 0.3])
    predicted = Rotation.from_euler("xyz", [-0.3, 0.15, 0.25])
    rotation_jac = np.array([[0.3, -0.2, 0.1], [0.1, 0.4, -0.3], [-0.2, 0.1, 0.5]])
    position_jac = np.array([[0.1, 0.2, 0], [-0.2, 0.1, 0.1], [0, -0.1, 0.2]])
    constraint = CausalFirstPoseConstraint(
        [0.002, -0.004, 0.003], Rotation.from_euler("xyz", [0.1, 0.07, -0.08]).as_matrix()
    )

    def raw(x):
        pose = Rotation.from_rotvec(rotation_jac @ x) * predicted
        return np.r_[np.array([0.01, -0.02, 0.03]) + position_jac @ x, (target * pose.inv()).as_rotvec()]

    step = 1e-6
    raw_jac = np.column_stack([(raw(np.eye(3)[i] * step) - raw(-np.eye(3)[i] * step)) / (2 * step) for i in range(3)])
    weights, scale = np.array([100.0] * 3 + [20.0] * 3), np.linspace(0.5, 1.5, 48)
    score = {"residual": np.zeros(96), "jacobian": np.zeros((96, 48))}
    score["residual"][:6] = raw(np.zeros(3)) * weights
    score["jacobian"][:6, :3] = raw_jac * weights[:, None] / scale[None, :3]
    rollout = {"rotation_jacobian": np.zeros((8, 3, 48))}
    rollout["rotation_jacobian"][0, :, :3] = rotation_jac / scale[None, :3]
    values, derivative = constraint.evaluate(score, rollout, raw(np.zeros(3)), scale)

    def corrected_values(x):
        error = constraint.correction(raw(x))
        return np.array([1 - np.linalg.norm(error[:3]) / 0.01, 1 - np.linalg.norm(error[3:]) / 0.05])

    numeric = np.column_stack(
        [
            (corrected_values(np.eye(3)[i] * step) - corrected_values(-np.eye(3)[i] * step)) / (2 * step)
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(values, corrected_values(np.zeros(3)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(derivative[:, :3], numeric, atol=1e-7, rtol=1e-7)


def test_staged_search_and_independent_audit_preserve_both_first_pose_gates():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.02
    constraint = CausalFirstPoseConstraint([0.001, -0.001, 0], np.eye(3), position_cap_m=0.009, rotation_cap_rad=0.045)

    def constrained(*args, **kwargs):
        return solve_staged_first_feasibility(*args, first_pose_constraint=constraint, **kwargs)

    adapter = RestorationHorizonSolverAudit(solver=constrained)
    witness = adapter.solve(*args)
    assert witness["witness_found"] is True
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["first_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    assert constraint.satisfied(checked["tcp_errors"][0])
    np.testing.assert_allclose(
        constraint.correction(checked["tcp_errors"][0]), witness["innovation_corrected_first_tcp_error"], atol=1e-10
    )


def test_impossible_completed_innovation_never_becomes_a_candidate():
    args, _audit_args, _targets = solve_and_audit_inputs()
    constraint = CausalFirstPoseConstraint([1, 0, 0], np.eye(3))
    result = solve_staged_first_feasibility(*args, first_pose_constraint=constraint)
    assert result["witness_found"] is False
    assert "best_joint_safe" not in result
    assert result["response_evaluations"] <= 320


def test_invalid_correction_cannot_weaken_original_pose_caps():
    with pytest.raises(ValueError, match="no larger than the original caps"):
        CausalFirstPoseConstraint(np.zeros(3), np.eye(3), position_cap_m=0.02)
    with pytest.raises(ValueError, match="proper world-frame SO\(3\)"):
        CausalFirstPoseConstraint(np.zeros(3), np.zeros((3, 3)))
