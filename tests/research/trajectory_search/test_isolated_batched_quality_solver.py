"""Private solver binding cannot mutate frozen code, globals or model inputs."""

import numpy as np
import pytest

from s2a.controllers.osc import OSCInterface
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness
from s2a.research.trajectory_search.isolated_batched_quality_solver import IsolatedBatchedQualitySolver
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan


def inputs():
    q, velocity = np.zeros(7), np.zeros(7)
    mass, jacobian = np.eye(7), np.c_[np.eye(6), np.zeros(6)]
    kp, kd = np.full(6, 100.0), np.full(6, 20.0)
    model = build_joint_preview(mass, jacobian, kp, kd, q, q, null_kp=10, dt=0.002)
    seed = np.zeros((8, 6))
    seed[:, 0], seed[:, 3] = 0.01, 1e-6
    lower, upper = -np.ones(7) * 2, np.ones(7) * 2
    target = rollout_joint_preview(model, velocity, seed, np.eye(4), lower, upper, enforce_limits=False)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    args = (
        model,
        velocity,
        seed,
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        target["pose"],
        target["twist"],
        lower,
        upper,
    )
    return args, mass, jacobian, kp, kd, q


def test_isolated_binding_preserves_original_code_and_passes_independent_witness_audit():
    args, mass, jacobian, kp, kd, q = inputs()
    original_global = solve_interior_plan.__globals__["rollout_joint_preview"]
    candidate = IsolatedBatchedQualitySolver(args[0])
    assert candidate._solver.__code__ is solve_interior_plan.__code__
    assert candidate._solver.__globals__ is not solve_interior_plan.__globals__
    result = candidate.solve(*args)
    assert solve_interior_plan.__globals__["rollout_joint_preview"] is original_global is rollout_joint_preview
    assert result["witness_found"]
    audit = independently_check_witness(
        result,
        mass,
        jacobian,
        kp,
        kd,
        q,
        args[1],
        q,
        args[4],
        args[3],
        args[5],
        args[6],
        args[7][0],
        args[9],
        args[10],
        past_inner_error_rad=np.zeros((25, 7)),
    )
    assert audit["passed"] and audit["first_pose_caps_satisfied"]


def test_model_mutation_or_solver_error_never_changes_original_global_binding():
    args, *_ = inputs()
    candidate = IsolatedBatchedQualitySolver(args[0])
    args[0]["q"][0] += 0.1
    with pytest.raises(ValueError, match="unchanged frozen model"):
        candidate.solve(*args)
    assert solve_interior_plan.__globals__["rollout_joint_preview"] is rollout_joint_preview
    args, *_ = inputs()
    candidate = IsolatedBatchedQualitySolver(args[0])
    invalid = list(args)
    invalid[2] = np.zeros((2, 5))
    with pytest.raises(ValueError):
        candidate.solve(*invalid)
    assert solve_interior_plan.__globals__["rollout_joint_preview"] is rollout_joint_preview
