"""Future recovery slack must never relax the first generated target."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.restoration_horizon_feedback import RestorationHorizonSolverAudit
from s2a.research.trajectory_search.first_hard_restoration import (
    solve_first_hard_restoration,
    split_quality_constraints,
)
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_slack_changes_only_the_later_seven_position_and_rotation_constraints():
    values = np.linspace(-0.5, 0.5, 16)
    jacobian = np.arange(48 * 16).reshape(16, 48)
    first, first_jac, future, future_jac = split_quality_constraints(values, jacobian, 0.3)
    np.testing.assert_array_equal(first, values[:2])
    np.testing.assert_array_equal(first_jac[:, :-1], jacobian[:2])
    np.testing.assert_array_equal(first_jac[:, -1], 0)
    np.testing.assert_allclose(future, values[2:] + 0.3)
    np.testing.assert_array_equal(future_jac[:, -1], 1)
    with pytest.raises(ValueError, match="nonnegative"):
        split_quality_constraints(values, jacobian, -0.1)


def test_old_minimax_rejects_first_pose_but_hard_first_keeps_auditable_candidate():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.3
    old = RestorationHorizonSolverAudit()
    old_witness = old.solve(*args)
    with pytest.raises(ValueError, match="first pose"):
        old.audit(old_witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    new = RestorationHorizonSolverAudit(solver=solve_first_hard_restoration)
    witness = new.solve(*args)
    assert witness["witness_found"] and witness["first_pose_optimized_hard"]
    check = new.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert check["passed"] and check["first_pose_caps_satisfied"] and check["braking_constraint_enforced"]
    assert not check["all_pose_caps_enforced"] and check["quality_slack"] > 0


def test_first_hard_candidate_cannot_disable_braking_or_use_bad_seed():
    args, _, _ = solve_and_audit_inputs()
    with pytest.raises(ValueError, match="braking"):
        solve_first_hard_restoration(*args, enforce_braking=False)
    changed = list(args)
    changed[2] = changed[2].copy()
    changed[2][0, 0] = 2
    with pytest.raises(ValueError, match="box and rate"):
        solve_first_hard_restoration(*changed)
