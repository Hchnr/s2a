"""Do not dispatch an eight-point plan audited against another reference."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.preview.full_horizon_feedback import FullHorizonSolverAudit
from tests.research.trajectory_search.test_full_horizon_quality import fixture


def solve_and_audit_inputs():
    args, source = fixture()
    model, velocity, _seed, previous, interface, site, tool, targets, _twists, lower, upper = args
    mass, jacobian, kp, kd, q, _ = source
    audit_args = (mass, jacobian, kp, kd, q, velocity, q, interface, previous, site, tool, targets[0], lower, upper)
    return args, audit_args, targets


def test_full_horizon_solver_audits_every_target_with_bound_reference():
    args, audit_args, targets = solve_and_audit_inputs()
    adapter = FullHorizonSolverAudit()
    witness = adapter.solve(*args)
    assert witness["witness_found"] is True
    targets[-1, 0, 3] += 1.0
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["pose_horizon"] == 8 and checked["all_pose_caps_satisfied"]
    with pytest.raises(ValueError, match="current solver witness"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))


def test_changed_eighth_pose_or_other_witness_is_rejected():
    args, audit_args, _targets = solve_and_audit_inputs()
    adapter = FullHorizonSolverAudit()
    witness = adapter.solve(*args)
    with pytest.raises(ValueError, match="current solver witness"):
        adapter.audit(copy.deepcopy(witness), *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    witness["tcp_errors"][-1, 0] += 0.1
    with pytest.raises(AssertionError):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    with pytest.raises(ValueError, match="current solver witness"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))


def test_wrong_first_target_and_overlapping_solves_are_rejected():
    args, audit_args, _targets = solve_and_audit_inputs()
    adapter = FullHorizonSolverAudit()
    witness = adapter.solve(*args)
    with pytest.raises(ValueError, match="previous quality witness"):
        adapter.solve(*args)
    changed = list(audit_args)
    changed[11] = changed[11].copy()
    changed[11][0, 3] += 0.01
    with pytest.raises(ValueError, match="first target"):
        adapter.audit(witness, *changed, past_inner_error_rad=np.zeros((25, 7)))


def test_wrong_horizon_is_rejected_before_the_solver_runs():
    args, _audit_args, _targets = solve_and_audit_inputs()
    calls = []
    adapter = FullHorizonSolverAudit(solver=lambda *_a, **_kw: calls.append(True))
    changed = list(args)
    changed[7] = changed[7][:7]
    with pytest.raises(ValueError):
        adapter.solve(*changed)
    assert calls == []
