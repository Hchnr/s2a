"""Feasible initialization preserves original constraints and response costs."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.controllers.osc import OSCInterface
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.preview.restoration_horizon_feedback import RestorationHorizonSolverAudit
from s2a.research.trajectory_search import feasible_start_restoration as candidate
from s2a.research.trajectory_search import first_hard_restoration as original
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs as tight_inputs


def domain_for(args):
    model, velocity, seed, previous, interface, site, _tool, _targets, _twists, lower, upper = args
    rollout = rollout_joint_preview(
        model, velocity, [interface.decode(row) for row in seed], site, lower, upper, enforce_limits=False
    )
    return candidate.linear_domain(model, seed, previous, interface, lower, upper, np.zeros((25, 7)), rollout)


def test_already_safe_seed_preserves_original_result_and_global_binding():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.3
    assert candidate.domain_check(args[2], domain_for(args), args[4])["accepted"]
    original_rollout = original.solve_first_hard_restoration.__globals__["rollout_joint_preview"]
    baseline = original.solve_first_hard_restoration(*args)
    changed = candidate.solve_feasible_start_restoration(*args)
    assert changed["initialization"]["seed_linear_safe"] and not changed["initialization"]["warm_start_used"]
    assert changed["initialization"]["seed_rollout_reused"]
    assert changed["initialization"]["actual_model_rollouts"] == changed["response_evaluations"] <= 320
    assert (
        changed["status"] == baseline["status"] and changed["response_evaluations"] == baseline["response_evaluations"]
    )
    np.testing.assert_array_equal(
        changed["best_joint_safe"]["planned_commands"], baseline["best_joint_safe"]["planned_commands"]
    )
    assert original.solve_first_hard_restoration.__globals__["rollout_joint_preview"] is original_rollout


def test_tight_joint_seed_is_projected_and_original_audit_remains_mandatory():
    args = tight_inputs()
    domain = domain_for(args)
    assert not candidate.domain_check(args[2], domain, args[4])["accepted"]
    warm, info = candidate.nearest_linear_plan(args[2], domain, args[4])
    assert warm is not None and info["warm_start_used"] and 1 <= info["start_lp_attempts"] <= 2
    assert candidate.domain_check(warm, domain, args[4])["accepted"]
    solved = candidate.solve_feasible_start_restoration(*args)
    assert solved["initialization"]["warm_start_used"] and solved["initialization"]["seed_rollout_reused"]
    assert solved["initialization"]["actual_model_rollouts"] == solved["response_evaluations"] <= 320
    if solved.get("best_joint_safe"):
        adapter = RestorationHorizonSolverAudit(solver=candidate.solve_feasible_start_restoration)
        witness = adapter.solve(*args)
        model, velocity, _seed, previous, interface, site, tool, targets, _twists, lower, upper = args
        check = adapter.audit(
            witness,
            model["mass"],
            model["jacobian"],
            np.full(6, 100.0),
            np.full(6, 20.0),
            model["q"],
            velocity,
            model["q"],
            interface,
            previous,
            site,
            tool,
            targets[0],
            lower,
            upper,
            past_inner_error_rad=np.zeros((25, 7)),
        )
        assert check["passed"] and check["first_pose_caps_satisfied"] and check["braking_constraint_satisfied"]


def test_zero_latch_repair_anchors_every_period_and_rechecks_domain(monkeypatch):
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    seed = np.tile([0, 0, 0, 0.1, 0, 0], (8, 1))
    row = np.zeros((1, 48))
    row[0, 0] = 1
    domain = dict(
        lo=-np.ones(48),
        hi=np.ones(48),
        rate=np.eye(48) - np.eye(48, k=-6),
        center=np.zeros(48),
        joint_a=row,
        joint_lo=np.array([0.1]),
        joint_hi=np.array([0.2]),
        terminal_a=np.zeros((0, 48)),
        terminal_offset=np.zeros(0),
    )
    zero = seed.copy()
    zero[:, 3] = 0
    zero[0, 0] = 0.1
    repaired = seed.copy()
    repaired[0, 0] = 0.1
    calls = []

    def fake_linprog(*_args, **kwargs):
        calls.append(kwargs["bounds"])
        plan = zero if len(calls) == 1 else repaired
        return SimpleNamespace(status=0, message="synthetic", success=True, x=np.r_[plan.ravel(), np.zeros(48)])

    monkeypatch.setattr(candidate, "linprog", fake_linprog)
    plan, info = candidate.nearest_linear_plan(seed, domain, interface)
    assert info["latch_repair"] and info["start_lp_attempts"] == 2 and len(info["sign_anchors"]) == 8
    assert candidate.domain_check(plan, domain, interface)["accepted"]
    assert all(calls[1][6 * i + 3][0] > 0 for i in range(8))


def test_rejects_caller_warm_start_or_original_invalid_command():
    args, _audit_args, _targets = solve_and_audit_inputs()
    with pytest.raises(ValueError, match="internally chosen"):
        candidate.solve_feasible_start_restoration(*args, initial_guess=args[2])
    changed = list(args)
    changed[2] = changed[2].copy()
    changed[2][0, 0] = 2
    with pytest.raises(ValueError, match="box and rate"):
        candidate.solve_feasible_start_restoration(*changed)
