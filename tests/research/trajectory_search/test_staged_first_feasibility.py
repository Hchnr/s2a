"""Two-stage search must keep first-pose safety and one shared budget."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.restoration_horizon_feedback import RestorationHorizonSolverAudit
from s2a.research.trajectory_search import staged_first_feasibility as staged
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs as tight_inputs


def test_first_violation_gradient_matches_finite_differences():
    values = np.r_[-0.4, 0.2, np.ones(14)]
    jacobian = np.arange(16 * 4, dtype=float).reshape(16, 4) / 100
    value, gradient = staged.first_violation_objective(values, jacobian)
    assert value == pytest.approx(0.08)
    for index in range(4):
        shift = np.zeros(4)
        shift[index] = 1e-6
        plus = staged.first_violation_objective(values + jacobian @ shift, jacobian)[0]
        minus = staged.first_violation_objective(values - jacobian @ shift, jacobian)[0]
        np.testing.assert_allclose((plus - minus) / 2e-6, gradient[index], atol=1e-10, rtol=0)


def test_first_stage_finds_and_independently_audits_first_pose(monkeypatch):
    args, audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.02
    targets[-1, 0, 3] += 0.1
    calls = []
    real_rollout = staged.rollout_joint_preview

    def counted(*args, **kwargs):
        calls.append(1)
        return real_rollout(*args, **kwargs)

    monkeypatch.setattr(staged, "rollout_joint_preview", counted)
    raw = staged.solve_staged_first_feasibility(*args)
    assert raw["status"] == "staged_first_restoration_candidate" and raw["first_hard_evaluations"] > 0
    assert raw["search_phases"][0]["phase"] == "first_feasibility"
    assert raw["search_phases"][0]["iterations"] <= 40
    assert len(calls) == raw["response_evaluations"] == raw["initialization"]["actual_model_rollouts"] <= 320
    known = [p["iterations"] for p in raw["search_phases"] if p["iterations"] is not None]
    assert sum(known) <= 120
    adapter = RestorationHorizonSolverAudit(solver=staged.solve_staged_first_feasibility)
    witness = adapter.solve(*args)
    check = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert check["passed"] and check["first_pose_caps_satisfied"] and check["braking_constraint_satisfied"]
    assert not check["all_pose_caps_enforced"]


def test_inaccessible_first_target_never_becomes_a_dispatch_candidate():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 1.0
    result = staged.solve_staged_first_feasibility(*args)
    assert result["status"] == "staged_first_no_witness" and "best_joint_safe" not in result
    assert result["response_evaluations"] <= 320 and result["first_hard_evaluations"] == 0
    assert result["closest_first_rejected"]["first_normalized_violation"] > 0


def test_tight_joint_seed_uses_checked_initializer_within_budget():
    result = staged.solve_staged_first_feasibility(*tight_inputs())
    assert result["initialization"]["warm_start_used"]
    assert result["initialization"]["check"]["accepted"]
    assert 1 <= result["initialization"]["start_lp_attempts"] <= 2
    assert result["response_evaluations"] == result["initialization"]["actual_model_rollouts"] <= 320
