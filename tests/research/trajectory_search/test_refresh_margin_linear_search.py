"""Exercise the fixed linearized constraints and the actual local-model check."""

import numpy as np
from scipy.optimize import linprog

from s2a.research.trajectory_search.margin_two_step_search import solve_margin_two_step
from s2a.research.trajectory_search.refresh_margin_linear_search import (
    TRUST_REGION,
    linearized_problem,
    solve_linearized_refresh,
)
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_linear_program_respects_tightened_second_pose_and_fixed_trust_region():
    identity = np.eye(48)
    domain = dict(
        lo=-np.ones(48),
        hi=np.ones(48),
        rate=identity,
        center=np.zeros(48),
        joint_a=identity,
        joint_lo=-np.ones(48),
        joint_hi=np.ones(48),
        terminal_a=np.zeros((0, 48)),
        terminal_offset=np.zeros(0),
    )
    values = np.r_[0.06, 0.07, 0.07, 0.10, np.zeros(12)]
    jacobian = np.zeros((16, 48))
    jacobian[2, 2] = 1.0
    problem = linearized_problem(np.zeros((8, 6)), domain, values, jacobian, 0.08)
    result = linprog(
        problem["objective"],
        A_ub=problem["matrix"],
        b_ub=problem["limits"],
        bounds=problem["bounds"],
        method="highs",
    )
    assert result.success
    np.testing.assert_allclose(result.x[2], 0.005, atol=1e-9, rtol=0)
    assert np.max(np.abs(result.x[:48])) <= TRUST_REGION
    assert np.max(problem["matrix"] @ result.x - problem["limits"]) <= 1e-9


def test_linearized_candidate_is_rechecked_with_the_local_model():
    args, _audit_args, _targets = solve_and_audit_inputs()
    baseline = solve_margin_two_step(*args)["best_joint_safe"]
    changed = list(args)
    changed[2] = baseline["planned_commands"]
    responses, programs = [], []
    result = solve_linearized_refresh(
        *changed,
        past_inner_error_rad=np.zeros((25, 7)),
        expected_seed_tcp_errors=baseline["tcp_errors"],
        expected_baseline_future_slack=baseline["future_quality_slack"],
        on_response=responses.append,
        on_lp=programs.append,
    )
    assert programs == ["attempted", "completed"]
    assert responses == ["attempted", "completed", "attempted", "completed"]
    assert result["optimizer"]["success"] and result["response_completions"] == 2
    assert result["candidate"] is not None and result["tested_candidate"]["eligible"] is True
    assert result["tested_candidate"]["linear_safety"]["accepted"] is True
