"""Linear-infeasible warm plans do not bypass a known feasible search start."""

import numpy as np
from test_configurable_tracking_terminal_plan import inputs
from test_trajectory_capped_joint_plan import feasible_inputs

from s2a.feasible_start_trajectory_joint_plan import solve_interior_plan
from s2a.long_trajectory_capped_joint_plan import solve_interior_plan as prior_solve


def test_verified_warm_plan_is_retained_without_changing_prior_result():
    args, warm = feasible_inputs()
    prior = prior_solve(*args, initial_guess=warm)
    result = solve_interior_plan(*args, initial_guess=warm)
    assert result["provided_start_linear_feasible"]
    assert result["selected_start_linear_feasible"]
    assert result["nonlinear_start_policy"] == "provided_start"
    np.testing.assert_array_equal(result["planned_commands"], prior["planned_commands"])


def test_joint_violating_warm_start_uses_checked_lp_before_nonlinear_search():
    args = list(inputs())
    result = solve_interior_plan(*args, initial_guess=args[2])
    assert result["linear_status"] == 0
    assert not result["provided_start_linear_feasible"]
    assert result["selected_start_linear_feasible"]
    assert result["nonlinear_start_policy"] == "verified_linear_witness"
    assert result["closest_linear_feasible_attempt"] is not None
    assert result["response_evaluations"] <= 160 and result["iterations"] <= 60
    if result["witness_found"]:
        assert np.all(np.linalg.norm(result["all_tcp_errors"][:, :3], axis=1) <= 0.01 + 1e-10)
        assert np.all(np.linalg.norm(result["all_tcp_errors"][:, 3:], axis=1) <= 0.05 + 1e-10)
        assert result["minimum_tightened_inner_slack_rad"] >= -1e-10
