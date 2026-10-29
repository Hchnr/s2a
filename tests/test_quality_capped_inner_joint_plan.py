"""A joint-feasible plan must also meet the declared first-period TCP accuracy."""

import numpy as np
from test_configurable_tracking_terminal_plan import inputs

from s2a.past_inner_error_joint_plan import solve_interior_plan as joint_solve
from s2a.quality_capped_inner_joint_plan import solve_interior_plan


def test_witness_retains_joint_and_absolute_tracking_constraints():
    args = list(inputs())
    error = np.full((25, 7), 0.003)
    witness = joint_solve(*args, past_inner_error_rad=error)
    args[7] = witness["rollout"]["pose"].copy()
    args[8] = witness["rollout"]["twist"].copy()
    result = solve_interior_plan(*args, initial_guess=witness["planned_commands"], past_inner_error_rad=error)
    assert result["witness_found"]
    assert result["first_pose_caps_enforced"]
    assert result["minimum_tightened_inner_slack_rad"] >= -1e-10
    assert np.linalg.norm(result["first_tcp_error"][:3]) <= 0.01 + 1e-10
    assert np.linalg.norm(result["first_tcp_error"][3:]) <= 0.05 + 1e-10


def test_joint_feasibility_alone_cannot_accept_a_far_first_target():
    args = list(inputs())
    args[7] = args[7].copy()
    args[7][:, 0, 3] += 1.0
    result = solve_interior_plan(*args)
    assert result["linear_status"] == 0
    assert not result["witness_found"]
    assert result["status"] == "nonlinear_search_no_witness"
