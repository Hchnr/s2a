"""A joint-feasible plan must also meet the declared first-period TCP accuracy."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.past_inner_error_joint_plan import solve_interior_plan as joint_solve
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs


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


def test_explicit_original_rollout_preserves_default_solver_and_invalid_hook_is_rejected():
    args = list(inputs())
    calls = []

    def same_rollout(*values, **kwargs):
        calls.append(kwargs["enforce_limits"])
        return rollout_joint_preview(*values, **kwargs)

    default = solve_interior_plan(*args)
    explicit = solve_interior_plan(*args, rollout_fn=same_rollout)
    assert calls and all(value is False for value in calls)
    assert explicit["status"] == default["status"]
    assert explicit["response_evaluations"] == default["response_evaluations"]
    assert explicit["witness_found"] == default["witness_found"]
    if default["witness_found"]:
        np.testing.assert_array_equal(explicit["planned_commands"], default["planned_commands"])
    with pytest.raises(ValueError, match="callable"):
        solve_interior_plan(*args, rollout_fn=1)
