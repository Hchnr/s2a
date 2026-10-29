"""Absolute tracking and joint constraints remain decisive in both search phases."""

import numpy as np
from test_configurable_tracking_terminal_plan import inputs

from s2a.feasibility_first_joint_plan import solve_interior_plan
from s2a.interior_joint_preview import rollout_joint_preview
from s2a.past_inner_error_joint_plan import solve_interior_plan as joint_solve


def test_two_phase_witness_survives_and_replays_original_constraints():
    args = list(inputs())
    profile = np.full((25, 7), 0.003)
    joint = joint_solve(*args, past_inner_error_rad=profile)
    args[7] = joint["rollout"]["pose"].copy()
    args[8] = joint["rollout"]["twist"].copy()
    targets_before = args[7].copy()
    result = solve_interior_plan(*args, initial_guess=joint["planned_commands"], past_inner_error_rad=profile)
    assert result["witness_found"] and result["first_pose_caps_enforced"]
    assert result["response_evaluations"] <= 160 and result["iterations"] <= 60
    assert result["search_phases"][0]["phase"] == "pose_feasibility"
    np.testing.assert_array_equal(args[7], targets_before)
    commands = result["planned_commands"]
    assert np.max(abs(commands)) <= 1 + 1e-12
    assert np.max(abs(np.diff(np.vstack((args[3], commands)), axis=0))) <= 0.25 + 1e-12
    rollout = rollout_joint_preview(
        args[0],
        args[1],
        np.asarray([args[4].decode(c) for c in commands]),
        args[5],
        args[9],
        args[10],
        enforce_limits=False,
    )
    q = rollout["inner_joint_position"]
    margin = np.asarray(result["requested_position_margin_rad"])
    assert np.min(np.minimum(q - args[9] - margin, args[10] - margin - q)) >= -1e-10
    braking = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
    assert np.min(np.minimum(braking - args[9], args[10] - braking)) >= 0.05 - 1e-10
    assert np.all(result["terminal_normals"] @ rollout["joint_velocity"][-1] <= 1e-10)
    assert np.linalg.norm(result["first_tcp_error"][:3]) <= 0.01 + 1e-10
    assert np.linalg.norm(result["first_tcp_error"][3:]) <= 0.05 + 1e-10


def test_unreachable_target_is_not_accepted_as_a_joint_only_witness():
    args = list(inputs())
    args[7] = args[7].copy()
    args[7][:, 0, 3] += 1
    result = solve_interior_plan(*args)
    assert result["linear_status"] == 0
    assert not result["witness_found"]
    assert result["status"] == "nonlinear_search_no_witness"
    assert result["response_evaluations"] <= 160 and result["iterations"] <= 60
    closest = result["closest_linear_feasible_attempt"]
    assert closest is not None
    assert np.linalg.norm(closest["first_tcp_error"][:3]) > 0.01
    assert closest["maximum_normalized_pose_cap_violation"] > 0
