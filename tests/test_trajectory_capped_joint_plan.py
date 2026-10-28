"""Later predicted tracking failures cannot hide behind an accurate first period."""

import numpy as np
from test_configurable_tracking_terminal_plan import inputs

from s2a.feasibility_first_joint_plan import solve_interior_plan as first_only
from s2a.geometry import pose_error
from s2a.interior_joint_preview import rollout_joint_preview
from s2a.past_inner_error_joint_plan import solve_interior_plan as joint_solve
from s2a.trajectory_capped_joint_plan import solve_interior_plan


def feasible_inputs():
    args = list(inputs())
    joint = joint_solve(*args, past_inner_error_rad=np.full((25, 7), 0.003))
    args[7] = joint["rollout"]["pose"].copy()
    args[8] = joint["rollout"]["twist"].copy()
    return args, joint["planned_commands"]


def test_every_endpoint_and_inner_joint_bound_is_checked_on_returned_commands():
    args, warm = feasible_inputs()
    result = solve_interior_plan(*args, initial_guess=warm, past_inner_error_rad=np.full((25, 7), 0.003))
    assert result["witness_found"] and result["all_pose_caps_enforced"]
    assert result["response_evaluations"] <= 160 and result["iterations"] <= 60
    commands = result["planned_commands"]
    rollout = rollout_joint_preview(
        args[0],
        args[1],
        np.asarray([args[4].decode(c) for c in commands]),
        args[5],
        args[9],
        args[10],
        enforce_limits=False,
    )
    errors = np.asarray(
        [pose_error(target, pose @ args[6]) for target, pose in zip(args[7], rollout["pose"], strict=True)]
    )
    np.testing.assert_allclose(errors, result["all_tcp_errors"], atol=1e-12, rtol=0)
    assert np.all(np.linalg.norm(errors[:, :3], axis=1) <= 0.01 + 1e-10)
    assert np.all(np.linalg.norm(errors[:, 3:], axis=1) <= 0.05 + 1e-10)
    q, margin = rollout["inner_joint_position"], result["requested_position_margin_rad"]
    assert np.min(np.minimum(q - args[9] - margin, args[10] - margin - q)) >= -1e-10
    braking = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
    assert np.min(np.minimum(braking - args[9], args[10] - braking)) >= 0.05 - 1e-10
    assert np.all(result["terminal_normals"] @ rollout["joint_velocity"][-1] <= 1e-10)
    assert np.max(abs(commands)) <= 1 + 1e-12
    assert np.max(abs(np.diff(np.vstack((args[3], commands)), axis=0))) <= 0.25 + 1e-12


def test_accurate_first_point_does_not_qualify_an_unreachable_later_point():
    args, warm = feasible_inputs()
    args[7][-1, 0, 3] += 1
    original = first_only(*args, initial_guess=warm)
    assert original["witness_found"]
    result = solve_interior_plan(*args, initial_guess=warm)
    assert not result["witness_found"]
    assert result["status"] == "nonlinear_search_no_witness"
    assert result["response_evaluations"] <= 160
    closest = result["closest_linear_feasible_attempt"]
    assert np.linalg.norm(closest["all_tcp_errors"][-1, :3]) > 0.01
