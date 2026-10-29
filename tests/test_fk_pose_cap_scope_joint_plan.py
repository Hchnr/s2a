"""Future model accuracy is a separate policy from actual first-period accuracy."""

import numpy as np
from test_fk_consistent_trajectory_joint_plan import setup

from s2a.fk_consistent_trajectory_joint_plan import solve_interior_plan as all_caps
from s2a.fk_pose_cap_scope_joint_plan import solve_interior_plan


def test_default_full_scope_preserves_prior_result():
    args, fk = setup()
    old = all_caps(*args, fk_jacobian=fk, initial_guess=args[2])
    new = solve_interior_plan(*args, fk_jacobian=fk, initial_guess=args[2])
    assert new["pose_caps_horizon"] == 8 and new["all_pose_caps_enforced"]
    np.testing.assert_array_equal(old["planned_commands"], new["planned_commands"])
    assert old["objective"] == new["objective"]


def test_first_scope_accepts_only_first_pose_with_all_joint_constraints_retained():
    args, fk = setup()
    args[7] = args[7].copy()
    args[7][-1, 0, 3] += 1
    first = solve_interior_plan(*args, fk_jacobian=fk, initial_guess=args[2], pose_cap_periods=1)
    full = solve_interior_plan(*args, fk_jacobian=fk, initial_guess=args[2])
    assert first["witness_found"] and not full["witness_found"]
    assert not first["all_pose_caps_enforced"] and first["pose_caps_horizon"] == 1
    assert np.linalg.norm(first["all_tcp_errors"][0, :3]) <= 0.01 + 1e-10
    assert np.linalg.norm(first["all_tcp_errors"][0, 3:]) <= 0.05 + 1e-10
    assert np.linalg.norm(first["all_tcp_errors"][-1, :3]) > 0.01
    raw = first["rollout"]
    q, margin = raw["inner_joint_position"], first["requested_position_margin_rad"]
    assert np.min(np.minimum(q - args[9] - margin, args[10] - margin - q)) >= -1e-10
    braking = raw["joint_position"] + 0.1 * raw["joint_velocity"]
    assert np.min(np.minimum(braking - args[9], args[10] - braking)) >= 0.05 - 1e-10
    assert np.all(first["terminal_normals"] @ raw["joint_velocity"][-1] <= 1e-10)
    commands = first["planned_commands"]
    assert np.max(abs(commands)) <= 1 + 1e-12
    assert np.max(abs(np.diff(np.vstack((args[3], commands)), axis=0))) <= 0.25 + 1e-12
    assert first["response_evaluations"] <= 160 and first["iterations"] <= 60
