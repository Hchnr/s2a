"""Longer causal preview retains original accuracy, joint and budget contracts."""

import numpy as np
import pytest
from test_configurable_tracking_terminal_plan import inputs

from s2a.geometry import pose_error
from s2a.interior_joint_preview import rollout_joint_preview
from s2a.long_trajectory_capped_joint_plan import solve_interior_plan
from s2a.trajectory_capped_joint_plan import solve_interior_plan as short_solve


@pytest.mark.parametrize("horizon", [8, 12, 16])
def test_longer_horizon_is_checked_at_every_endpoint_and_inner_sample(horizon):
    args = list(inputs())
    seed = np.zeros((horizon, 6))
    seed[:, 3] = 1e-9
    args[2] = seed
    rollout = rollout_joint_preview(args[0], args[1], seed, args[5], args[9], args[10], enforce_limits=False)
    args[7], args[8] = rollout["pose"], rollout["twist"]
    result = solve_interior_plan(*args, initial_guess=seed)
    assert result["witness_found"] and result["pose_caps_horizon"] == horizon
    assert result["response_evaluations"] <= 160 and result["iterations"] <= 60
    commands = result["planned_commands"]
    actual = rollout_joint_preview(
        args[0],
        args[1],
        np.asarray([args[4].decode(c) for c in commands]),
        args[5],
        args[9],
        args[10],
        enforce_limits=False,
    )
    errors = np.asarray([pose_error(t, p) for t, p in zip(args[7], actual["pose"], strict=True)])
    np.testing.assert_allclose(errors, result["all_tcp_errors"], atol=1e-12, rtol=0)
    assert np.all(np.linalg.norm(errors[:, :3], axis=1) <= 0.01 + 1e-10)
    assert np.all(np.linalg.norm(errors[:, 3:], axis=1) <= 0.05 + 1e-10)
    q, margin = actual["inner_joint_position"], result["requested_position_margin_rad"]
    assert q.shape == margin.shape == (25 * horizon, 7)
    assert np.min(np.minimum(q - args[9] - margin, args[10] - margin - q)) >= -1e-10
    braking = actual["joint_position"] + 0.1 * actual["joint_velocity"]
    assert np.min(np.minimum(braking - args[9], args[10] - braking)) >= 0.05 - 1e-10
    assert np.max(abs(commands)) <= 1 + 1e-12
    assert np.max(abs(np.diff(np.vstack((args[3], commands)), axis=0))) <= 0.25 + 1e-12
    if horizon == 8:
        old = short_solve(*args, initial_guess=seed)
        np.testing.assert_array_equal(commands, old["planned_commands"])
        assert result["objective"] == old["objective"]


def test_horizon_above_registered_sixteen_is_rejected():
    args = list(inputs())
    args[2] = np.tile(args[2][-1], (17, 1))
    with pytest.raises(ValueError, match="sixteen"):
        solve_interior_plan(*args)
