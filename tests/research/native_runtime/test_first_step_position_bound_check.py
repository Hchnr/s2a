"""The corner checker independently rejects corrupted bound or response data."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.tracking.first_step_position_bound import first_step_position_bound
from scripts.verification.reference_ports.check_first_step_position_bound import check_corners
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs


def fixture():
    model, velocity, plan, previous, interface, site, tool, targets, _twists, lower, upper = inputs()
    physical = np.asarray([interface.decode(command) for command in plan])
    seed_rollout = rollout_joint_preview(model, velocity, physical, site, lower, upper, enforce_limits=False)
    seed_q = seed_rollout["joint_position"][0]
    scale = (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min)
    derivative = seed_rollout["joint_position_jacobian"][0, :, :6] * scale
    target = targets[0, :3, 3] + [0.3, 0, 0]
    bound = first_step_position_bound(
        site,
        tool,
        model["jacobian"],
        seed_q - model["q"],
        derivative,
        plan[0],
        previous,
        interface.input_min,
        interface.input_max,
        target,
        np.vstack((np.eye(3), -np.eye(3))),
    )
    args = (
        model,
        velocity,
        interface,
        site,
        tool,
        target,
        plan[0],
        seed_q,
        derivative,
        bound["command_lower"],
        bound["command_upper"],
        lower,
        upper,
        bound["lower_bound_m"],
    )
    return args, bound


def test_scalar_corner_check_agrees_with_relaxed_affine_box():
    args, _ = fixture()
    result = check_corners(*args)
    assert result["passed"] and result["corner_responses"] == 64 and result["transition_steps"] == 1600
    assert result["maximum_joint_affine_difference_rad"] < 1e-10


def test_checker_rejects_inflated_bound_and_changed_affine_response():
    args, _ = fixture()
    bad_bound = list(args)
    bad_bound[-1] = 100.0
    with pytest.raises(ValueError, match="violates"):
        check_corners(*bad_bound)
    bad_derivative = list(args)
    bad_derivative[8] = args[8].copy()
    bad_derivative[8][0, 0] += 0.01
    with pytest.raises(ValueError, match="affine"):
        check_corners(*bad_derivative)
