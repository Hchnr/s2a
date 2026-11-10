"""The independent checker must reject a false support certificate."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.tracking.stopped_pose_safety_bound import (
    error_directions,
    first_pose_linearization,
    safety_scopes,
    weighted_pose_bounds,
)
from s2a.research.trajectory_search.feasible_start_restoration import linear_domain
from scripts.verification.reference_ports.check_stopped_pose_safety_bound import (
    check_candidate,
    check_support_certificate,
)
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs


def test_dual_checker_accepts_valid_bound_and_rejects_understatement():
    scope = dict(lower=np.zeros(2), upper=np.ones(2), seed=np.zeros(2), a=np.ones((1, 2)), b=np.ones(1))
    certificate = dict(
        primal=np.array([0.0, 1.0]),
        primal_support=2.0,
        support_upper=2.0,
        dual_indices=np.array([0]),
        dual_values=np.array([2.0]),
        dual_residual=np.array([-1.0, 0.0]),
    )
    assert check_support_certificate(np.array([1.0, 2.0]), scope, certificate)["verified"]
    certificate["support_upper"] = 1.9
    with pytest.raises(ValueError, match="does not bound"):
        check_support_certificate(np.array([1.0, 2.0]), scope, certificate)


def test_scalar_response_and_duals_validate_a_complete_fixed_model_example():
    model, velocity, seed, previous, interface, site, tool, _targets, _twists, lower, upper = inputs()
    rollout = rollout_joint_preview(
        model,
        velocity,
        np.asarray([interface.decode(command) for command in seed]),
        site,
        lower,
        upper,
        enforce_limits=False,
    )
    target = rollout["pose"][0] @ tool
    target[:3, 3] += [0.012, 0.003, -0.002]
    target[:3, :3] = Rotation.from_rotvec([0.02, -0.01, 0.03]).as_matrix() @ target[:3, :3]
    error = pose_error(target, rollout["pose"][0] @ tool)
    domain = linear_domain(model, seed, previous, interface, lower, upper, np.zeros((25, 7)), rollout, np.empty((0, 7)))
    scopes = safety_scopes(domain, seed)
    scale = (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min)
    seed_q = rollout["joint_position"][0]
    command_jac = rollout["joint_position_jacobian"][0, :, :6] * scale
    linear = first_pose_linearization(
        site,
        tool,
        model["jacobian"],
        seed_q - model["q"],
        command_jac,
        seed[0],
        scopes["box"]["lower"],
        scopes["box"]["upper"],
        target,
    )
    n, d = error_directions(target[:3, :3], error)
    bounds = weighted_pose_bounds(linear, n, d, scopes, [0, 0.5, 1])
    checked = check_candidate(
        model,
        velocity,
        interface,
        site,
        tool,
        target,
        lower,
        upper,
        seed_q,
        command_jac,
        seed,
        error,
        linear,
        scopes,
        bounds,
    )
    assert checked["passed"] and checked["support_certificates_verified"] == 9
    assert checked["candidate_membership"]["box"]["within_relaxation"]
    assert checked["maximum_joint_affine_difference_rad"] < 1e-10
