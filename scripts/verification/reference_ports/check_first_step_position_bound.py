"""Independently check first-command box corners using scalar joint responses."""

from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform
from s2a.research.response_models.forward.joint_limit_response import predict_joint_response


def check_corners(
    model,
    velocity,
    interface,
    site,
    site_from_tcp,
    target_position,
    seed_command,
    seed_joint_position,
    joint_command_jacobian,
    command_lower,
    command_upper,
    joint_lower,
    joint_upper,
    claimed_lower_bound_m,
    *,
    tolerance=1e-10,
):
    """Audit all 64 corners; no continuous-box or physical claim is made."""
    site, site_from_tcp = validate_transform(site), validate_transform(site_from_tcp)
    q0 = finite(model["q"])
    n_joint = len(q0)
    velocity = finite(velocity, (n_joint,))
    target_position, seed_command = finite(target_position, (3,)), finite(seed_command, (6,))
    seed_joint_position = finite(seed_joint_position, (n_joint,))
    joint_command_jacobian = finite(joint_command_jacobian, (n_joint, 6))
    command_lower, command_upper = finite(command_lower, (6,)), finite(command_upper, (6,))
    joint_lower, joint_upper = finite(joint_lower, (n_joint,)), finite(joint_upper, (n_joint,))
    if (
        site.shape != (4, 4)
        or site_from_tcp.shape != (4, 4)
        or np.any(command_lower > command_upper)
        or np.any(command_lower < interface.input_min)
        or np.any(command_upper > interface.input_max)
        or not np.isfinite(claimed_lower_bound_m)
        or claimed_lower_bound_m < 0
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Valid fixed geometry, command box, and lower bound are required")
    jacobian = finite(model["jacobian"], (6, n_joint))
    offset = site[:3, :3] @ site_from_tcp[:3, 3]
    max_affine_difference, minimum_distance = 0.0, float("inf")
    count = 0
    for raw in product(*zip(command_lower, command_upper, strict=True)):
        command = np.asarray(raw)
        response = predict_joint_response(
            model, velocity, interface.decode(command), joint_lower, joint_upper, substeps=25, enforce_limits=False
        )
        actual_q = response["joint_position"]
        affine_q = seed_joint_position + joint_command_jacobian @ (command - seed_command)
        max_affine_difference = max(max_affine_difference, float(np.max(np.abs(actual_q - affine_q))))
        motion = jacobian @ (actual_q - q0)
        tcp = site[:3, 3] + motion[:3] + Rotation.from_rotvec(motion[3:]).apply(offset)
        minimum_distance = min(minimum_distance, float(np.linalg.norm(target_position - tcp)))
        count += 1
    if count != 64 or max_affine_difference > tolerance:
        raise ValueError("Independent corner response does not match the affine seed model")
    if minimum_distance + tolerance < claimed_lower_bound_m:
        raise ValueError("A corner violates the reported position lower bound")
    return dict(
        passed=True,
        corner_responses=count,
        transition_steps=25 * count,
        maximum_joint_affine_difference_rad=max_affine_difference,
        minimum_corner_position_error_m=minimum_distance,
        compared_lower_bound_m=float(claimed_lower_bound_m),
        scope="64 relaxed-box corners only; no physical or continuous-box qualification",
    )
