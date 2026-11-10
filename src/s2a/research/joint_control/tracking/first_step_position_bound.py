"""Conservative frozen-model position bound over a first-command box."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform
from s2a.research.joint_control.inverse.full_coupled_pose_inverse import hat, left_jacobian


def first_step_position_bound(
    site,
    site_from_tcp,
    jacobian,
    seed_joint_displacement,
    joint_command_jacobian,
    seed_command,
    previous_command,
    input_min,
    input_max,
    target_position,
    directions,
    *,
    rate_limit=0.25,
    numerical_allowance_m=1e-10,
):
    """Bound first-TCP distance in the free model over a relaxed command box.

    The joint displacement is affine in normalized command. Other joint,
    terminal, pose, and future-period constraints are deliberately omitted.
    This calculation uses ordinary floating point, not interval arithmetic.
    """
    site, site_from_tcp = validate_transform(site), validate_transform(site_from_tcp)
    if site.shape != (4, 4) or site_from_tcp.shape != (4, 4):
        raise ValueError("One current SITE and one fixed TCP transform are required")
    seed_joint_displacement = finite(seed_joint_displacement)
    if seed_joint_displacement.ndim != 1 or not len(seed_joint_displacement):
        raise ValueError("A nonempty seed joint displacement is required")
    n_joint = len(seed_joint_displacement)
    jacobian = finite(jacobian, (6, n_joint))
    joint_command_jacobian = finite(joint_command_jacobian, (n_joint, 6))
    seed_command, previous_command = finite(seed_command, (6,)), finite(previous_command, (6,))
    input_min, input_max = finite(input_min, (6,)), finite(input_max, (6,))
    target_position, directions = finite(target_position, (3,)), finite(directions)
    if directions.ndim != 2 or directions.shape[1] != 3 or not len(directions):
        raise ValueError("At least one three-dimensional direction is required")
    if (
        not np.isfinite(rate_limit)
        or rate_limit <= 0
        or not np.isfinite(numerical_allowance_m)
        or numerical_allowance_m < 0
        or np.any(input_min >= input_max)
    ):
        raise ValueError("Ordered input ranges and nonnegative finite allowances are required")
    lower = np.maximum(input_min, previous_command - rate_limit)
    upper = np.minimum(input_max, previous_command + rate_limit)
    if np.any(lower > upper) or np.any(seed_command < lower - 1e-12) or np.any(seed_command > upper + 1e-12):
        raise ValueError("Seed must lie inside the nonempty current command box")

    movement = jacobian @ seed_joint_displacement
    motion_jacobian = jacobian @ joint_command_jacobian
    offset = site[:3, :3] @ site_from_tcp[:3, 3]
    rotated_offset = Rotation.from_rotvec(movement[3:]).apply(offset)
    seed_position = site[:3, 3] + movement[:3] + rotated_offset
    position_jacobian = motion_jacobian[:3] - hat(rotated_offset) @ left_jacobian(movement[3:]) @ motion_jacobian[3:]
    lower_delta, upper_delta = lower - seed_command, upper - seed_command

    angular_low = np.minimum(motion_jacobian[3:] * lower_delta, motion_jacobian[3:] * upper_delta).sum(axis=1)
    angular_high = np.maximum(motion_jacobian[3:] * lower_delta, motion_jacobian[3:] * upper_delta).sum(axis=1)
    angular_component_bound = np.maximum(np.abs(angular_low), np.abs(angular_high))
    remainder_bound = 0.5 * float(np.linalg.norm(offset)) * float(angular_component_bound @ angular_component_bound)
    residual = target_position - seed_position
    rows = []
    for index, direction in enumerate(directions):
        length = float(np.linalg.norm(direction))
        if length <= 1e-15:
            rows.append(dict(index=index, skipped=True, reason="zero_direction"))
            continue
        unit = direction / length
        projected_jacobian = unit @ position_jacobian
        support = float(np.maximum(projected_jacobian * lower_delta, projected_jacobian * upper_delta).sum())
        raw_lower = float(unit @ residual - support - remainder_bound - numerical_allowance_m)
        rows.append(
            dict(
                index=index,
                skipped=False,
                unit_direction=unit,
                linear_support_m=support,
                projected_seed_error_m=float(unit @ residual),
                lower_bound_m=max(0.0, raw_lower),
            )
        )
    usable = [row["lower_bound_m"] for row in rows if not row["skipped"]]
    if not usable:
        raise ValueError("At least one nonzero direction is required")
    return dict(
        lower_bound_m=max(usable),
        seed_tcp_position=seed_position,
        seed_position_error_m=float(np.linalg.norm(residual)),
        position_command_jacobian=position_jacobian,
        angular_component_bound_rad=angular_component_bound,
        rotation_remainder_bound_m=remainder_bound,
        command_lower=lower,
        command_upper=upper,
        numerical_allowance_m=float(numerical_allowance_m),
        directions=rows,
        scope="floating-point outer bound for frozen free model and first-command box only",
    )
