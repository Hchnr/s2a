"""Independently audit the first two pose points and original joint constraints."""

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness


def audit_two_step_witness(
    witness,
    mass,
    jacobian,
    kp,
    kd,
    q,
    velocity,
    initial_joint,
    interface,
    previous,
    site,
    site_from_tcp,
    targets,
    lower,
    upper,
    *,
    past_inner_error_rad,
):
    """Rebuild all eight points, enforcing the original caps on the first two."""
    if witness.get("hard_pose_prefix") != 2 or witness.get("prefix_selection_hard") is not True:
        raise ValueError("A selected two-point model witness is required")
    verified = audit_full_horizon_witness(
        witness,
        mass,
        jacobian,
        kp,
        kd,
        q,
        velocity,
        initial_joint,
        interface,
        previous,
        site,
        site_from_tcp,
        targets,
        lower,
        upper,
        past_inner_error_rad=past_inner_error_rad,
        enforce_pose_caps=False,
        enforce_braking=True,
    )
    errors = finite(verified["tcp_errors"], (8, 6))
    prefix_ok = bool(
        np.all(np.linalg.norm(errors[:2, :3], axis=1) <= 0.01 + 1e-10)
        and np.all(np.linalg.norm(errors[:2, 3:], axis=1) <= 0.05 + 1e-10)
    )
    if not prefix_ok or verified.get("braking_constraint_enforced") is not True:
        raise ValueError("Both first pose points and all original safety constraints must pass")
    future = max(
        0.0,
        float(
            np.max(
                np.r_[
                    np.linalg.norm(errors[2:, :3], axis=1) / 0.01 - 1,
                    np.linalg.norm(errors[2:, 3:], axis=1) / 0.05 - 1,
                ]
            )
        ),
    )
    if not np.isclose(future, witness["future_quality_slack"], atol=1e-8, rtol=0):
        raise ValueError("Independent six-point future quality gap differs")
    verified.update(
        hard_pose_prefix=2,
        prefix_pose_caps_satisfied=True,
        prefix_pose_caps_enforced=True,
        future_quality_slack=future,
        selection_scope="joint_safety_and_two_first_poses",
    )
    return verified
