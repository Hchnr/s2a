"""Independent joint and TCP verification of every predicted control period."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, pose_error
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness


def audit_full_horizon_witness(
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
    enforce_pose_caps=True,
    enforce_braking=True,
):
    """Independently verify joints then reconstruct every TCP from joint endpoints."""
    if type(enforce_pose_caps) is not bool:
        raise ValueError("Explicit boolean pose-cap enforcement required")
    if type(enforce_braking) is not bool:
        raise ValueError("Explicit boolean extra-braking enforcement required")
    commands = finite(witness["planned_commands"])
    if commands.ndim != 2 or commands.shape[1] != 6 or not 1 <= len(commands) <= 8:
        raise ValueError("One bounded generated command plan required")
    targets = finite(targets, (len(commands), 4, 4))
    result = independently_check_witness(
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
        targets[0],
        lower,
        upper,
        past_inner_error_rad=past_inner_error_rad,
        enforce_pose_caps=enforce_pose_caps,
        enforce_braking=enforce_braking,
    )
    pose, prior_q = finite(site, (4, 4)).copy(), finite(q, (7,)).copy()
    jacobian = finite(jacobian, (6, 7))
    endpoints = finite(witness["rollout"]["joint_position"], (len(commands), 7))
    reconstructed, errors = [], []
    for target, endpoint in zip(targets, endpoints, strict=True):
        movement = jacobian @ (endpoint - prior_q)
        pose[:3, 3] += movement[:3]
        pose[:3, :3] = Rotation.from_rotvec(movement[3:]).as_matrix() @ pose[:3, :3]
        reconstructed.append(pose.copy())
        errors.append(pose_error(target, pose @ site_from_tcp))
        prior_q = endpoint
    reconstructed, errors = np.asarray(reconstructed), np.asarray(errors)
    np.testing.assert_allclose(reconstructed, witness["rollout"]["pose"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(errors, witness["tcp_errors"], atol=1e-10, rtol=0)
    pose_ok = bool(
        np.all(np.linalg.norm(errors[:, :3], axis=1) <= 0.01 + 1e-10)
        and np.all(np.linalg.norm(errors[:, 3:], axis=1) <= 0.05 + 1e-10)
    )
    if enforce_pose_caps and not pose_ok:
        raise ValueError("All predicted TCP points must satisfy the original quality caps")
    result.update(
        tcp_errors=errors,
        pose_horizon=len(commands),
        all_pose_caps_satisfied=pose_ok,
        all_pose_caps_enforced=enforce_pose_caps,
        pose_reconstruction_max_difference=float(np.max(abs(reconstructed - witness["rollout"]["pose"]))),
    )
    return result
