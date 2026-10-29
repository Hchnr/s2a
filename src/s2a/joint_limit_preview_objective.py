"""Exact TCP pose/twist residual and analytic physical-input derivatives."""

import numpy as np

from s2a.full_coupled_pose_inverse import hat, right_inverse_jacobian
from s2a.geometry import finite, pose_error, validate_transform


def preview_objective(rollout, site_from_tcp, desired_tcp, desired_twists):
    """Use only provided generated targets; velocities are world TCP twists.

    The rollout's frozen joint model has no completed-innovation correction in
    this prototype. That omission is explicit, not borrowed from another model.
    """
    site_from_tcp, desired_tcp = validate_transform(site_from_tcp), validate_transform(desired_tcp)
    poses, twists = np.asarray(rollout["pose"]), np.asarray(rollout["twist"])
    horizon = len(poses)
    if site_from_tcp.shape != (4, 4) or desired_tcp.shape != (horizon, 4, 4):
        raise ValueError("One TCP transform and aligned generated target poses required")
    desired_twists = finite(desired_twists, (horizon, 6))
    residuals, jacobians = [], []
    weights = np.array([100.0] * 3 + [20.0] * 3 + [5.0] * 3 + [1.0] * 3)
    for k in range(horizon):
        pose, twist = poses[k], twists[k]
        pjac, rjac, vjac = (
            np.asarray(rollout[key])[k] for key in ("position_jacobian", "rotation_jacobian", "twist_jacobian")
        )
        offset = pose[:3, :3] @ site_from_tcp[:3, 3]
        tcp = pose @ site_from_tcp
        tcp_twist = np.r_[twist[:3] + np.cross(twist[3:], offset), twist[3:]]
        error = pose_error(desired_tcp[k], tcp)
        position_jac = pjac - hat(offset) @ rjac
        rotation_jac = -right_inverse_jacobian(error[3:]) @ rjac
        tcp_velocity_jac = vjac[:3] - hat(offset) @ vjac[3:] - hat(twist[3:]) @ hat(offset) @ rjac
        residual = np.r_[error, desired_twists[k] - tcp_twist]
        derivative = np.vstack((-position_jac, rotation_jac, -tcp_velocity_jac, -vjac[3:]))
        residuals.append(weights * residual)
        jacobians.append(weights[:, None] * derivative)
    residual, jacobian = np.concatenate(residuals), np.vstack(jacobians)
    return dict(
        residual=residual,
        jacobian=jacobian,
        objective=0.5 * float(residual @ residual),
        gradient=jacobian.T @ residual,
        scope="frozen joint model, exact TCP residual; no innovation or native control qualification",
    )
