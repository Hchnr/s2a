"""Reconstruct site pose/twist and derivatives from predicted joint states."""

import numpy as np

from s2a.geometry import finite, validate_transform


def fk_consistent_rollout(rollout, fk_jacobian, *, jacobian_difference_step_rad=1e-6):
    """Replace frozen-J Cartesian integration with FK(q) and J(q) qdot.

    The joint response and all joint constraints remain unchanged. fk_jacobian
    returns one world SITE pose and its world linear/angular geometric Jacobian.
    dJ/dq for the twist pullback is a fixed central difference, not a future
    measurement. Virtual derivative queries can cross physical limits; none are
    commands. This does not correct frozen joint dynamics or model contacts.
    """
    step = float(jacobian_difference_step_rad)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Positive finite Jacobian difference step required")
    q = finite(rollout["joint_position"])
    if q.ndim != 2 or not all(q.shape):
        raise ValueError("Nonempty horizon-by-joint positions required")
    horizon, joints = q.shape
    velocity = finite(rollout["joint_velocity"], q.shape)
    qjac = finite(rollout["joint_position_jacobian"])
    if qjac.ndim != 3 or qjac.shape[:2] != q.shape or qjac.shape[2] < 1:
        raise ValueError("Aligned joint-position derivatives required")
    vjac = finite(rollout["joint_velocity_jacobian"], qjac.shape)
    poses, twists, pjac, rjac, tjac = [], [], [], [], []
    for period in range(horizon):
        pose, jac = fk_jacobian(q[period].copy())
        pose, jac = validate_transform(pose), finite(jac, (6, joints))
        if pose.shape != (4, 4):
            raise ValueError("One world SITE transform required per joint state")
        derivative = np.zeros((6, joints))
        if np.any(velocity[period] != 0):
            for joint in range(joints):
                plus, minus = q[period].copy(), q[period].copy()
                plus[joint] += step
                minus[joint] -= step
                jplus = finite(fk_jacobian(plus)[1], (6, joints))
                jminus = finite(fk_jacobian(minus)[1], (6, joints))
                derivative[:, joint] = ((jplus - jminus) @ velocity[period]) / (2 * step)
        poses.append(pose.copy())
        twists.append(jac @ velocity[period])
        pjac.append(jac[:3] @ qjac[period])
        rjac.append(jac[3:] @ qjac[period])
        tjac.append(jac @ vjac[period] + derivative @ qjac[period])
    result = dict(rollout)
    result.update(
        linearized_pose=np.asarray(rollout["pose"]).copy(),
        pose=np.asarray(poses),
        twist=np.asarray(twists),
        position_jacobian=np.asarray(pjac),
        rotation_jacobian=np.asarray(rjac),
        twist_jacobian=np.asarray(tjac),
        kinematic_mode="FK(q), world J(q) qdot; analytic pose pullback and central-difference dJ/dq twist pullback",
        jacobian_difference_step_rad=step,
    )
    return result
