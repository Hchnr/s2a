"""Explicit NumPy operation order for the pinned independent OSC replay.

Reference: robosuite control_utils.py SHA256
1d0f1a063894b1e8289339e223665e14e8aec492c31e9e00f42edea8dc735f43.
This is an opt-in algebra primitive, not a replacement for the historical
predictor. NumPy agreement does not certify native JIT, a complete response,
contacts, actuator behaviour, or control success. Inputs are float64 arrays in
the world frame at the controller SITE; the caller owns validation and state.
"""

import numpy as np


def replay_order_uncoupled_torque(mass, jacobian, pose, goal, q, velocity, initial_joint, kp, kd, *, null_kp=10.0):
    """Return pre-bias, pre-clipping torque with explicit replay grouping.

    Keep independent position/orientation inertia products, separate spatial
    velocity products, one concatenated task projection, and mass-times-spring
    before nullspace projection. Algebraically equivalent reassociation can
    change double-precision results and strict downstream reserve ratios.
    """
    jp, jr = jacobian[:3], jacobian[3:]
    inverse = np.linalg.inv(mass)
    full_inverse = np.dot(np.dot(jacobian, inverse), jacobian.T)
    position_inverse = np.dot(np.dot(jp, inverse), jp.T)
    orientation_inverse = np.dot(np.dot(jr, inverse), jr.T)
    lambda_full = np.linalg.pinv(full_inverse)
    lambda_position = np.linalg.pinv(position_inverse)
    lambda_orientation = np.linalg.pinv(orientation_inverse)
    jbar = np.dot(inverse, jacobian.T).dot(lambda_full)
    null = np.eye(jacobian.shape[-1], jacobian.shape[-1]) - np.dot(jbar, jacobian)
    position_error = goal[:3, 3] - pose[:3, 3]
    orientation_error = 0.5 * (
        np.cross(pose[:3, 0], goal[:3, 0]) + np.cross(pose[:3, 1], goal[:3, 1]) + np.cross(pose[:3, 2], goal[:3, 2])
    )
    # These are matmul in the independent replay, unlike control_utils' dot
    # products above. dot and matmul can round differently for strided views.
    force = kp[:3] * position_error - kd[:3] * (jp @ velocity)
    moment = kp[3:] * orientation_error - kd[3:] * (jr @ velocity)
    task = jacobian.T @ np.r_[lambda_position @ force, lambda_orientation @ moment]
    joint_kv = np.sqrt(null_kp) * 2
    spring = np.dot(mass, null_kp * (initial_joint - q) - joint_kv * velocity)
    return task + np.dot(null.T, spring)
