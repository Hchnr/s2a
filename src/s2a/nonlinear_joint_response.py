"""Robot-only numerical OSC prediction with explicit frozen/updated M/J ablation.

No contact or joint-limit force is represented. The gripper configuration and
other unmodelled DOFs belong to the caller's fixed current-state approximation.
This is a response diagnostic, not a qualified simulator or control policy.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform


def uncoupled_torque(mass, jacobian, pose, goal, q, velocity, initial_joint, kp, kd, *, null_kp=10.0):
    """Native OSC sine-angle error, uncoupled task inertia and nullspace spring.

    Returns torque before bias compensation and actuator clipping. All spatial
    quantities use the same world frame and the controller SITE, not TCP.
    """
    inverse_j = np.linalg.solve(mass, jacobian.T)
    task_inverse = jacobian @ inverse_j
    position_error = goal[:3, 3] - pose[:3, 3]
    rotation_error = 0.5 * np.cross(pose[:3, :3].T, goal[:3, :3].T).sum(axis=0)
    wrench = kp * np.r_[position_error, rotation_error] - kd * (jacobian @ velocity)
    task = jacobian[:3].T @ (np.linalg.pinv(task_inverse[:3, :3]) @ wrench[:3])
    task += jacobian[3:].T @ (np.linalg.pinv(task_inverse[3:, 3:]) @ wrench[3:])
    null = np.eye(len(q)) - inverse_j @ np.linalg.pinv(task_inverse) @ jacobian
    return task + null.T @ mass @ (null_kp * (initial_joint - q) - 2 * np.sqrt(null_kp) * velocity)


def rollout_nonlinear_joint_response(
    q,
    velocity,
    physical_delta,
    initial_joint,
    kp,
    kd,
    dynamics,
    torque_lower,
    torque_upper,
    *,
    dt,
    substeps,
    update_matrices,
    null_kp=10.0,
):
    """Hold one reset goal; query only the recursively predicted q and velocity.

    ``dynamics(q, velocity)`` returns mass, jacobian, pose, bias, damping and
    optionally passive_extra (passive force excluding diagonal viscous damping).
    Only M/J selection differs between the two modes, in both OSC and plant.
    FK, bias and passive viscous damping follow each predicted state in both.
    Bias is added before clipping and subtracted in the free plant. Integration
    is semi-implicit Euler with implicit diagonal viscous damping:
    (M + dt D) v_next = M v + dt (clip(tau_OSC + bias) - bias + passive_extra),
    q_next = q + dt v_next. No position projection or goal re-anchoring occurs.

    Exact-zero rotation latch requires additional controller state and is
    deliberately rejected; the registered nine-command diagnostic has none.
    """
    q = finite(q).copy()
    if q.ndim != 1 or len(q) < 6:
        raise ValueError("At least six robot joint coordinates required")
    n = len(q)
    velocity, initial_joint = finite(velocity, (n,)).copy(), finite(initial_joint, (n,))
    delta, kp, kd = finite(physical_delta, (6,)), finite(kp, (6,)), finite(kd, (6,))
    lower, upper = finite(torque_lower, (n,)), finite(torque_upper, (n,))
    if (
        not np.isfinite(dt)
        or dt <= 0
        or type(substeps) is not int
        or substeps < 1
        or type(update_matrices) is not bool
        or not np.isfinite(null_kp)
        or null_kp < 0
        or np.any(kp <= 0)
        or np.any(kd <= 0)
        or np.any(lower >= upper)
        or np.all(delta[3:] == 0)
    ):
        raise ValueError("Positive time/gains, strict torque ranges, explicit mode and nonzero rotation reset required")

    def sample():
        raw = dynamics(q.copy(), velocity.copy())
        m = finite(raw["mass"], (n, n)).copy()
        j = finite(raw["jacobian"], (6, n)).copy()
        pose = validate_transform(raw["pose"]).copy()
        bias, damping = finite(raw["bias"], (n,)).copy(), finite(raw["damping"], (n,)).copy()
        passive = finite(raw.get("passive_extra", np.zeros(n)), (n,)).copy()
        if pose.shape != (4, 4) or not np.allclose(m, m.T, atol=1e-12, rtol=1e-12) or np.any(damping < 0):
            raise ValueError("Single SITE pose, symmetric SPD mass and nonnegative damping required")
        np.linalg.cholesky(m)
        return m, j, pose, bias, damping, passive

    first = sample()
    goal = first[2].copy()
    goal[:3, 3] += delta[:3]
    goal[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ goal[:3, :3]
    records = []
    for k in range(substeps):
        current = first if k == 0 else sample()
        mass, jacobian = current[:2] if update_matrices else first[:2]
        pose, bias, damping, passive = current[2:]
        requested = (
            uncoupled_torque(mass, jacobian, pose, goal, q, velocity, initial_joint, kp, kd, null_kp=null_kp) + bias
        )
        torque = np.clip(requested, lower, upper)
        velocity = np.linalg.solve(mass + dt * np.diag(damping), mass @ velocity + dt * (torque - bias + passive))
        q = q + dt * velocity
        records.append(
            dict(
                joint_position=q.copy(),
                joint_velocity=velocity.copy(),
                requested_torque=requested,
                applied_torque=torque,
                clipped=torque != requested,
                mass_change=float(np.max(np.abs(current[0] - first[0]))),
                jacobian_change=float(np.max(np.abs(current[1] - first[1]))),
            )
        )
    endpoint = sample()
    return dict(
        **{key: np.asarray([r[key] for r in records]) for key in records[0]},
        goal=goal,
        endpoint_pose=endpoint[2],
        endpoint_twist=endpoint[1] @ velocity,
        update_matrices=update_matrices,
        dynamics_queries=substeps + 1,
    )
