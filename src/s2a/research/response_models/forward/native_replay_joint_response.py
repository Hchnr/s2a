"""Separate causal response variant with independent-replay arithmetic.

Derived from native_goal_joint_response.py; the original default is unchanged.
Only torque operation order, damping grouping and clipping expression change.
Not qualified for physical control or an entire search by algebra tests.
"""

import numpy as np

from s2a.geometry import finite, validate_transform
from s2a.research.response_models.forward.native_goal_joint_response import native_goal_rotation
from s2a.research.response_models.forward.native_osc_torque import replay_order_uncoupled_torque


def rollout_replay_order_joint_response(
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
    latched_rotation_goal=None,
):
    """Keep the original causal recurrence and use a native-rounded reset goal.

    ``dynamics(q, velocity)`` supplies mass, jacobian, SITE pose, bias, diagonal
    damping and optional passive_extra. Only M/J selection differs between
    fixed/updated modes. Bias is added before torque clipping and subtracted in
    the free plant; diagonal damping is implicit. Each query receives copies
    of this rollout's own predicted state. No inner observation is accepted.

    Exactly zero rotation retains ``latched_rotation_goal``, a finite 3x3
    goal available from the controller's causal history, and still resets
    position at the current state. Missing history is rejected. Nonzero
    rotation constructs a new native goal and ignores this optional argument.
    """
    q = finite(q).copy()
    if q.ndim != 1 or len(q) < 6:
        raise ValueError("At least six robot joint coordinates required")
    n = len(q)
    velocity, initial_joint = finite(velocity, (n,)).copy(), finite(initial_joint, (n,))
    delta, kp, kd = finite(physical_delta, (6,)), finite(kp, (6,)), finite(kd, (6,))
    lower, upper = finite(torque_lower, (n,)), finite(torque_upper, (n,))
    zero_rotation = bool(np.all(delta[3:] == 0))
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
        or (zero_rotation and latched_rotation_goal is None)
    ):
        raise ValueError(
            "Positive time/gains, strict torque ranges, explicit mode and nonzero rotation reset or explicit latch required"
        )
    latched = finite(latched_rotation_goal, (3, 3)).copy() if zero_rotation else None

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
    goal[:3, :3] = latched if zero_rotation else native_goal_rotation(delta[3:]) @ goal[:3, :3]
    records = []
    for k in range(substeps):
        current = first if k == 0 else sample()
        mass, jacobian = current[:2] if update_matrices else first[:2]
        pose, bias, damping, passive = current[2:]
        requested = (
            replay_order_uncoupled_torque(
                mass, jacobian, pose, goal, q, velocity, initial_joint, kp, kd, null_kp=null_kp
            )
            + bias
        )
        torque = np.minimum(upper, np.maximum(lower, requested))
        velocity = np.linalg.solve(mass + np.diag(dt * damping), mass @ velocity + dt * (torque - bias + passive))
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
