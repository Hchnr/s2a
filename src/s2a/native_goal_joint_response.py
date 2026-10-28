"""Robot response with the pinned native JIT quaternion goal arithmetic.

The recurrence is derived, unchanged except for goal construction, from
nonlinear_joint_response.py SHA256
7f22ca6a5e3b744166606536dfecdb623571592712959eab49249371b4361d5b.
Exact-zero rotation additionally accepts an explicit causal historical goal;
it remains rejected when that state is missing. The original executed source
is preserved. Contacts, joint limits, frictionloss
constraints and omitted-DOF acceleration coupling remain unmodelled.

Native robosuite enables Numba for quat2mat, but not axisangle2quat. Its JIT
float32 norm uses SciPy BLAS sdot, which can differ from NumPy dot; its in-place
normalization multiplies in float64 before storing float32. A Python-only copy
also changes scalar promotion between NumPy 1.26 and 2.x. Explicit arithmetic
below avoids those accidental substitutions without importing robosuite.

The companion test file provides a CPU-only native-1.26 functional check of
the archived source, compiled with numba.njit(cache=False). The nine archived
commands, seed42 random1000 and zero/tiny/pi cases matched exactly (1012 cases)
on NumPy1.26.4/Numba0.68.0. Component tests on NumPy2 do not certify another
BLAS, compiler or native runtime. This goal correction is not control or task
qualification, nor a claim that every remaining OSC operation is bitwise JIT.
"""

import math

import numpy as np
from scipy.linalg.blas import sdot

from s2a.geometry import finite, validate_transform
from s2a.nonlinear_joint_response import uncoupled_torque


def native_goal_rotation(physical_delta):
    """Return the native non-orthogonal rounded rotation increment, as float64.

    Axis-angle to quaternion is float64; reorder wxyz after casting float32.
    The float32 norm is the same SciPy BLAS sdot used by pinned native Numba.
    Normalize using float64 products rounded once into float32, then compute
    the outer product and off-diagonal sums in float32. Diagonal subtraction
    begins with float64 1.0, as native JIT does. Do not project back onto SO(3).

    Zero returns identity here; the rollout requires an explicit historical
    goal for zero commands because native zero commands latch that state.
    """
    delta = finite(physical_delta, (3,))
    angle = float(np.linalg.norm(delta))
    if not np.isfinite(angle):
        raise ValueError("Finite rotation norm required")
    if angle == 0:
        return np.eye(3)
    quaternion = np.zeros(4, dtype=np.float64)
    quaternion[3] = np.cos(angle / 2.0)
    quaternion[:3] = (delta / angle) * np.sin(angle / 2.0)
    q = quaternion.astype(np.float32)[[3, 0, 1, 2]]
    norm_squared = float(sdot(q, q))
    if norm_squared < np.finfo(np.float64).eps * 4.0:
        return np.eye(3)
    q = (q.astype(np.float64) * math.sqrt(2.0 / norm_squared)).astype(np.float32)
    outer = np.outer(q, q)
    return np.array(
        [
            [1.0 - float(outer[2, 2]) - float(outer[3, 3]), outer[1, 2] - outer[3, 0], outer[1, 3] + outer[2, 0]],
            [outer[1, 2] + outer[3, 0], 1.0 - float(outer[1, 1]) - float(outer[3, 3]), outer[2, 3] - outer[1, 0]],
            [outer[1, 3] - outer[2, 0], outer[2, 3] + outer[1, 0], 1.0 - float(outer[1, 1]) - float(outer[2, 2])],
        ],
        dtype=np.float64,
    )


def rollout_native_goal_joint_response(
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
