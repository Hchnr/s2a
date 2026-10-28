"""Long-horizon frozen joint rollout with reset goals and active-face derivatives.

The joint hard impacts remain an ideal model. M/J stay frozen; soft contacts,
torque saturation and dJ/dt are not represented. Derivatives are local to a
fixed strictly active projection face; weak faces are explicitly reported.
"""

import numpy as np
from scipy.linalg import block_diag, expm
from scipy.spatial.transform import Rotation

from s2a.full_coupled_pose_inverse import left_jacobian
from s2a.geometry import finite, transform, validate_transform
from s2a.joint_limit_response import build_joint_response
from s2a.nullspace_convex_qp import solve_qp


def build_joint_preview(mass, jacobian, kp, kd, q, initial_joint, *, null_kp, dt):
    """Cache one inner transition and its dependence on each period's anchor q."""
    base = build_joint_response(mass, jacobian, kp, kd, q, initial_joint, null_kp=null_kp, dt=dt)
    mass, j = base["mass"], base["jacobian"]
    n = len(base["q"])
    inverse_j = np.linalg.solve(mass, j.T)
    task_inverse = j @ inverse_j
    lambdas = block_diag(np.linalg.pinv(task_inverse[:3, :3]), np.linalg.pinv(task_inverse[3:, 3:]))
    null = np.eye(n) - inverse_j @ np.linalg.pinv(task_inverse) @ j
    null_force = null.T @ mass
    forcing = j.T @ lambdas @ np.diag(kp)
    spring = forcing @ j + null_kp * null_force
    damping = j.T @ lambdas @ np.diag(kd) @ j + 2 * np.sqrt(null_kp) * null_force
    augmented = np.zeros((3 * n, 3 * n))
    augmented[:n, n : 2 * n] = np.eye(n)
    augmented[n : 2 * n, :n] = -np.linalg.solve(mass, spring)
    augmented[n : 2 * n, n : 2 * n] = -np.linalg.solve(mass, damping)
    augmented[n : 2 * n, 2 * n :] = -null_kp * np.linalg.solve(mass, null_force)
    base["anchor_response"] = expm(augmented * dt)[: 2 * n, 2 * n :]
    return base


def _project(mass, value, constraints, bounds, initial):
    """Return projection, local derivative and weak-face warning."""
    n = len(value)
    if not len(bounds):
        return value.copy(), np.eye(n), False
    slack = bounds - constraints @ value
    if np.all(slack >= 0):
        return value.copy(), np.eye(n), bool(slack.min() <= 1e-10)
    result = solve_qp(mass, -mass @ value, constraints, bounds, initial)
    active = np.asarray(result["active"], dtype=int)
    positive = np.asarray(result["multipliers"]) > 1e-10
    normals = constraints[active[positive]]
    derivative = np.eye(n)
    if len(normals):
        inverse_normals = np.linalg.solve(mass, normals.T)
        derivative -= inverse_normals @ np.linalg.solve(normals @ inverse_normals, normals)
    return result["solution"], derivative, bool(np.any(~positive))


def rollout_joint_preview(model, velocity, physical_deltas, site, lower, upper, *, enforce_limits):
    """Reset each held delta about the predicted measured pose for up to H=40.

    Jacobians differentiate physical deltas, not normalized commands. Rotation
    Jacobians are left perturbations in the world frame. No future observation
    or object state is consumed; all inputs belong to the current planning call.
    """
    q0, j, mass = model["q"], model["jacobian"], model["mass"]
    n = len(q0)
    velocity = finite(velocity, (n,))
    physical_deltas = finite(physical_deltas)
    site = validate_transform(site)
    lower, upper = finite(lower, (n,)), finite(upper, (n,))
    if (
        physical_deltas.ndim != 2
        or physical_deltas.shape[1] != 6
        or not 1 <= len(physical_deltas) <= 40
        or site.shape != (4, 4)
        or np.any(lower >= upper)
        or type(enforce_limits) is not bool
        or not np.isclose(model["dt"] * 25, 0.05, atol=1e-12, rtol=0)
    ):
        raise ValueError("One to forty deltas, one SITE, strict joint ranges and fixed25x2ms required")
    if np.any(np.all(physical_deltas[:, 3:] == 0, axis=1)):
        raise ValueError("Every planned period needs a nonzero native reset representative; latch mode is unsupported")
    width = 6 * len(physical_deltas)
    q, p, rotation = q0.copy(), site[:3, 3].copy(), site[:3, :3].copy()
    qjac, vjac = np.zeros((n, width)), np.zeros((n, width))
    pjac, rjac = np.zeros((3, width)), np.zeros((3, width))
    identity = np.eye(n)
    box_a, box_b = np.vstack((identity, -identity)), np.r_[upper, -lower]
    records = []
    for k, delta in enumerate(physical_deltas):
        anchor, anchor_jac = q.copy(), qjac.copy()
        state, derivative = np.r_[np.zeros(n), velocity], np.vstack((np.zeros((n, width)), vjac))
        delta_jac = np.zeros((6, width))
        delta_jac[:, 6 * k : 6 * (k + 1)] = np.eye(6)
        affine = model["affine"] + model["anchor_response"] @ (anchor - q0)
        affine_jac = model["anchor_response"] @ anchor_jac
        weak_steps = projected_steps = 0
        for _ in range(25):
            state = model["transition"] @ state + model["input"] @ delta + affine
            derivative = model["transition"] @ derivative + model["input"] @ delta_jac + affine_jac
            if enforce_limits:
                candidate = anchor + state[:n]
                position, projection, weak_p = _project(mass, candidate, box_a, box_b, np.clip(candidate, lower, upper))
                position = np.clip(position, lower, upper)
                position_jac = projection @ (anchor_jac + derivative[:n])
                at_lower, at_upper = position <= lower + 1e-10, position >= upper - 1e-10
                normals = np.vstack((-identity[at_lower], identity[at_upper]))
                v, vprojection, weak_v = _project(mass, state[n:], normals, np.zeros(len(normals)), np.zeros(n))
                derivative[n:] = vprojection @ derivative[n:]
                state[n:] = v
                projected_steps += int(not np.array_equal(position, candidate))
                weak_steps += int(weak_p or weak_v)
                state[:n], derivative[:n] = position - anchor, position_jac - anchor_jac
        movement = j @ state[:n]
        movement_jac = j @ derivative[:n]
        increment = Rotation.from_rotvec(movement[3:]).as_matrix()
        rjac = left_jacobian(movement[3:]) @ movement_jac[3:] + increment @ rjac
        rotation = increment @ rotation
        p, pjac = p + movement[:3], pjac + movement_jac[:3]
        q, qjac = anchor + state[:n], anchor_jac + derivative[:n]
        velocity, vjac = state[n:].copy(), derivative[n:].copy()
        records.append(
            dict(
                pose=transform(p, rotation),
                twist=j @ velocity,
                joint_position=q.copy(),
                joint_velocity=velocity.copy(),
                position_jacobian=pjac.copy(),
                rotation_jacobian=rjac.copy(),
                twist_jacobian=j @ vjac,
                joint_position_jacobian=qjac.copy(),
                joint_velocity_jacobian=vjac.copy(),
                weak_projection_steps=weak_steps,
                position_projected_steps=projected_steps,
            )
        )
    result = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    result["initial_max_penetration_rad"] = float(max(0, np.max(lower - q0), np.max(q0 - upper)))
    return result
