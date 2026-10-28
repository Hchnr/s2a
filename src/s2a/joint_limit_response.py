"""Frozen joint OSC response with an explicitly ideal unilateral hard-limit model.

This candidate preserves native uncoupled task gains and the joint nullspace
spring. It does not model MuJoCo soft limits, contacts, torque clipping or dJ/dt.
It is not a replacement for qualified physical response evidence.
"""

import numpy as np
from scipy.linalg import block_diag, expm

from s2a.geometry import finite
from s2a.nullspace_convex_qp import solve_qp


def project_box(mass, value, lower, upper):
    """Mass-metric nearest feasible vector, allowing inward motion at either face."""
    value = np.asarray(value, dtype=float)
    n = len(value)
    mass, value = finite(mass, (n, n)), finite(value, (n,))
    lower, upper = finite(lower, (n,)), finite(upper, (n,))
    if not np.allclose(mass, mass.T, atol=1e-12, rtol=1e-12) or np.any(lower > upper):
        raise ValueError("Symmetric positive mass and ordered bounds required")
    np.linalg.cholesky(mass)
    if np.all(value >= lower) and np.all(value <= upper):
        return value.copy()
    identity = np.eye(n)
    result = solve_qp(
        mass, -mass @ value, np.vstack((identity, -identity)), np.r_[upper, -lower], np.clip(value, lower, upper)
    )
    return np.clip(result["solution"], lower, upper)


def build_joint_response(mass, jacobian, kp, kd, q, initial_joint, *, null_kp, dt):
    """Exact frozen linear OSC transition around current q and identity rotation goal.

    Input is one held physical world pose increment, not normalized action.
    State is [joint displacement from q, joint velocity]. No object state enters.
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    q, initial_joint = finite(q, (n,)), finite(initial_joint, (n,))
    mass, j = finite(mass, (n, n)), finite(jacobian, (6, n))
    kp, kd = finite(kp, (6,)), finite(kd, (6,))
    if (
        n < 6
        or not np.isfinite(dt)
        or dt <= 0
        or not np.isfinite(null_kp)
        or null_kp < 0
        or np.any(kp <= 0)
        or np.any(kd <= 0)
        or not np.allclose(mass, mass.T, atol=1e-12, rtol=1e-12)
        or np.linalg.matrix_rank(j) != 6
    ):
        raise ValueError("Full task rank, positive gains/time and symmetric SPD robot mass required")
    np.linalg.cholesky(mass)
    inverse_j = np.linalg.solve(mass, j.T)
    task_inverse = j @ inverse_j
    lambdas = block_diag(np.linalg.pinv(task_inverse[:3, :3]), np.linalg.pinv(task_inverse[3:, 3:]))
    null = np.eye(n) - inverse_j @ np.linalg.pinv(task_inverse) @ j
    null_force = null.T @ mass
    forcing = j.T @ lambdas @ np.diag(kp)
    spring = forcing @ j + null_kp * null_force
    damping = j.T @ lambdas @ np.diag(kd) @ j + 2 * np.sqrt(null_kp) * null_force
    # Augment with the held six-dimensional input and one constant affine term.
    system = np.zeros((2 * n + 7, 2 * n + 7))
    system[:n, n : 2 * n] = np.eye(n)
    system[n : 2 * n, :n] = -np.linalg.solve(mass, spring)
    system[n : 2 * n, n : 2 * n] = -np.linalg.solve(mass, damping)
    system[n : 2 * n, 2 * n : 2 * n + 6] = np.linalg.solve(mass, forcing)
    system[n : 2 * n, -1] = np.linalg.solve(mass, null_kp * null_force @ (initial_joint - q))
    transition = expm(system * dt)
    return dict(
        mass=mass.copy(),
        jacobian=j.copy(),
        q=q.copy(),
        dt=float(dt),
        transition=transition[: 2 * n, : 2 * n],
        input=transition[: 2 * n, 2 * n : 2 * n + 6],
        affine=transition[: 2 * n, -1],
    )


def predict_joint_response(model, velocity, physical_delta, lower, upper, *, substeps, enforce_limits):
    """Hold one command; optionally project position and outward boundary velocity.

    Projection models ideal impacts, not native soft-limit forces. Inward
    velocities remain feasible. Starting penetrations are explicitly reported.
    """
    q, mass = np.asarray(model["q"]), np.asarray(model["mass"])
    n = len(q)
    velocity, delta = finite(velocity, (n,)), finite(physical_delta, (6,))
    lower, upper = finite(lower, (n,)), finite(upper, (n,))
    if type(substeps) is not int or substeps < 1 or type(enforce_limits) is not bool or np.any(lower >= upper):
        raise ValueError("Positive integer substeps, explicit limit mode and nonempty joint ranges required")
    state = np.r_[np.zeros(n), velocity]
    projected_steps = 0
    for _ in range(substeps):
        state = model["transition"] @ state + model["input"] @ delta + model["affine"]
        if enforce_limits:
            candidate = q + state[:n]
            position = project_box(mass, candidate, lower, upper)
            at_lower, at_upper = position <= lower + 1e-10, position >= upper - 1e-10
            if np.any(at_lower) or np.any(at_upper):
                # Only active boundary normals constrain velocity. Zero is feasible.
                a = np.vstack((-np.eye(n)[at_lower], np.eye(n)[at_upper]))
                v = state[n:].copy()
                if np.any(a @ v > 0):
                    state[n:] = solve_qp(mass, -mass @ v, a, np.zeros(len(a)), np.zeros(n))["solution"]
            projected_steps += int(not np.array_equal(position, candidate))
            state[:n] = position - q
    return dict(
        joint_position=q + state[:n],
        joint_velocity=state[n:],
        task_displacement=model["jacobian"] @ state[:n],
        task_velocity=model["jacobian"] @ state[n:],
        position_projected_steps=projected_steps,
        initial_max_penetration_rad=float(max(0, np.max(lower - q), np.max(q - upper))),
        evidence="frozen linear model with optional ideal hard impacts",
    )
