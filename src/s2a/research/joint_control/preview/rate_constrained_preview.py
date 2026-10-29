"""218 fixed-OSC translation preview with bounds on every planned actual command."""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import Bounds, LinearConstraint, minimize


def plan_translation(position, velocity, references, reference_velocities, previous_command, contract, *, innovation):
    position, velocity, previous = (np.asarray(v, dtype=float) for v in (position, velocity, previous_command))
    references, targets, innovation = (
        np.asarray(v, dtype=float) for v in (references, reference_velocities, innovation)
    )
    if (
        position.shape != (3,)
        or velocity.shape != (3,)
        or previous.shape != (3,)
        or references.ndim != 2
        or references.shape[1] != 3
        or not 1 <= len(references) <= 6
        or targets.shape != references.shape
        or innovation.shape != (2, 3)
        or not all(np.isfinite(v).all() for v in (position, velocity, previous, references, targets, innovation))
    ):
        raise ValueError(
            "Finite current translation state, actual previous command, completed innovation and1–6 generated references required"
        )
    if contract["control_freq"] != 20 or contract["control_frame"] != "MuJoCo world":
        raise ValueError("Fixed20Hz world controller required")
    values = [
        np.asarray(contract[k], dtype=float) for k in ("kp", "kd", "input_min", "input_max", "output_min", "output_max")
    ]
    if any(v.shape != (6,) or not np.isfinite(v).all() for v in values):
        raise ValueError("Finite actual six-axis controller contract required")
    kp, kd, lo, hi, outlo, outhi = (v[:3] for v in values)
    if (
        np.any(kp <= 0)
        or np.any(kd <= 0)
        or np.any(lo >= hi)
        or np.any(outlo >= outhi)
        or np.any(previous < lo)
        or np.any(previous > hi)
    ):
        raise ValueError("Positive gains, ordered interface and feasible actual previous command required")
    horizon, dt = len(references), 0.05
    difference = np.eye(horizon) - np.eye(horizon, k=-1)
    commands = np.zeros((horizon, 3))
    positions, velocities = np.zeros_like(commands), np.zeros_like(commands)
    diagnostics = []
    for axis in range(3):
        transition = expm(np.array([[0.0, 1.0], [-kp[axis], -kd[axis]]]) * dt)
        r, b, c, d = 1 - transition[0, 0], transition[0, 1], kp[axis] * transition[0, 1], transition[1, 1]
        if r < 1e-5:
            raise ValueError("Insufficient fixed controller response")
        slope = (outhi[axis] - outlo[axis]) / (hi[axis] - lo[axis])
        offset = outlo[axis] - slope * lo[axis]
        base_p, base_v = position[axis], velocity[axis]
        row_p, row_v = np.zeros(horizon), np.zeros(horizon)
        matrix, target = np.empty((2 * horizon, horizon)), np.empty(2 * horizon)
        for step in range(horizon):
            base_p, base_v = (
                base_p + b * base_v + r * offset + innovation[0, axis],
                d * base_v + c * offset + innovation[1, axis],
            )
            row_p = row_p + b * row_v
            row_p[step] += r * slope
            row_v = d * row_v
            row_v[step] += c * slope
            matrix[2 * step], matrix[2 * step + 1] = row_p / 0.01, dt * row_v / 0.01
            target[2 * step], target[2 * step + 1] = (
                (references[step, axis] - base_p) / 0.01,
                dt * (targets[step, axis] - base_v) / 0.01,
            )
        lower, upper = np.full(horizon, -0.25), np.full(horizon, 0.25)
        lower[0] += previous[axis]
        upper[0] += previous[axis]

        def objective(u):
            error = matrix @ u - target
            return 0.5 * float(error @ error)

        def gradient(u):
            return matrix.T @ (matrix @ u - target)

        result = minimize(
            objective,
            np.full(horizon, previous[axis]),
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(np.full(horizon, lo[axis]), np.full(horizon, hi[axis])),
            constraints=LinearConstraint(difference, lower, upper),
            options=dict(ftol=1e-12, maxiter=200),
        )
        if not result.success:
            raise RuntimeError(f"Preview QP axis{axis} failed: {result.message}")
        u = result.x
        if (
            np.any(u < lo[axis] - 1e-9)
            or np.any(u > hi[axis] + 1e-9)
            or np.any(difference @ u < lower - 1e-9)
            or np.any(difference @ u > upper + 1e-9)
        ):
            raise RuntimeError("Preview QP returned infeasible planned commands")
        commands[:, axis] = u
        p, v = position[axis], velocity[axis]
        for step in range(horizon):
            physical = offset + slope * u[step]
            p, v = p + b * v + r * physical + innovation[0, axis], d * v + c * physical + innovation[1, axis]
            positions[step, axis], velocities[step, axis] = p, v
        diagnostics.append(dict(axis=axis, iterations=int(result.nit), objective=float(result.fun)))
    return dict(
        command=commands[0].copy(),
        planned_commands=commands,
        predicted_positions=positions,
        predicted_velocities=velocities,
        horizon=horizon,
        axes=diagnostics,
    )
