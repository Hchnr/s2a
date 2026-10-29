"""Autonomous null-mode forecast of the frozen native OSC joint model."""

import numpy as np
from scipy.linalg import null_space

from s2a.geometry import finite


def forecast_null_modes(mass, jacobian, q, velocity, initial_joint, *, null_kp=10.0, dt=0.05):
    """Mass-weighted null rows annihilate M^-1 J^T task forcing.

    The native scalar null spring/damper gives a critically damped autonomous
    mode only for frozen M/J without joint/contact reactions or torque clipping.
    This is not a global controllability claim about the nonlinear robot.
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    mass, j = finite(mass, (n, n)), finite(jacobian, (6, n))
    q, velocity, initial_joint = finite(q, (n,)), finite(velocity, (n,)), finite(initial_joint, (n,))
    if (
        n < 6
        or np.linalg.matrix_rank(j) != 6
        or not np.allclose(mass, mass.T, atol=1e-12, rtol=1e-12)
        or not np.isfinite(null_kp)
        or null_kp < 0
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Full-rank task Jacobian, positive mass/time and nonnegative null gain required")
    np.linalg.cholesky(mass)
    basis = null_space(j)
    rows = (mass @ basis).T
    if len(rows):
        rows /= np.linalg.norm(rows, axis=1)[:, None]
    current, speed, equilibrium = rows @ q, rows @ velocity, rows @ initial_joint
    omega = np.sqrt(null_kp)
    decay = np.exp(-omega * dt)
    future = equilibrium + decay * ((1 + omega * dt) * (current - equilibrium) + dt * speed)
    future_speed = decay * ((1 - omega * dt) * speed - null_kp * dt * (current - equilibrium))
    return dict(
        rows=rows,
        current=current,
        next=future,
        next_velocity=future_speed,
        displacement=future - current,
        task_forcing_residual=rows @ np.linalg.solve(mass, j.T),
        scope="frozen free OSC model only; constraints and nonlinear configuration changes omitted",
    )
