"""Frozen-pose OSC linear sensitivity, excluding contacts and torque clipping."""

import numpy as np
from scipy.linalg import block_diag, expm


def goal_response(mass, jacobian, kp, kd, dt):
    """Return displacement/twist sensitivity to a held six-dimensional goal.

    Assumes the controller uses separate position/orientation operational inertia.
    Rotation is a local world-frame perturbation, not a finite pose predictor.
    """
    mass, jacobian = np.asarray(mass, dtype=float), np.asarray(jacobian, dtype=float)
    kp, kd = np.asarray(kp, dtype=float), np.asarray(kd, dtype=float)
    n = len(mass)
    if mass.shape != (n, n) or jacobian.shape != (6, n) or kp.shape != (6,) or kd.shape != (6,):
        raise ValueError("Joint mass, world SITE Jacobian and six gains required")
    if not all(np.isfinite(x).all() for x in (mass, jacobian, kp, kd)) or not np.isfinite(dt):
        raise ValueError("Finite dynamics required")
    if dt <= 0 or np.any(kp <= 0) or np.any(kd < 0):
        raise ValueError("Positive period/stiffness and nonnegative damping required")
    if not np.allclose(mass, mass.T, rtol=0, atol=1e-10):
        raise ValueError("Symmetric mass required")
    np.linalg.cholesky(mass)
    a = jacobian @ np.linalg.solve(mass, jacobian.T)
    b = a @ block_diag(np.linalg.pinv(a[:3, :3]), np.linalg.pinv(a[3:, 3:]))
    system = np.zeros((18, 18))
    system[:6, 6:12] = np.eye(6)
    system[6:12, :6] = -b * kp[None, :]
    system[6:12, 6:12] = -b * kd[None, :]
    system[6:12, 12:] = b * kp[None, :]
    response = expm(system * dt)[:12, 12:]
    return response, a, b
