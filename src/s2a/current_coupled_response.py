"""228 frozen current OSC dynamics with the actual local SO(3) spring derivative."""

import numpy as np
from scipy.linalg import block_diag, expm

from s2a.geometry import finite
from s2a.osc import OSCInterface
from s2a.strict_convex_qp import solve_qp


def rotation_spring(relative_goal):
    a = finite(relative_goal, (3, 3))
    if not np.allclose(a.T @ a, np.eye(3), atol=1e-6, rtol=0) or np.linalg.det(a) <= 0:
        raise ValueError("Proper actual relative rotation goal required")
    forcing = 0.5 * np.array([a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]])
    # Current orientation is perturbed on the left in the world frame:
    # error(Exp(theta) R_current) = forcing - stiffness @ theta + O(theta^2).
    stiffness = 0.5 * (np.trace(a) * np.eye(3) - a.T)
    return forcing, stiffness


def build_response(mass, jacobian, kp, kd, relative_rotation_goal):
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1] or len(mass) < 6:
        raise ValueError("Square robot joint mass with at least six DOFs required")
    mass = finite(mass, mass.shape)
    j = finite(jacobian, (6, len(mass)))
    kp, kd = finite(kp, (6,)), finite(kd, (6,))
    if not np.allclose(mass, mass.T, atol=1e-10, rtol=0) or np.any(kp <= 0) or np.any(kd <= 0):
        raise ValueError("Symmetric positive mass and positive fixed PD gains required")
    np.linalg.cholesky(mass)
    if np.linalg.matrix_rank(j) != 6:
        raise ValueError("Full rank current world SITE Jacobian required")
    forcing, stiffness = rotation_spring(relative_rotation_goal)
    inverse_inertia = j @ np.linalg.solve(mass, j.T)
    coupling = inverse_inertia @ block_diag(
        np.linalg.pinv(inverse_inertia[:3, :3]), np.linalg.pinv(inverse_inertia[3:, 3:])
    )
    spring = block_diag(np.diag(kp[:3]), np.diag(kp[3:]) @ stiffness)
    system = np.zeros((18, 18))
    system[:6, 6:12] = np.eye(6)
    system[6:12, :6] = -coupling @ spring
    system[6:12, 6:12] = -coupling @ np.diag(kd)
    system[6:12, 12:] = coupling @ np.diag(kp)
    transition = expm(system * 0.05)
    return dict(
        dt=0.05,
        mass=mass.copy(),
        jacobian=j.copy(),
        kp=kp.copy(),
        kd=kd.copy(),
        relative_rotation_goal=np.asarray(relative_rotation_goal).copy(),
        rotation_forcing=forcing,
        rotation_stiffness=stiffness,
        inverse_inertia=inverse_inertia,
        acceleration_coupling=coupling,
        initial_velocity_response=transition[:12, 6:12],
        goal_forcing_response=transition[:12, 12:],
    )


def nominal_prediction(model, twist, physical_translation):
    velocity = finite(twist, (6,))
    translation = finite(physical_translation, (3,))
    f = finite(model["initial_velocity_response"], (12, 6))
    g = finite(model["goal_forcing_response"], (12, 6))
    forcing = finite(model["rotation_forcing"], (3,))
    return f @ velocity + g @ np.r_[translation, forcing]


def solve_translation(
    position, twist, desired_position, desired_velocity, previous_command, contract, model, *, innovation
):
    position, twist = finite(position, (3,)), finite(twist, (6,))
    target, reference_velocity = finite(desired_position, (3,)), finite(desired_velocity, (3,))
    previous, bias = finite(previous_command, (3,)), finite(innovation, (2, 3))
    if contract["control_freq"] != 20 or contract["control_frame"] != "MuJoCo world" or model["dt"] != 0.05:
        raise ValueError("Fixed20Hz world-frame model and contract required")
    for key in ("kp", "kd"):
        if not np.allclose(finite(contract[key], (6,)), model[key], atol=1e-12, rtol=0):
            raise ValueError("Model differs from actual fixed controller gains")
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    lo, hi, outlo, outhi = (
        getattr(interface, key)[:3] for key in ("input_min", "input_max", "output_min", "output_max")
    )
    if np.any(previous < lo) or np.any(previous > hi):
        raise ValueError("Feasible actual previous translation command required")
    slope = (outhi - outlo) / (hi - lo)
    offset = outlo - slope * lo
    base = nominal_prediction(model, twist, offset)
    g = np.asarray(model["goal_forcing_response"])[:, :3] * slope[None, :]
    matrix = np.vstack([g[:3], 0.05 * g[6:9]]) / 0.01
    target_vector = (
        np.r_[target - position - base[:3] - bias[0], 0.05 * (reference_velocity - base[6:9] - bias[1])] / 0.01
    )
    lower, upper = np.maximum(lo, previous - 0.25), np.minimum(hi, previous + 0.25)
    result = solve_qp(
        matrix.T @ matrix, -matrix.T @ target_vector, np.vstack([np.eye(3), -np.eye(3)]), np.r_[upper, -lower], previous
    )
    command = np.minimum(upper, np.maximum(lower, result["solution"]))
    delta = outlo + (command - lo) * (outhi - outlo) / (hi - lo)
    nominal = nominal_prediction(model, twist, delta)
    predicted = nominal.copy()
    predicted[:3] += bias[0]
    predicted[6:9] += bias[1]
    error = matrix @ command - target_vector
    return dict(
        command=command,
        physical_delta=delta,
        nominal_response=nominal,
        predicted_response=predicted,
        innovation=bias.copy(),
        lower=lower,
        upper=upper,
        objective=0.5 * float(error @ error),
        solver_iterations=result["iterations"],
        solver_max_stationarity=result["max_stationarity"],
    )
