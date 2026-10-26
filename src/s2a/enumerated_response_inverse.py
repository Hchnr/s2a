"""Invert coupled motion/velocity response inside actual command constraints."""

from itertools import product

import numpy as np


def vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"Finite vector of size {size} required")
    return value


def box_least_squares(matrix, target, lower, upper):
    """Enumerate all 27 faces of a three-variable box, including its interior.

    Each face solves the original least-squares objective by SVD. No iterative
    convergence flag or change of objective is involved. Equal objective values
    prefer the minimum-norm command, with deterministic face order as last tie.
    """
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    lower, upper = vector(lower, 3), vector(upper, 3)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 3
        or target.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(target).all()
        or np.any(lower > upper)
    ):
        raise ValueError("Finite least-squares system and ordered three-variable box required")
    best, best_key = None, None
    for status in product((0, -1, 1), repeat=3):
        status = np.asarray(status)
        free = (status == 0) & (lower != upper)
        command = np.where(status == 1, upper, lower).copy()
        rhs = target - matrix[:, ~free] @ command[~free]
        if np.any(free):
            command[free] = np.linalg.lstsq(matrix[:, free], rhs, rcond=None)[0]
        tolerance = 32 * np.finfo(float).eps * (1 + np.maximum(np.abs(lower), np.abs(upper)))
        if np.any(command < lower - tolerance) or np.any(command > upper + tolerance):
            continue
        command = np.clip(command, lower, upper)
        residual = matrix @ command - target
        key = (float(residual @ residual), float(command @ command))
        if best_key is None or key < best_key:
            best, best_key = command, key
    if best is None:
        raise RuntimeError("No finite box least-squares candidate")
    residual = matrix @ best - target
    gradient = matrix.T @ residual
    tolerance = 1e-10 * (1 + np.linalg.norm(matrix) * np.linalg.norm(residual))
    for axis in range(3):
        if lower[axis] == upper[axis]:
            continue
        if best[axis] <= lower[axis] + 1e-12:
            valid = gradient[axis] >= -tolerance
        elif best[axis] >= upper[axis] - 1e-12:
            valid = gradient[axis] <= tolerance
        else:
            valid = abs(gradient[axis]) <= tolerance
        if not valid:
            raise RuntimeError("Box least-squares candidate violates KKT conditions")
    return best


def translation_command(
    offset,
    jacobian,
    desired_motion,
    desired_velocity,
    *,
    dt,
    input_min,
    input_max,
    output_min,
    output_max,
    previous_command,
    max_rate,
):
    """Response rows are [motion3, dt*next_velocity3]; Jacobian is per physical metre."""
    offset = vector(offset, 6)
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape != (6, 3) or not np.isfinite(jacobian).all():
        raise ValueError("Finite6x3 physical translation response required")
    if not np.isfinite(dt) or not np.isfinite(max_rate) or dt <= 0 or max_rate <= 0:
        raise ValueError("Positive finite period and normalized command rate required")
    lo, hi, outlo, outhi, previous = (
        vector(v, 3) for v in (input_min, input_max, output_min, output_max, previous_command)
    )
    if np.any(lo >= hi) or np.any(outlo >= outhi) or np.any(previous < lo) or np.any(previous > hi):
        raise ValueError("Ordered interface bounds and feasible previous command required")
    target = np.r_[vector(desired_motion, 3), dt * vector(desired_velocity, 3)]
    scale = (outhi - outlo) / (hi - lo)
    matrix = jacobian * scale[None, :]
    constant = offset + jacobian @ (outlo - lo * scale)
    lower = np.maximum(lo, previous - max_rate * dt)
    upper = np.minimum(hi, previous + max_rate * dt)
    if np.any(lower > upper):
        raise ValueError("Infeasible command/rate intersection")
    command = box_least_squares(matrix, target - constant, lower, upper)
    physical = outlo + (command - lo) * scale
    predicted = offset + jacobian @ physical
    if not np.isfinite(command).all() or np.any(command < lower - 1e-12) or np.any(command > upper + 1e-12):
        raise RuntimeError("Invalid bounded inverse output")
    return dict(
        command=command,
        physical_delta=physical,
        predicted_motion=predicted[:3],
        predicted_linear_velocity=predicted[3:] / dt,
        squared_objective=float(np.sum((predicted - target) ** 2)),
        command_lower=lower,
        command_upper=upper,
    )
