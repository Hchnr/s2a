"""Invert coupled motion/velocity response inside actual command constraints."""

import numpy as np
from scipy.optimize import lsq_linear


def vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"Finite vector of size {size} required")
    return value


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
    fixed = lower == upper
    command = lower.copy()
    free = ~fixed
    if np.any(free):
        rhs = target - constant - matrix[:, fixed] @ command[fixed]
        solution = lsq_linear(matrix[:, free], rhs, bounds=(lower[free], upper[free]), method="bvls", tol=1e-12)
        if not solution.success:
            raise RuntimeError("Bounded response inverse did not converge")
        command[free] = solution.x
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
