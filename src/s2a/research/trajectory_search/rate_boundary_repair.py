"""Bounded numerical repair of a near-feasible box and rate plan."""

import numpy as np

from s2a.geometry import finite


def repair_nearby_rate_plan(commands, previous, lower, upper, *, max_rate, interior, max_correction):
    """Sequentially clamp a near-feasible plan; leave other constraints to audit."""
    commands = finite(commands)
    if commands.ndim != 2 or commands.shape[1] != 6 or not 1 <= len(commands) <= 8:
        raise ValueError("One to eight six-dimensional commands are required")
    previous, lower, upper = (finite(value, (6,)) for value in (previous, lower, upper))
    limits = finite([max_rate, interior, max_correction], (3,))
    if np.any(lower >= upper) or not 0 < limits[1] < limits[0] or limits[2] <= 0:
        raise ValueError("Increasing box and positive, bounded repair limits are required")
    if np.any(previous < lower) or np.any(previous > upper):
        raise ValueError("The previous command must be inside the box")
    repaired = np.empty_like(commands)
    last = previous.copy()
    for index, row in enumerate(commands):
        low = np.maximum(lower, last - (max_rate - interior))
        high = np.minimum(upper, last + (max_rate - interior))
        if np.any(low > high):
            raise ValueError("Command box and rate interval do not intersect")
        repaired[index] = np.clip(row, low, high)
        last = repaired[index]
    correction = float(np.max(np.abs(repaired - commands)))
    if correction > max_correction:
        raise ValueError("The requested plan is not close enough for numerical repair")
    change = np.abs(np.diff(np.vstack((previous, repaired)), axis=0))
    if np.any(repaired < lower) or np.any(repaired > upper) or np.max(change) > max_rate:
        raise ValueError("Numerical repair did not satisfy the original box and rate limits")
    return {
        "planned_commands": repaired,
        "maximum_command_change": correction,
        "maximum_rate_after_repair": float(np.max(change)),
        "scope": "box and rate only; pose and joint constraints require independent recomputation",
    }
