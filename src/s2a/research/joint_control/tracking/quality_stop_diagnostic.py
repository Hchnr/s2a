"""Causal seed and empirical error at an actual stopped feedback endpoint."""

import numpy as np

from s2a.geometry import finite


def stopped_feedback_inputs(periods, start_step):
    """Shift the last dispatched plan; read no unexecuted decision or future state."""
    if type(start_step) is not int or start_step < 0 or len(periods) != 5:
        raise ValueError("Exactly five completed registered periods required")
    for offset, period in enumerate(periods):
        if period["step"] != start_step + offset:
            raise ValueError("Consecutive completed periods required")
    last = periods[-1]
    previous = finite(last["command"], (7,)).copy()
    plan = finite(last["decision"]["refined_plan"], (8, 6)).copy()
    if not np.array_equal(plan[0], previous[:6]):
        raise ValueError("The recorded plan must begin with the last actual command")
    shifted = np.vstack((plan[1:], plan[-1]))
    if (
        np.max(abs(shifted)) > 1 + 1e-12
        or np.max(abs(np.diff(np.vstack((previous[:6], shifted)), axis=0))) > 0.25 + 1e-12
    ):
        raise ValueError("Shifted completed plan must respect original box and rate")
    saved = last["completed_feedback"]["past_error_after_completion"]
    error = finite(saved["inner_position_error_rad"], (25, 7)).copy()
    if (
        saved["completed_periods"] != len(periods)
        or saved["last_completed_step"] != start_step + len(periods) - 1
        or saved["available"] is not True
        or saved["future_guarantee"] is not False
        or np.any(error < 0)
        or np.any(np.diff(error, axis=0) < 0)
    ):
        raise ValueError("A nondecreasing past-only completed error profile is required")
    return {
        "state": finite(last["actual_state"], (79,)).copy(),
        "previous": previous,
        "seed": shifted,
        "past_inner_error_rad": error,
        "completed_periods": len(periods),
        "stop_step": start_step + len(periods),
    }
