"""196 measured finger/wrench/motor history; no labels or future observations."""

import numpy as np


def contact_history(data, lags=3):
    if type(lags) is not int or lags < 1:
        raise ValueError("Positive integer history length required")
    q = np.asarray(data["x_finger_qpos"], dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or not len(q):
        raise ValueError("Nonempty measured two-finger positions required")
    count = len(q)
    fields = {}
    for key, width in [
        ("x_finger_qpos", 2),
        ("x_finger_qvel", 2),
        ("x_motor_before", 2),
        ("x_current_wrench_world", 6),
    ]:
        value = np.asarray(data[key], dtype=float)
        if value.shape != (count, width) or not np.isfinite(value).all():
            raise ValueError("Finite aligned current observations required: " + key)
        fields[key] = value
    available = np.asarray(data["x_previous_command_available"])
    if available.shape != (count,) or not np.isin(available, [0, 1]).all() or available[0]:
        raise ValueError("Explicit episode start and binary previous-command availability required")
    result = np.zeros((count, 13 * lags))
    start = 0
    for i in range(count):
        if not available[i]:
            start = i
        for lag in range(1, lags + 1):
            previous = i - lag
            if previous < start:
                continue
            row = np.r_[
                q[i] - q[previous],
                fields["x_finger_qvel"][previous],
                fields["x_current_wrench_world"][i] - fields["x_current_wrench_world"][previous],
                fields["x_motor_before"][i] - fields["x_motor_before"][previous],
                1.0,
            ]
            result[i, 13 * (lag - 1) : 13 * lag] = row
    return result
