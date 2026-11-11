"""Carry the actual completed first window into a new current-image request."""

import copy

import numpy as np


def first_window_inner(periods, limits):
    """Bind the last real 25 inner samples to the current eight-period boundary."""
    limits = np.asarray(limits, dtype=float)
    if (
        type(periods) is not list
        or len(periods) != 8
        or [p.get("step") for p in periods] != list(range(8))
        or limits.shape != (7, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("Eight actual consecutive periods and fixed joint limits are required")
    final = periods[-1]
    samples = final.get("actual_inner")
    if type(samples) is not list or len(samples) != 25:
        raise ValueError("All 25 actual final-period inner samples are required")
    q = np.asarray([item["q"] for item in samples], dtype=float)
    velocity = np.asarray([item["qvel"] for item in samples], dtype=float)
    times = np.asarray([item["time"] for item in samples], dtype=float)
    if (
        q.shape != (25, 7)
        or velocity.shape != (25, 7)
        or times.shape != (25,)
        or not all(np.isfinite(value).all() for value in (q, velocity, times))
        or not np.allclose(np.diff(times), 0.002, atol=1e-12, rtol=0)
        or not np.isclose(times[-1], final["endpoint_receipt"]["sim_time"], atol=1e-10, rtol=0)
    ):
        raise ValueError("The actual final inner samples must match the physical endpoint clock")
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    below = int(np.sum(np.any(margins < 0.05, axis=1)))
    outside = int(np.sum(np.any(margins < 0, axis=1)))
    if final["inner_below_buffer_samples"] != below or final["inner_outside_samples"] != outside:
        raise ValueError("The final observed inner safety counts differ")
    return dict(
        prefix_step=7,
        samples=copy.deepcopy(samples),
        limits=limits.copy(),
        minimum_margin_rad=float(margins.min()),
        below_buffer_samples=below,
        outside_samples=outside,
    )
