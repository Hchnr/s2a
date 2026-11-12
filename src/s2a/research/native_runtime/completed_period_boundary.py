"""Bind one actually completed period to a later streaming boundary."""

import copy

import numpy as np


def completed_period_inner(period, limits):
    """Preserve all 25 final inner samples and independently recheck safety counts."""
    limits = np.asarray(limits, dtype=float)
    if (
        type(period) is not dict
        or type(period.get("step")) is not int
        or period["step"] < 0
        or type(period.get("endpoint_receipt")) is not dict
        or type(period["endpoint_receipt"].get("endpoint_step")) is not int
        or period["endpoint_receipt"].get("endpoint_step") != period["step"] + 1
        or type(period["endpoint_receipt"].get("sim_time")) not in (int, float)
        or not np.isfinite(period["endpoint_receipt"]["sim_time"])
        or type(period.get("inner_below_buffer_samples")) is not int
        or type(period.get("inner_outside_samples")) is not int
        or limits.shape != (7, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("One completed period, its next endpoint and fixed joint limits are required")
    samples = period.get("actual_inner")
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
        or not np.isclose(times[-1], period["endpoint_receipt"]["sim_time"], atol=1e-10, rtol=0)
    ):
        raise ValueError("The actual final inner samples must match the endpoint clock")
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    below = int(np.sum(np.any(margins < 0.05, axis=1)))
    outside = int(np.sum(np.any(margins < 0, axis=1)))
    if period.get("inner_below_buffer_samples") != below or period.get("inner_outside_samples") != outside:
        raise ValueError("The final observed inner safety counts differ")
    return dict(
        prefix_step=period["step"],
        samples=copy.deepcopy(samples),
        limits=limits.copy(),
        minimum_margin_rad=float(margins.min()),
        below_buffer_samples=below,
        outside_samples=outside,
    )
