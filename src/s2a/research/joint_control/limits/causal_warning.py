"""Causal joint-limit warnings and an explicitly retrospective endpoint audit."""

import numpy as np

from s2a.geometry import finite

RULES = ("buffer", "static_0p10", "velocity_0p2s")
BUFFER_RAD = 0.05
LOOKAHEAD_SECONDS = 0.2
HORIZON_STEPS = 4


def joint_warnings(q, velocity, limits):
    """Score only current joint measurements; an alarm is not a safety guarantee."""
    q = finite(q, (7,))
    velocity = finite(velocity, (7,))
    limits = finite(limits, (7, 2))
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Joint limits must be strictly increasing")
    margin = np.stack((q - limits[:, 0], limits[:, 1] - q), axis=1)
    outward = np.stack((np.maximum(-velocity, 0), np.maximum(velocity, 0)), axis=1)
    risks = {
        "buffer": (margin, BUFFER_RAD),
        "static_0p10": (margin, 0.10),
        "velocity_0p2s": (margin - LOOKAHEAD_SECONDS * outward, BUFFER_RAD),
    }
    result = {}
    for rule, (projected, threshold) in risks.items():
        joint, side = np.unravel_index(np.argmin(projected), projected.shape)
        result[rule] = {
            "alarm": bool(projected[joint, side] <= threshold),
            "joint_index": int(joint),
            "side": ("lower", "upper")[side],
            "margin_rad": float(margin[joint, side]),
            "outward_velocity_rad_s": float(outward[joint, side]),
            "projected_margin_rad": float(projected[joint, side]),
            "threshold_rad": threshold,
        }
    return result


def summarize_rule(flags, q, limits):
    """Compare warnings with recorded future endpoints, never part of a warning."""
    q = np.asarray(q, dtype=float)
    limits = finite(limits, (7, 2))
    flags = np.asarray(flags, dtype=bool)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) != len(flags) + 1 or not np.isfinite(q).all():
        raise ValueError("One finite terminal state after the warning sequence is required")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Joint limits must be strictly increasing")
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q).min(axis=1)
    outside = margins < 0
    observed = np.flatnonzero(outside)
    first_outside = None if not len(observed) else int(observed[0])
    alerts = np.flatnonzero(flags)
    eligible = margins[:-1] >= BUFFER_RAD
    before_outside = np.arange(len(flags)) < (len(q) if first_outside is None else first_outside)
    early = np.flatnonzero(flags & eligible & before_outside)
    confusion = {"alarm_event": 0, "alarm_no_event": 0, "no_alarm_event": 0, "no_alarm_no_event": 0}
    censored = 0
    for t, flag in enumerate(flags):
        if not eligible[t]:
            continue
        if t + HORIZON_STEPS >= len(q):
            censored += 1
            continue
        event = bool(np.any(outside[t + 1 : t + 1 + HORIZON_STEPS]))
        key = ("alarm" if flag else "no_alarm") + ("_event" if event else "_no_event")
        confusion[key] += 1
    first_early = None if not len(early) else int(early[0])
    return {
        "first_outside_endpoint": first_outside,
        "outside_endpoints": int(outside.sum()),
        "first_alarm_endpoint": None if not len(alerts) else int(alerts[0]),
        "alarm_periods": int(flags.sum()),
        "first_alarm_while_buffer_safe_before_first_outside": first_early,
        "lead_periods_to_first_outside": (
            None if first_early is None or first_outside is None else first_outside - first_early
        ),
        "fully_observed_safe_buffer_windows": confusion,
        "safe_buffer_windows_censored": censored,
    }
