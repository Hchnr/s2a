"""Reconstruct hard joint-preview constraint slacks without running a solver."""

import numpy as np

from s2a.geometry import finite


def audit_joint_preview_constraints(
    rollout,
    commands,
    previous,
    interface,
    lower,
    upper,
    *,
    past_inner_error_rad,
    terminal_normals,
    braking_time_s=0.1,
):
    """Audit a supplied frozen rollout; this does not establish physical safety.

    Positive slacks satisfy the constraint. Group arrays preserve command
    (H, 6), inner-state (25H, n), endpoint (H, n), or terminal-row shape.
    First/future minima exclude terminal rows, whose constraint applies only
    at the horizon's final endpoint. Empty groups have minimum ``None``.
    """
    commands = finite(commands)
    if commands.ndim != 2 or commands.shape[1] != 6 or not len(commands):
        raise ValueError("Nonempty H by six command array required")
    horizon = len(commands)
    previous = finite(previous, (6,))
    input_min, input_max = finite(interface.input_min, (6,)), finite(interface.input_max, (6,))
    if np.any(input_min >= input_max):
        raise ValueError("Strictly increasing command ranges required")
    lower = finite(lower)
    if lower.ndim != 1 or not len(lower):
        raise ValueError("Nonempty joint limit vectors required")
    joint_count = len(lower)
    upper = finite(upper, (joint_count,))
    if np.any(lower >= upper):
        raise ValueError("Strictly increasing joint ranges required")
    profile = finite(past_inner_error_rad, (25, joint_count))
    if np.any(profile < 0) or np.any(np.diff(profile, axis=0) < 0):
        raise ValueError("Nonnegative nondecreasing inner-time error profile required")
    braking_time = finite(braking_time_s, ())
    if braking_time <= 0:
        raise ValueError("Positive finite braking time required")
    normals = finite(terminal_normals)
    if normals.shape == (0,):
        normals = np.empty((0, joint_count))
    if normals.ndim != 2 or normals.shape[1] != joint_count:
        raise ValueError("Terminal direction rows must match the joint count")
    inner_q = finite(rollout["inner_joint_position"], (25 * horizon, joint_count))
    endpoint_q = finite(rollout["joint_position"], (horizon, joint_count))
    endpoint_velocity = finite(rollout["joint_velocity"], (horizon, joint_count))

    position_margin = np.tile(0.05 + profile.max(axis=0), (25 * horizon, 1))
    position_margin[:25] = 0.05 + profile
    change = np.diff(np.vstack((previous, commands)), axis=0)
    braking_q = endpoint_q + float(braking_time) * endpoint_velocity
    raw_groups = {
        "box_lower": (commands - input_min, 1e-12, 1),
        "box_upper": (input_max - commands, 1e-12, 1),
        "rate_lower": (0.25 + change, 1e-12, 1),
        "rate_upper": (0.25 - change, 1e-12, 1),
        "inner_lower": (inner_q - lower - position_margin, 1e-10, 25),
        "inner_upper": (upper - position_margin - inner_q, 1e-10, 25),
        "braking_lower": (braking_q - lower - 0.05, 1e-10, 1),
        "braking_upper": (upper - 0.05 - braking_q, 1e-10, 1),
        "terminal": (-normals @ endpoint_velocity[-1], 1e-10, None),
    }
    activity_tolerance = 1e-8
    groups, first_minima, future_minima = {}, {}, {}

    def minimum(array):
        return float(np.min(array)) if array.size else None

    for name, (slack, tolerance, first_count) in raw_groups.items():
        # Finite inputs can still overflow during a constraint reconstruction.
        slack = finite(slack).copy()
        groups[name] = dict(
            slack=slack,
            minimum=minimum(slack),
            active_indices=np.argwhere(slack <= activity_tolerance),
            violated_indices=np.argwhere(slack < -tolerance),
            tolerance=tolerance,
        )
        if first_count is not None:
            first_minima[name] = minimum(slack[:first_count])
            future_minima[name] = minimum(slack[first_count:])
    return dict(
        horizon=horizon,
        joint_count=joint_count,
        position_margin_rad=position_margin,
        groups=groups,
        first_period_minima=first_minima,
        future_period_minima=future_minima,
        feasible=all(not len(group["violated_indices"]) for group in groups.values()),
        activity_tolerance=activity_tolerance,
    )
