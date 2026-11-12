"""Explicitly transform already generated gripper intent for development probes."""

import numpy as np

from s2a.references.finite_horizon_window import remaining_h40


def amplified_opening_h40(poses, openings, *, consumed, lead_steps, gain):
    """Preserve pose targets while leading and amplifying an H40 opening reference."""
    if type(lead_steps) is not int or not 0 <= lead_steps <= 40 or not np.isfinite(gain) or gain < 1:
        raise ValueError("A bounded integer lead and finite gain at least one are required")
    poses, openings = np.asarray(poses, dtype=float), np.asarray(openings, dtype=float)
    window_poses, _, metadata = remaining_h40(poses, openings, consumed=consumed)
    amplified = np.clip(1 - gain * (1 - openings), 0, 1)
    shifted = np.concatenate((amplified[lead_steps:], np.repeat(amplified[-1:], lead_steps)))
    _, window_openings, _ = remaining_h40(poses, shifted, consumed=consumed)
    generated = max(0, 40 - consumed - lead_steps)
    metadata.update(
        opening_lead_steps=lead_steps,
        opening_gain=float(gain),
        opening_generated_endpoints=generated,
        opening_terminal_hold_endpoints=40 - generated,
    )
    return window_poses, window_openings, metadata
