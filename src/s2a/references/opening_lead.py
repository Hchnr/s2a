"""Apply an explicit lead to an already generated finite opening reference."""

import numpy as np

from s2a.references.finite_horizon_window import remaining_h40


def opening_lead_h40(poses, openings, *, consumed, lead_steps):
    """Keep pose timing and expose a bounded, lead-shifted H40 opening preview."""
    if type(lead_steps) is not int or not 0 <= lead_steps <= 40:
        raise ValueError("Opening lead must be an integer within the generated horizon")
    poses, openings = np.asarray(poses, dtype=float), np.asarray(openings, dtype=float)
    # This validates both inputs and establishes the unchanged pose/hold contract.
    window_poses, _, metadata = remaining_h40(poses, openings, consumed=consumed)
    shifted = np.concatenate((openings[lead_steps:], np.repeat(openings[-1:], lead_steps)))
    _, window_openings, _ = remaining_h40(poses, shifted, consumed=consumed)
    generated = max(0, 40 - consumed - lead_steps)
    metadata.update(
        opening_lead_steps=lead_steps,
        opening_generated_endpoints=generated,
        opening_terminal_hold_endpoints=40 - generated,
    )
    return window_poses, window_openings, metadata


def score_opening_against_references(measured, original, shifted):
    """Keep actual aperture errors relative to both model and modified references."""
    values = np.asarray([measured, original, shifted], dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Measured and reference openings must be finite values within [0, 1]")
    return dict(
        measured=float(measured),
        original_reference=float(original),
        lead_reference=float(shifted),
        original_error=float(measured - original),
        lead_error=float(measured - shifted),
    )
