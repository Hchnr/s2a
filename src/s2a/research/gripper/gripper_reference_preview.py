"""Finite preview of already generated opening references, no future sensors."""

import numpy as np


def opening_preview(generated_openings, current_index, *, preview_steps):
    """Advance only a nondecreasing opening segment, retaining its exact value.

    Returns a command-side opening reference and provenance. This does not
    modify the desired state trajectory or certify that early release is safe.
    A segment containing any decrease keeps the current reference unchanged.
    """
    values = np.asarray(generated_openings, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Requires finite generated openings in [0,1]")
    if type(current_index) is not int or not 0 <= current_index < len(values):
        raise ValueError("Current index must address an available generated endpoint")
    if type(preview_steps) is not int or preview_steps < 0:
        raise ValueError("Explicit nonnegative integer preview required")
    stop = min(current_index + preview_steps, len(values) - 1)
    segment = values[current_index : stop + 1]
    advances = bool(np.all(np.diff(segment) >= -1e-12) and values[stop] > values[current_index])
    selected = stop if advances else current_index
    return dict(
        reference=float(values[selected]),
        original_reference=float(values[current_index]),
        selected_index=selected,
        current_index=current_index,
        available_end_index=len(values) - 1,
        advanced=advances,
        preview_steps=preview_steps,
    )
