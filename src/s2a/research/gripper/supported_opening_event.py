"""Constrain early opening events to a future event in the generated chunk."""

import numpy as np

from s2a.geometry import finite


def future_closure_support(opening, *, steps=16, closed_below=0.35):
    """Mark endpoints supported by a later low opening already in the H40 chunk.

    This reads generated reference values, not future robot measurements. The
    current endpoint and values beyond the generated chunk are never support.
    """
    raw = finite(opening)
    if (
        raw.ndim < 1
        or raw.shape[-1] != 40
        or type(steps) is not int
        or not 1 <= steps < 40
        or not np.isfinite(closed_below)
        or not 0 < closed_below < 1
        or np.any((raw < 0) | (raw > 1))
    ):
        raise ValueError("bounded H40 opening, finite threshold and 1..39 future steps required")
    support = np.zeros(raw.shape, dtype=bool)
    for offset in range(1, steps + 1):
        support[..., :-offset] |= raw[..., offset:] < closed_below
    return support


def supported_probability(opening, probability, *, steps=16, closed_below=0.35):
    """Remove predictions unsupported by a later event in the same reference."""
    raw, probability = finite(opening), finite(probability)
    if raw.shape != probability.shape or np.any((probability < 0) | (probability > 1)):
        raise ValueError("matching bounded opening and probability arrays required")
    mask = future_closure_support(raw, steps=steps, closed_below=closed_below)
    return np.where(mask, probability, 0.0)
