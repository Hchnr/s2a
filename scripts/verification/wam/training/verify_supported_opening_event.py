"""Scalar reconstruction of generated-reference support and event gating."""

import numpy as np

from scripts.verification.wam.training.verify_causal_opening_event import verify_selection


def verify_support(opening, support, *, steps, closed_below):
    """Check each endpoint against its exact later generated-reference suffix."""
    raw = np.asarray(opening, dtype=float)
    proposed = np.asarray(support)
    if (
        raw.ndim != 2
        or raw.shape[1] != 40
        or proposed.shape != raw.shape
        or proposed.dtype != np.dtype(bool)
        or not np.isfinite(raw).all()
        or np.any((raw < 0) | (raw > 1))
        or type(steps) is not int
        or not 1 <= steps < 40
        or not np.isfinite(closed_below)
        or not 0 < closed_below < 1
    ):
        raise ValueError("independent support inputs differ")
    expected = np.zeros(raw.shape, dtype=bool)
    for row in range(len(raw)):
        for endpoint in range(40):
            expected[row, endpoint] = any(
                raw[row, later] < closed_below for later in range(endpoint + 1, min(endpoint + steps + 1, 40))
            )
    if not np.array_equal(expected, proposed):
        raise ValueError("independent generated-reference support differs")
    return dict(queries=len(raw), endpoints=raw.size, supported_endpoints=int(expected.sum()))


def verify_supported_selection(raw, truth, probability, tasks, support, config, selection):
    """Independently check support, then re-evaluate all registered thresholds."""
    checked = verify_support(raw, support, steps=16, closed_below=config["closed_below"])
    masked_probability = np.where(np.asarray(support), np.asarray(probability, dtype=float), 0.0)
    selection_check = verify_selection(raw, truth, masked_probability, tasks, config, selection)
    return dict(support=checked, selection=selection_check)
