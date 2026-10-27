"""Offline temporal cross terms for one sequence, before pooling requests."""

import numpy as np

from s2a.geometry import finite


def temporal_terms(increments):
    error = finite(increments)
    if error.ndim != 2 or error.shape[1] != 3 or len(error) == 0:
        raise ValueError("One nonempty sequence of base-frame xyz errors required")
    diagonal = np.cumsum(np.sum(error**2, axis=1))
    cumulative_energy = np.sum(np.cumsum(error, axis=0) ** 2, axis=1)
    return dict(local_diagonal_cumulative_m2=diagonal, local_temporal_cross_m2=cumulative_energy - diagonal)
