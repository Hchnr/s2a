"""Plan057 linear maps from terminal coefficient changes to endpoint errors."""

import numpy as np
from scipy.sparse.linalg import LinearOperator


def operators(cache):
    phi = np.asarray(cache["phi"], dtype=float)
    if phi.ndim != 3 or phi.shape[1] != 40 or phi.shape[2] < 1 or not len(phi):
        raise ValueError("Complete H40 feature sequences required")
    n, horizon, width = phi.shape
    rotation = np.asarray(cache["before_rotation"])
    if rotation.shape != (n, horizon, 3, 3):
        raise ValueError("Fixed predicted before-step rotations required")
    scale, bias = cache["action_scale"], cache["action_bias"]
    if np.shape(scale) != (10,) or np.shape(bias) != (10,) or (scale == 0).any():
        raise ValueError("Invertible active10 action transform required")
    weights = np.asarray(cache["endpoint_weights"])
    if weights.shape != (40,) or (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-12):
        raise ValueError("Explicit normalized H40 endpoint weights required")
    flattened = phi.reshape(-1, width)
    root = np.sqrt(weights / n)[:, None] / 0.05

    def position_forward(delta):
        local = (flattened @ np.asarray(delta).reshape(width, 3)).reshape(n, horizon, 3) / scale[:3]
        world = np.einsum("ntij,ntj->nti", rotation, local)
        return (np.cumsum(world, axis=1) * root).ravel()

    def position_reverse(endpoint):
        endpoint = np.asarray(endpoint).reshape(n, horizon, 3) * root
        costate = np.cumsum(endpoint[:, ::-1], axis=1)[:, ::-1]
        local = np.einsum("ntji,ntj->nti", rotation, costate) / scale[:3]
        return (flattened.T @ local.reshape(-1, 3)).ravel()

    position = LinearOperator(
        (n * horizon * 3, width * 3), matvec=position_forward, rmatvec=position_reverse, dtype=float
    )
    opening = LinearOperator(
        (n * horizon, width),
        matvec=lambda d: (flattened @ d) / scale[9],
        rmatvec=lambda e: (flattened.T @ e) / scale[9],
        dtype=float,
    )
    baseline = (cache["raw_clean"] + (flattened @ cache["coefficients052"]).reshape(n, horizon, 10) - bias) / scale
    predicted = cache["initial_position"][:, None, :] + np.cumsum(
        np.einsum("ntij,ntj->nti", rotation, baseline[:, :, :3]),
        axis=1,
    )
    position_offset = ((predicted - cache["desired_positions"]) * root).ravel()
    opening_offset = (baseline[:, :, 9] - cache["desired_opening"]).ravel()
    opening_weights = np.tile(weights / n, n)
    return position, position_offset, opening, opening_offset, opening_weights
