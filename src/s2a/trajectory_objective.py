"""Offline physical endpoint objectives conditional on fixed predicted rotations.

These gradients are with respect to physical local translations/openings, before
any readout normalization chain rule. They are not gradients through SO(3), the
sampler, the controller, contact dynamics or the robot.
"""

import numpy as np

from s2a.geometry import finite


def position_loss(local_translation, before_rotation, initial_position, desired_endpoints, endpoint_weights):
    """Weighted squared endpoint error and exact local-translation gradient.

    Rotations must come from the frozen predicted trajectory, not future truth.
    Caller chooses weights explicitly (e.g. pooled H8/H16/H40), independently of
    validation outcomes. Each call handles one uninterrupted trajectory.
    """
    delta, rotation = finite(local_translation), finite(before_rotation)
    if delta.ndim != 2 or delta.shape[1] != 3 or len(delta) == 0:
        raise ValueError("Nonempty local xyz sequence required")
    n = len(delta)
    if rotation.shape != (n, 3, 3):
        raise ValueError("One fixed before-step rotation per increment required")
    if not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-9, rtol=0) or not np.allclose(
        np.linalg.det(rotation), 1, atol=1e-9, rtol=0
    ):
        raise ValueError("Proper SO(3) rotations required")
    initial, desired, weights = finite(initial_position), finite(desired_endpoints), finite(endpoint_weights)
    if initial.shape != (3,) or desired.shape != (n, 3) or weights.shape != (n,):
        raise ValueError("Initial xyz, complete endpoint xyz and explicit weights required")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Nonnegative endpoint weights with positive sum required")
    weights = weights / weights.sum()
    predicted = initial + np.cumsum(np.einsum("nij,nj->ni", rotation, delta), axis=0)
    error = predicted - desired
    costate = 2 * np.cumsum((weights[:, None] * error)[::-1], axis=0)[::-1]
    gradient = np.einsum("nji,nj->ni", rotation, costate)
    return float(np.sum(weights * np.sum(error**2, axis=1))), gradient


def opening_loss(predicted, desired, endpoint_weights):
    """Physical unclipped L1 objective; zero subgradient at exact agreement.

    No clipping in training: clipping would create zero gradients outside the
    legal range. Deployment still owns its existing [0,1] command constraint.
    """
    predicted, desired, weights = finite(predicted), finite(desired), finite(endpoint_weights)
    if (
        predicted.ndim != 1
        or not len(predicted)
        or desired.shape != predicted.shape
        or weights.shape != predicted.shape
    ):
        raise ValueError("Equal nonempty opening/target/weight vectors required")
    if (weights < 0).any() or weights.sum() <= 0 or (desired < 0).any() or (desired > 1).any():
        raise ValueError("Legal desired openings and nonnegative positive-sum weights required")
    weights = weights / weights.sum()
    error = predicted - desired
    return float(np.sum(weights * abs(error))), weights * np.sign(error)
