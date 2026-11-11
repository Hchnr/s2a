"""Causal opening-event features and a regularized binary predictor."""

import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from s2a.geometry import finite, matrix_to_rot6d, validate_transform


def event_features(anchor, current_opening, poses, opening):
    """Use one measured anchor and its already generated, complete H40 reference."""
    anchor, poses = validate_transform(anchor), validate_transform(poses)
    opening = np.clip(finite(opening, (40,)), 0, 1)
    current = float(current_opening)
    if anchor.shape != (4, 4) or poses.shape != (40, 4, 4) or not np.isfinite(current) or not 0 <= current <= 1:
        raise ValueError("one measured TCP/opening and forty generated targets are required")
    rotation = anchor[:3, :3]
    position = poses[:, :3, 3]
    local = (position - anchor[:3, 3]) @ rotation / 0.2
    previous = np.vstack((anchor[:3, 3], position[:-1]))
    velocity = (position - previous) @ rotation / 0.05
    remaining = (position[-1] - position) @ rotation / 0.2
    fraction = np.arange(1, 41) / 40
    anchor_p = np.broadcast_to(anchor[:3, 3] / 0.5, (40, 3))
    anchor_r = np.broadcast_to(matrix_to_rot6d(rotation), (40, 6))
    current_values = np.full(40, current)
    summaries = np.broadcast_to([opening.min(), opening[-1], opening.max() - opening.min()], (40, 3))
    z = position[:, 2] / 0.5
    nonlinear = np.column_stack(
        (
            opening**2,
            current_values**2,
            opening * current_values,
            fraction * opening,
            fraction * current_values,
            z * opening,
            z**2,
            fraction * local[:, 2],
            np.sum(local**2, axis=1),
            summaries[:, 0] * opening,
        )
    )
    result = np.column_stack(
        (
            np.ones(40),
            anchor_p,
            anchor_r,
            position / 0.5,
            matrix_to_rot6d(poses[:, :3, :3]),
            local,
            velocity,
            opening,
            current_values,
            fraction,
            summaries,
            remaining,
            nonlinear,
        )
    )
    return finite(result, (40, 44))


def loss_and_gradient(coefficients, features, closed, *, positive_weight, ridge):
    """Weighted logistic loss; the intercept is not regularized."""
    x, y, theta = finite(features), finite(closed), finite(coefficients)
    if (
        x.ndim != 2
        or x.shape[1] != 44
        or theta.shape != (44,)
        or y.shape != (len(x),)
        or not len(x)
        or np.any((y != 0) & (y != 1))
        or positive_weight <= 0
        or ridge <= 0
    ):
        raise ValueError("finite binary training rows and positive loss settings required")
    weights = np.where(y == 1, positive_weight, 1.0)
    logits = x @ theta
    loss = np.dot(weights, np.logaddexp(0, logits) - y * logits) / weights.sum()
    loss += 0.5 * ridge * np.dot(theta[1:], theta[1:])
    gradient = x.T @ (weights * (expit(logits) - y)) / weights.sum()
    gradient[1:] += ridge * theta[1:]
    return float(loss), gradient


def fit_event_model(features, closed, *, positive_weight, ridge, max_iterations, gradient_tolerance):
    x, y = finite(features), finite(closed)
    if (
        type(max_iterations) is not int
        or max_iterations <= 0
        or not np.isfinite(gradient_tolerance)
        or gradient_tolerance <= 0
    ):
        raise ValueError("finite optimizer budget and tolerance required")

    def objective(theta):
        return loss_and_gradient(theta, x, y, positive_weight=positive_weight, ridge=ridge)

    result = minimize(
        objective,
        np.zeros(44),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "gtol": gradient_tolerance / 10, "ftol": 1e-15, "maxls": 30},
    )
    loss, gradient = loss_and_gradient(result.x, x, y, positive_weight=positive_weight, ridge=ridge)
    maximum = float(np.max(np.abs(gradient)))
    if not result.success or maximum > gradient_tolerance:
        raise RuntimeError(f"event fit did not meet its convergence contract: {result.message}; gradient={maximum}")
    return result.x.copy(), dict(
        loss=loss, max_abs_gradient=maximum, iterations=int(result.nit), status=int(result.status)
    )


def predict_probability(features, coefficients):
    x, theta = finite(features), finite(coefficients, (44,))
    if x.shape[-1] != 44:
        raise ValueError("event feature width differs")
    return expit(x @ theta)


def apply_opening_event(opening, probability, threshold, *, closed_below=0.35, commanded_opening=0.1):
    raw, probability = finite(opening), finite(probability)
    if (
        raw.shape != probability.shape
        or np.any((probability < 0) | (probability > 1))
        or not 0 < threshold < 1
        or not 0 <= commanded_opening < closed_below < 1
    ):
        raise ValueError("bounded probabilities and opening event thresholds required")
    raw = np.clip(raw, 0, 1)
    return np.where((probability >= threshold) & (raw >= closed_below), commanded_opening, raw)


def group_folds(identities, seed=42):
    """Assign ten distinct training demos per task to five whole-demo folds."""
    identities = tuple(identities)
    if seed != 42 or len(identities) != len(set(identities)) or len(identities) != 30:
        raise ValueError("thirty distinct training demo identities and seed42 required")
    assignment = {}
    for task in (0, 1, 2):
        demos = [demo for index, demo in identities if index == task]
        if len(demos) != 10 or any(not isinstance(demo, str) or not demo.startswith("demo_") for demo in demos):
            raise ValueError("exactly ten named demos per task required")
        ordered = sorted(demos, key=lambda demo: (hashlib.sha256(f"{seed}|{task}|{demo}".encode()).digest(), demo))
        assignment.update({(task, demo): i % 5 for i, demo in enumerate(ordered)})
    return assignment
