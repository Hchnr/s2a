"""Fixed local condition basis with original state statistics and affine action."""

import numpy as np

from s2a.joint_wrench_condition_forward import nominal, raw_condition, targets, with_history

__all__ = ["design", "nominal", "raw_condition", "targets", "with_history", "predict", "condition_features"]
BLOCKS = (
    tuple(range(7)) + tuple(range(21, 28)),
    tuple(range(28, 37)),
    tuple(range(37, 43)),
    tuple(range(43, 51)),
    tuple(range(51, 71)),
    tuple(range(71, 83)),
    tuple(range(83, 89)),
)


def condition_features(raw, parameters):
    raw = np.asarray(raw, dtype=float)
    mean = np.asarray(parameters["state_mean"], dtype=float)
    scale = np.asarray(parameters["state_scale"], dtype=float)
    centers = np.asarray(parameters["anchor_centers"], dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 89 or not len(raw):
        raise ValueError("Nonempty Nx89 current condition required")
    if mean.shape != (89,) or scale.shape != (89,) or centers.shape != (66, 89):
        raise ValueError("Fixed89 statistics and66 raw training centers required")
    if not all(np.isfinite(v).all() for v in (raw, mean, scale, centers)) or np.any(scale <= 0):
        raise ValueError("Finite inputs and positive scales required")
    z, c = (raw - mean) / scale, (centers - mean) / scale
    distance = sum(np.mean((z[:, None, block] - c[None, :, block]) ** 2, axis=2) for block in BLOCKS) / len(BLOCKS)
    kernel = (1 + distance) ** -2
    return np.c_[np.ones(len(z)), z, kernel]


def design(data, parameters):
    phi = condition_features(raw_condition(data), parameters)
    action = np.asarray(data["action_goal_delta"], dtype=float)[:, :3]
    mean = np.asarray(parameters["translation_mean"], dtype=float)
    scale = np.asarray(parameters["translation_scale"], dtype=float)
    if (
        mean.shape != (3,)
        or scale.shape != (3,)
        or not all(np.isfinite(v).all() for v in (action, mean, scale))
        or np.any(scale <= 0)
    ):
        raise ValueError("Finite translation and positive fixed3 scales required")
    u = (action - mean) / scale
    return np.c_[phi, *(phi * u[:, i, None] for i in range(3))]


def predict(data, parameters):
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (624, 12) or not np.isfinite(coefficients).all():
        raise ValueError("Finite624x12 coefficients required")
    return nominal(data, parameters) + design(data, parameters) @ coefficients
