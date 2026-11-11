"""Causal postprocessing of generated TCP references; no model or file I/O."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, transform, validate_transform


def _local_coordinates(anchor, poses, pi_margin):
    anchor, poses = validate_transform(anchor), validate_transform(poses)
    if anchor.shape != (4, 4) or poses.ndim != 3 or len(poses) != 40:
        raise ValueError("one anchor and exactly forty SE(3) endpoints required")
    rotation = anchor[:3, :3].T
    local_position = (rotation @ (poses[:, :3, 3] - anchor[:3, 3]).T).T
    local_rotation = Rotation.from_matrix(rotation @ poses[:, :3, :3]).as_rotvec()
    if np.any(np.linalg.norm(local_rotation, axis=1) >= np.pi - pi_margin):
        raise ValueError("rotation logarithm is ambiguous near pi")
    return local_position, local_rotation


def features(anchor, poses, opening, current_opening, config):
    """Features depend only on current measurements and a generated reference."""
    opening = finite(opening, (40,))
    current = float(current_opening)
    if not np.isfinite(current) or not 0 <= current <= 1:
        raise ValueError("measured opening must lie in [0, 1]")
    u, v = _local_coordinates(anchor, poses, config["pi_margin_rad"])
    fraction = np.arange(1, 41) / 40
    pose = np.column_stack((u / config["position_scale_m"], v / config["rotation_scale_rad"], fraction))
    aperture = np.clip(opening, 0, 1)
    gripper = np.column_stack(
        (
            np.ones(40),
            aperture,
            np.maximum(aperture - 0.25, 0),
            np.maximum(aperture - 0.5, 0),
            np.maximum(aperture - 0.75, 0),
            np.full(40, current),
            fraction,
        )
    )
    return pose, gripper


def training_targets(anchor, predicted, truth, config):
    """Offline labels are aligned cumulative endpoint residuals."""
    u, v = _local_coordinates(anchor, predicted, config["pi_margin_rad"])
    target_u, target_v = _local_coordinates(anchor, truth, config["pi_margin_rad"])
    return np.column_stack(((target_u - u) / config["position_scale_m"], (target_v - v) / config["rotation_scale_rad"]))


def fit_pose(pose_features, pose_targets, ridge):
    x, y = finite(pose_features), finite(pose_targets)
    if x.ndim != 2 or x.shape[1] != 7 or y.shape != (len(x), 6) or len(x) == 0 or ridge <= 0:
        raise ValueError("invalid pose training data or ridge")
    return np.linalg.solve(x.T @ x / len(x) + ridge * np.eye(7), x.T @ y / len(x))


def fit_opening(gripper_features, opening, truth, penalty, time_limit):
    """Minimize mean absolute error plus L1 coefficient penalty in one LP."""
    x, opening, truth = finite(gripper_features), finite(opening), finite(truth)
    if (
        x.ndim != 2
        or x.shape[1] != 7
        or opening.shape != (len(x),)
        or truth.shape != (len(x),)
        or len(x) == 0
        or np.any((truth < 0) | (truth > 1))
        or penalty <= 0
        or time_limit <= 0
    ):
        raise ValueError("invalid opening training data or LP settings")
    y = truth - np.clip(opening, 0, 1)
    xx, eye = sparse.csr_matrix(x), sparse.eye(len(x), format="csr")
    inequalities = sparse.vstack((sparse.hstack((xx, -xx, -eye)), sparse.hstack((-xx, xx, -eye))), format="csr")
    objective = np.r_[np.full(14, penalty), np.full(len(x), 1 / len(x))]
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.r_[y, -y],
        bounds=(0, None),
        method="highs",
        options={"time_limit": time_limit},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"opening LP did not finish successfully: {result.message}")
    coefficients = result.x[:7] - result.x[7:14]
    residual = x @ coefficients - y
    actual = float(np.mean(abs(residual)) + penalty * np.sum(abs(coefficients)))
    if abs(actual - result.fun) > 1e-7:
        raise ValueError("opening LP objective is inconsistent with returned coefficients")
    dual = finite(result.ineqlin.marginals, (2 * len(x),))
    return coefficients, {
        "objective": actual,
        "iterations": int(result.nit),
        "status": int(result.status),
        "inequality_dual": dual.tolist(),
    }


def _cap_norm(values, limits):
    norm = np.linalg.norm(values, axis=1)
    return values * np.minimum(1, limits / np.maximum(norm, 1e-30))[:, None]


def apply_calibration(anchor, poses, opening, current_opening, pose_coefficients, opening_coefficients, config):
    """Apply fixed bounded corrections in the measured anchor frame."""
    anchor, poses = validate_transform(anchor), validate_transform(poses)
    pose_coefficients = finite(pose_coefficients, (7, 6))
    opening_coefficients = finite(opening_coefficients, (7,))
    x, g = features(anchor, poses, opening, current_opening, config)
    u, v = _local_coordinates(anchor, poses, config["pi_margin_rad"])
    fraction = np.arange(1, 41) / 40
    correction = x @ pose_coefficients
    dp = _cap_norm(correction[:, :3] * config["position_scale_m"], fraction * config["position_cap_m"])
    dr = _cap_norm(correction[:, 3:] * config["rotation_scale_rad"], fraction * config["rotation_cap_rad"])
    new_position = anchor[:3, 3] + (anchor[:3, :3] @ (u + dp).T).T
    new_rotation = anchor[:3, :3] @ Rotation.from_rotvec(v + dr).as_matrix()
    delta_opening = np.clip(g @ opening_coefficients, -config["opening_cap"], config["opening_cap"])
    new_opening = np.clip(np.clip(opening, 0, 1) + delta_opening, 0, 1)
    return transform(new_position, new_rotation), new_opening
