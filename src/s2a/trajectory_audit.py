"""Offline model-target diagnostics against the matching published labels.

These are prediction diagnostics, not forward-control or task-success scores.
All poses are integrated in SE(3); terminal placeholders are excluded by mask.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, integrate_local, pose_error, rot6d_to_matrix


def trajectory_diagnostics(predicted, reference, mask):
    predicted, reference = finite(predicted), finite(reference)
    mask = np.asarray(mask, dtype=bool)
    if predicted.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError("prediction, reference and mask shapes differ")
    if predicted.ndim != 2 or predicted.shape[1] < 10 or len(predicted) == 0:
        raise ValueError("nonempty physical action trajectories are required")
    valid = mask[:, :9].all(axis=1)
    grip_valid = mask[:, 9]
    result = dict(steps=len(predicted), pose_steps=int(valid.sum()), opening_steps=int(grip_valid.sum()))
    if valid.any():
        pred, ref = predicted[valid], reference[valid]
        pr, rr = rot6d_to_matrix(pred[:, 3:9]), rot6d_to_matrix(ref[:, 3:9])
        rotation_error = Rotation.from_matrix(pr @ rr.swapaxes(-1, -2)).magnitude()
        result.update(
            translation_rmse_m=float(np.sqrt(np.mean(np.sum((pred[:, :3] - ref[:, :3]) ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(rotation_error**2))),
            predicted_translation_mean_m=float(np.mean(np.linalg.norm(pred[:, :3], axis=1))),
            reference_translation_mean_m=float(np.mean(np.linalg.norm(ref[:, :3], axis=1))),
            predicted_rotation_mean_rad=float(np.mean(Rotation.from_matrix(pr).magnitude())),
            reference_rotation_mean_rad=float(np.mean(Rotation.from_matrix(rr).magnitude())),
            raw_column_norm_error_max=float(np.max(abs(np.linalg.norm(pred[:, 3:9].reshape(-1, 2, 3), axis=-1) - 1))),
            raw_column_dot_max=float(np.max(abs(np.sum(pred[:, 3:6] * pred[:, 6:9], axis=1)))),
        )
    if grip_valid.any():
        pred, ref = predicted[grip_valid, 9], reference[grip_valid, 9]
        result.update(
            opening_outside_fraction=float(np.mean((pred < 0) | (pred > 1))),
            opening_mae_raw=float(np.mean(abs(pred - ref))),
            opening_mae_clipped=float(np.mean(abs(np.clip(pred, 0, 1) - ref))),
        )
    prefix = int(np.cumprod(valid).sum())
    result["valid_prefix_steps"] = prefix
    if prefix:
        p = integrate_local(np.eye(4), predicted[:prefix, :9])
        r = integrate_local(np.eye(4), reference[:prefix, :9])
        error = pose_error(r[1:], p[1:])
        result.update(
            integrated_position_rmse_m=float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
            integrated_rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
            integrated_final_position_error_m=float(np.linalg.norm(error[-1, :3])),
            integrated_final_rotation_error_rad=float(np.linalg.norm(error[-1, 3:])),
        )
    return result
