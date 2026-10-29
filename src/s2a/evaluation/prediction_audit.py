"""Independent Action128 contract diagnostics without importing model code."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import inverse, pose_error, rot6d_to_matrix, transform


def denormalize_action(normalized, spec):
    values = np.asarray(normalized, dtype=float).copy()
    mask = np.asarray(spec["normalization_mask"], dtype=bool)
    scale, bias = np.asarray(spec["scale"]), np.asarray(spec["bias"])
    if (
        values.ndim != 2
        or values.shape[1] != 128
        or mask.shape != (128,)
        or scale.shape != (128,)
        or bias.shape != (128,)
    ):
        raise ValueError("invalid Action128 normalization shapes")
    if not np.isfinite(values).all() or not np.isfinite(scale).all() or np.any(scale[mask] == 0):
        raise ValueError("nonfinite values or zero normalization scale")
    values[:, mask] = (values[:, mask] - bias[mask]) / scale[mask]
    return values


def audit_action_response(content, spec, *, domain, fps):
    metadata = content["action128"]
    values = np.asarray(content["action"], dtype=float)
    normalized = np.asarray(metadata["normalized_action"], dtype=float)
    expected = denormalize_action(normalized, spec)
    if values.shape != expected.shape or not np.isfinite(values).all():
        raise ValueError("nonfinite or mismatched physical action")
    mask = np.asarray(metadata["action_dim_mask"], dtype=bool)
    if mask.shape != (128,) or not mask[:10].all() or mask[10:].any():
        raise ValueError("unexpected active dimensions")
    if metadata["action_units"] != "raw_physical_or_source_units" or metadata["action_domain"] != domain:
        raise ValueError("action units/domain mismatch")
    if metadata["action_fps"] != fps or np.any(values[:, ~mask]) or np.any(normalized[:, ~mask]):
        raise ValueError("action clock or inactive slots mismatch")
    timestamps = np.asarray(metadata["action_timestamps_seconds"])
    if timestamps.shape != (len(values),) or not np.allclose(np.diff(timestamps), 1 / fps, atol=1e-6):
        raise ValueError("action timestamps differ from planned clock")
    rotations = rot6d_to_matrix(values[:, 3:9])
    first, second = values[:, 3:6], values[:, 6:9]
    return dict(
        frames=len(values),
        denormalization_max_abs=float(np.max(np.abs(values - expected))),
        rotation_mean_rad=float(np.mean(Rotation.from_matrix(rotations).magnitude())),
        rotation_max_rad=float(np.max(Rotation.from_matrix(rotations).magnitude())),
        translation_max_m=float(np.max(np.linalg.norm(values[:, :3], axis=1))),
        opening_outside_fraction=float(np.mean((values[:, 9] < 0) | (values[:, 9] > 1))),
        rot6d_column_norm_error_mean=float(np.mean(np.abs(np.linalg.norm(first, axis=1) - 1))),
        rot6d_columns_dot_abs_mean=float(np.mean(np.abs(np.sum(first * second, axis=1)))),
    )


def geometry_roundtrip(poses, h):
    """Audit actual post-action labels through local TCP deltas and source poses."""
    tcp = poses @ h
    deltas = inverse(tcp[:-1]) @ tcp[1:]
    columns = np.column_stack((deltas[:, :3, 0], deltas[:, :3, 1]))
    decoded = transform(deltas[:, :3, 3], rot6d_to_matrix(columns))
    reconstructed = tcp[:-1] @ decoded @ inverse(h)
    errors = pose_error(poses[1:], reconstructed)
    return dict(
        position_max_m=float(np.max(np.linalg.norm(errors[:, :3], axis=1))),
        rotation_max_rad=float(np.max(np.linalg.norm(errors[:, 3:], axis=1))),
    )
