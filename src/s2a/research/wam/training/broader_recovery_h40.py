"""Window-level clipping and six-source separation for recovery data."""

import numpy as np

from s2a.research.wam.training.clean.clean_training_data import DOMAIN
from s2a.research.wam.training.recovery_event_coverage import validation_identity_check


def clipping_flags(physical_action128, normalization):
    """Reproduce normalization31 FP32 clipping flags for arbitrary row counts."""
    data = np.asarray(physical_action128, dtype=np.float64)
    transform = normalization["transform"]
    mask = np.asarray(transform["normalization_mask"])
    scale = np.asarray(transform["scale"], dtype=np.float64)
    bias = np.asarray(transform["bias"], dtype=np.float64)
    low, high = transform["clip_min"], transform["clip_max"]
    if (
        normalization["domain"] != DOMAIN
        or type(normalization["normalization_id"]) is not int
        or normalization["normalization_id"] != 31
        or type(normalization["projection_id"]) is not int
        or normalization["projection_id"] != 0
        or set(transform) != {"normalization_mask", "scale", "bias", "clip_min", "clip_max"}
        or data.ndim != 2
        or data.shape[1] != 128
        or not np.isfinite(data).all()
        or np.any(data[:, 10:] != 0)
        or mask.dtype != np.bool_
        or mask.shape != (128,)
        or np.any(mask[10:])
        or scale.shape != (128,)
        or bias.shape != (128,)
        or not np.isfinite(scale).all()
        or not np.isfinite(bias).all()
        or np.ndim(low) != 0
        or np.ndim(high) != 0
        or not np.isfinite([low, high]).all()
        or low >= high
    ):
        raise ValueError("The frozen active10 normalization contract is required")
    values = data.astype(np.float32)
    multiplied = np.multiply(values[:, mask], scale.astype(np.float32)[mask], dtype=np.float32)
    transformed = np.add(multiplied, bias.astype(np.float32)[mask], dtype=np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("FP32 normalized labels overflowed")
    flags = np.zeros((len(data), 10), dtype=bool)
    flags[:, np.flatnonzero(mask[:10])] = (transformed < np.float32(low)) | (transformed > np.float32(high))
    return flags


def validate_six_sources(groups, training, preflight, validation, verification):
    """Check six unique training sources against synchronized validation/fresh metadata."""
    keys = {(row["task_index"], row["demo"]) for row in groups}
    hashes = {row["initial_state_sha256"] for row in groups}
    if (
        len(groups) != 6
        or len(keys) != 6
        or len(hashes) != 6
        or any(sum(row["task_index"] == task for row in groups) != 2 for task in range(3))
    ):
        raise ValueError("Six unique training sources, two per task, are required")
    for index in (0, 3):
        validation_identity_check(groups[index : index + 3], training, preflight, validation, verification)
    return dict(
        training_groups=6,
        synchronized_validation_sources=6,
        fresh_metadata_sources=12,
        identity_sets_disjoint=True,
        unknown_near_duplicates_excluded=False,
        validation_future_labels_read=False,
        fresh_future_labels_read=False,
    )
