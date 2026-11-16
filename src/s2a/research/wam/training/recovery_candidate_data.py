"""Pure identities and clipping diagnostics for measured recovery windows."""

import hashlib
import json
import re
from collections import Counter

import numpy as np

from s2a.research.wam.training.clean.clean_training_data import SourceIdentity, normalize_active10

SEED_NAMESPACE = "s2a.recovery-candidate.v1"
_IDENTITY_FIELDS = frozenset(
    (
        "task_index",
        "demo",
        "initial_state_sha256",
        "case_index",
        "variant",
        "anchor_step",
        "image_sha256",
        "trajectory_sha256",
    )
)


def candidate_seed(identity, root):
    """Derive a candidate seed from a complete, variant-specific immutable identity."""
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_FIELDS:
        raise ValueError("The complete recovery-window identity is required")
    SourceIdentity(identity["task_index"], identity["demo"], identity["initial_state_sha256"])
    if (
        type(identity["case_index"]) is not int
        or identity["case_index"] < 0
        or identity["variant"] not in ("zero", "base_x", "base_y", "base_yaw")
        or type(identity["anchor_step"]) is not int
        or identity["anchor_step"] < 0
        or type(root) is not int
        or root not in (101, 102)
        or any(
            not isinstance(identity[key], str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key])
            for key in ("image_sha256", "trajectory_sha256")
        )
    ):
        raise ValueError("The case, variant, anchor, image, trajectory and root must be explicit")
    payload = json.dumps(
        dict(namespace=SEED_NAMESPACE, identity=identity, root_seed=root),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return dict(
        root_seed=root,
        sampling_seed=int.from_bytes(digest[:8], "big") & ((1 << 63) - 1),
        identity_sha256=digest.hex(),
        canonical_json=payload.decode("ascii"),
    )


def transition_multiplicities(windows, horizon=40):
    """Count overlapping outgoing transitions without conflating different cases."""
    if (
        type(horizon) is not int
        or horizon <= 0
        or not isinstance(windows, (list, tuple))
        or not windows
        or any(
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or any(type(value) is not int or value < 0 for value in item)
            for item in windows
        )
        or len({tuple(item) for item in windows}) != len(windows)
    ):
        raise ValueError("Distinct nonnegative case/anchor windows and a positive horizon are required")
    counts = Counter((case, step) for case, start in windows for step in range(start, start + horizon))
    return np.asarray(
        [[counts[(case, step)] for step in range(start, start + horizon)] for case, start in windows], dtype=np.int32
    )


def clipping_translation_effect(physical_action128, tcp_rotations, normalization):
    """Quantify only clipping loss, expressed and accumulated in the base frame.

    The same measured TCP orientations are used for both paths. FP32 affine
    roundoff before clipping is held fixed, so this does not measure general
    quantization, model prediction, feedback or physical tracking error.
    """
    physical = np.asarray(physical_action128, dtype=np.float64)
    rotations = np.asarray(tcp_rotations, dtype=np.float64)
    if (
        physical.shape != (40, 128)
        or rotations.shape != (40, 3, 3)
        or not np.isfinite(physical).all()
        or not np.isfinite(rotations).all()
        or not np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=1e-8, rtol=0)
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("Forty actual labels and proper measured TCP rotations are required")
    normalized = normalize_active10(physical, **normalization)
    transform = normalization["transform"]
    mask = np.asarray(transform["normalization_mask"], dtype=bool)
    if not np.array_equal(mask[:10], np.array([True, True, True] + [False] * 7)):
        raise ValueError("This clipping diagnostic requires the registered three-position mask")
    scale = np.asarray(transform["scale"], dtype=np.float32)[:3]
    bias = np.asarray(transform["bias"], dtype=np.float32)[:3]
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("Nonzero finite position normalization scales are required")
    values = physical[:, :3].astype(np.float32)
    affine = np.add(np.multiply(values, scale, dtype=np.float32), bias, dtype=np.float32)
    clipped = np.clip(affine, np.float32(transform["clip_min"]), np.float32(transform["clip_max"]))
    flags = affine != clipped
    if int(flags.sum()) != normalized["clipped_count"]:
        raise ValueError("Clipping flags must match the original active10 contract")
    local = (clipped.astype(np.float64) - affine.astype(np.float64)) / scale.astype(np.float64)
    cumulative = np.cumsum(np.einsum("tij,tj->ti", rotations, local), axis=0)
    distances = np.linalg.norm(cumulative, axis=1)
    return dict(
        clipped_mask=flags,
        local_translation_error=local,
        cumulative_base_translation_error=cumulative,
        max_prefix_error_m=float(np.max(distances)),
        final_error_m=float(distances[-1]),
    )
