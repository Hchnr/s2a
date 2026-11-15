"""Descriptive joint current-state matches against task-matched image anchors."""

import numpy as np

from s2a.geometry import validate_transform


def first_joint_anchor(query, anchors, *, max_position_m, max_rotation_rad, max_opening_abs):
    """Find the first single anchor satisfying every explicitly descriptive limit."""
    limits = (max_position_m, max_rotation_rad, max_opening_abs)
    if any(isinstance(value, bool) or not np.isfinite(value) or value < 0 for value in limits):
        raise ValueError("Finite nonnegative descriptive limits are required")
    pose = validate_transform(np.asarray(query["tcp_pose"], dtype=np.float64))
    opening = float(query["opening"])
    matched = [row for row in anchors if row["task_index"] == query["task_index"]]
    if pose.shape != (4, 4) or not np.isfinite(opening) or not 0 <= opening <= 1 or not matched:
        raise ValueError("One finite current TCP/opening and same-task anchors are required")
    transforms = validate_transform(np.asarray([row["tcp_pose"] for row in matched], dtype=np.float64))
    openings = np.asarray([row["opening"] for row in matched], dtype=np.float64)
    if (
        transforms.shape != (len(matched), 4, 4)
        or not np.isfinite(openings).all()
        or np.any((openings < 0) | (openings > 1))
    ):
        raise ValueError("Finite measured anchor TCP/opening values are required")
    positions = np.linalg.norm(transforms[:, :3, 3] - pose[:3, 3], axis=1)
    relative = pose[:3, :3].T @ transforms[:, :3, :3]
    cosines = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    skew = np.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        axis=1,
    )
    rotations = np.arctan2(np.linalg.norm(skew, axis=1) / 2.0, cosines)
    gaps = np.abs(openings - opening)
    indices = np.flatnonzero(
        (positions <= max_position_m) & (rotations <= max_rotation_rad) & (gaps <= max_opening_abs)
    )
    if not len(indices):
        return None
    index = int(indices[0])
    return dict(
        anchor_id=matched[index]["anchor_id"],
        source_kind=matched[index]["source_kind"],
        position_m=float(positions[index]),
        rotation_rad=float(rotations[index]),
        opening_abs=float(gaps[index]),
    )
