"""Task-matched current TCP anchor distances; not a model-support classifier."""

import numpy as np

from s2a.geometry import validate_transform


def nearest_position_anchor(query, anchors):
    """Choose by TCP position, then report rotation/opening on that same anchor."""
    pose = validate_transform(np.asarray(query["tcp_pose"], dtype=np.float64))
    opening = float(query["opening"])
    if pose.shape != (4, 4) or not np.isfinite(opening) or not 0 <= opening <= 1 or not anchors:
        raise ValueError("One finite current TCP/opening and nonempty task-matched anchors are required")
    matched = [row for row in anchors if row["task_index"] == query["task_index"]]
    if not matched:
        raise ValueError("At least one anchor for the same task is required")
    transforms = validate_transform(np.asarray([row["tcp_pose"] for row in matched], dtype=np.float64))
    apertures = np.asarray([row["opening"] for row in matched], dtype=np.float64)
    if (
        transforms.shape != (len(matched), 4, 4)
        or not np.isfinite(apertures).all()
        or np.any((apertures < 0) | (apertures > 1))
    ):
        raise ValueError("Finite measured anchor TCP/opening values are required")
    distances = np.linalg.norm(transforms[:, :3, 3] - pose[:3, 3], axis=1)
    index = int(np.argmin(distances))
    relative = pose[:3, :3].T @ transforms[index, :3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
    angle = float(np.arctan2(np.linalg.norm(skew) / 2.0, cosine))
    return dict(
        anchor_id=matched[index]["anchor_id"],
        source_kind=matched[index]["source_kind"],
        source_identity=matched[index]["source_identity"],
        anchor_step=matched[index]["anchor_step"],
        position_m=float(distances[index]),
        rotation_rad=angle,
        opening_abs=float(abs(apertures[index] - opening)),
    )


def summarize_distances(values, bins_m=(0.01, 0.02, 0.05, 0.1)):
    """Summarize correlated requests without treating thresholds as eligibility."""
    distances = np.asarray(values, dtype=np.float64)
    bins = np.asarray(bins_m, dtype=np.float64)
    if distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("A nonempty finite position-distance vector is required")
    if bins.ndim != 1 or not len(bins) or not np.isfinite(bins).all() or np.any(np.diff(bins) <= 0) or bins[0] <= 0:
        raise ValueError("Positive increasing descriptive distance bins are required")
    return dict(
        count=len(distances),
        median_m=float(np.percentile(distances, 50)),
        p90_m=float(np.percentile(distances, 90)),
        max_m=float(np.max(distances)),
        at_most_m={str(float(limit)): int(np.count_nonzero(distances <= limit)) for limit in bins},
        percentile_method="numpy_linear",
    )
