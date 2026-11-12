"""Expose a finite generated reference with explicitly marked terminal holding."""

import numpy as np

from s2a.geometry import validate_transform


def remaining_h40(poses, openings, *, consumed):
    """Return the remaining generated targets and hold their last endpoint to H40."""
    poses = validate_transform(np.asarray(poses, dtype=float))
    openings = np.asarray(openings, dtype=float)
    if (
        poses.shape != (40, 4, 4)
        or openings.shape != (40,)
        or not np.isfinite(openings).all()
        or np.any((openings < 0) | (openings > 1))
        or type(consumed) is not int
        or consumed not in (0, 8, 16, 24, 32)
    ):
        raise ValueError("One finite H40 reference and an eight-period aligned consumed prefix are required")
    count = 40 - consumed
    target_poses = np.concatenate((poses[consumed:], np.repeat(poses[-1:], consumed, axis=0)), axis=0)
    target_openings = np.concatenate((openings[consumed:], np.repeat(openings[-1:], consumed)), axis=0)
    return target_poses, target_openings, dict(generated_endpoints=count, terminal_hold_endpoints=consumed)
