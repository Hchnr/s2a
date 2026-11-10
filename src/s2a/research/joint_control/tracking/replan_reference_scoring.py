"""Score one actual TCP against both causal, time-aligned reference choices."""

import numpy as np

from s2a.geometry import finite, pose_error


def score_reference_pair(actual_tcp, old_targets, new_targets, offset):
    """Use the completed endpoint and the same absolute target for each chunk."""
    actual = finite(actual_tcp, (4, 4))
    old = finite(old_targets, (8, 4, 4))
    new = finite(new_targets, (8, 4, 4))
    if type(offset) is not int or not 0 <= offset < 8:
        raise ValueError("A completed offset within the registered window is required")
    result = {"old": pose_error(old[offset], actual), "new": pose_error(new[offset], actual)}
    if any(np.asarray(value).shape != (6,) or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Finite position and SO(3) errors required")
    return result
