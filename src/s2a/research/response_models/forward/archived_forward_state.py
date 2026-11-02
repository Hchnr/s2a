"""233 audit archival FK normalization without allowing integration or joint motion."""

import numpy as np
from scipy.spatial.transform import Rotation


def verify_forward_state(before, after, *, nq, quaternion_segments):
    before, after = (np.asarray(value, dtype=float) for value in (before, after))
    if before.ndim != 1 or after.shape != before.shape or type(nq) is not int or nq <= 0 or len(before) <= 1 + nq:
        raise ValueError("Aligned archival time/qpos/qvel and explicit nq required")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Finite archived and forwarded state required")
    strict = np.ones(len(before), dtype=bool)
    occupied = set()
    for segment in quaternion_segments:
        start = segment["qpos_start"]
        if type(start) is not int or start < 0 or start + 4 > nq or occupied.intersection(range(start, start + 4)):
            raise ValueError("Disjoint actual quaternion qpos segments required")
        occupied.update(range(start, start + 4))
        strict[1 + start : 1 + start + 4] = False
    np.testing.assert_array_equal(before[strict], after[strict])
    bound = 16 * np.finfo(np.float64).eps
    changes = []
    for segment in quaternion_segments:
        start = 1 + segment["qpos_start"]
        original, current = before[start : start + 4], after[start : start + 4]
        assert abs(np.linalg.norm(original) - 1) <= bound and abs(np.linalg.norm(current) - 1) <= bound
        difference = float(np.max(np.abs(current - original)))
        assert difference <= bound
        r0 = Rotation.from_quat(original[[1, 2, 3, 0]]).as_matrix()
        r1 = Rotation.from_quat(current[[1, 2, 3, 0]]).as_matrix()
        rotation_difference = float(np.max(np.abs(r0 - r1)))
        assert rotation_difference <= bound
        if difference:
            changes.append(
                dict(
                    joint=segment["joint"],
                    qpos_start=segment["qpos_start"],
                    max_component_difference=difference,
                    max_rotation_matrix_difference=rotation_difference,
                )
            )
    return changes
