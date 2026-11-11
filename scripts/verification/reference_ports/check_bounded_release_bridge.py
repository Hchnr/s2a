"""Independently check the saved-reference bridge without loading the selector."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def verify_selected_reference(decision, old_poses, old_openings, latest_poses, latest_openings):
    """Check absolute endpoint alignment and independently reconstruct the eight-point blend."""
    old_poses, latest_poses = np.asarray(old_poses, dtype=float), np.asarray(latest_poses, dtype=float)
    old_openings, latest_openings = np.asarray(old_openings, dtype=float), np.asarray(latest_openings, dtype=float)
    if (
        old_poses.shape != (40, 4, 4)
        or latest_poses.shape != (40, 4, 4)
        or old_openings.shape != (40,)
        or latest_openings.shape != (40,)
        or not all(np.isfinite(value).all() for value in (old_poses, old_openings, latest_poses, latest_openings))
    ):
        raise ValueError("Two complete finite generated references are required")
    if (
        decision.get("step") != 376
        or decision.get("absolute_first_endpoint") != 377
        or decision.get("latest_source_step") != 360
    ):
        raise ValueError("The selected reference must begin at the completed boundary")
    source, reason = decision.get("source"), decision.get("reason")
    expected_poses, expected_openings = latest_poses[16:].copy(), latest_openings[16:].copy()
    if source == "bridge" and reason == "bounded_bridge":
        if decision.get("committed_source_step") != 344:
            raise ValueError("The bridge must use the original committed chunk")
        for index in range(8):
            alpha = (index + 1) / 8
            old, latest = old_poses[32 + index], latest_poses[16 + index]
            expected_poses[index, :3, 3] = (1 - alpha) * old[:3, 3] + alpha * latest[:3, 3]
            rotations = Rotation.from_matrix(np.stack([old[:3, :3], latest[:3, :3]]))
            expected_poses[index, :3, :3] = Slerp([0, 1], rotations)([alpha]).as_matrix()[0]
            expected_openings[index] = (1 - alpha) * old_openings[32 + index] + alpha * latest_openings[16 + index]
    elif (source, reason) != ("latest", "not_closing"):
        raise ValueError("Only the registered bridge or a completed closing episode is admissible")
    np.testing.assert_allclose(decision["poses"], expected_poses, atol=1e-10, rtol=0)
    np.testing.assert_allclose(decision["openings"], expected_openings, atol=1e-10, rtol=0)
    return {"source": source, "geometry_independently_verified": True, "selection_recomputed": False}
