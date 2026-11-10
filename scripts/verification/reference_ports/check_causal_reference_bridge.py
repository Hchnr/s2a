"""Independently verify a saved transition with quaternion SLERP."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _poses(value):
    poses = np.asarray(value, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Finite homogeneous pose chunks are required")
    rotations = poses[:, :3, :3]
    if not (
        np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
        and np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-7, rtol=0)
        and np.allclose(np.linalg.det(rotations), 1, atol=1e-7, rtol=0)
    ):
        raise ValueError("Invalid SE(3) reference")
    return poses


def _openings(value, horizon):
    openings = np.asarray(value, dtype=float)
    if openings.shape != (horizon,) or not np.isfinite(openings).all() or np.any((openings < 0) | (openings > 1)):
        raise ValueError("Finite normalized openings are required")
    return openings


def verify_reference_bridge(
    old_poses,
    old_openings,
    latest_poses,
    latest_openings,
    produced,
    *,
    old_step,
    latest_step,
    start_offset,
    transition_steps,
    atol=1e-10,
):
    """Check every point from original references, without the producer code.

    This only checks the saved reference construction, not the optimization,
    safety, causality of the upstream references, or task success.
    """
    if (
        any(type(value) is not int for value in (old_step, latest_step, start_offset, transition_steps))
        or old_step < 0
        or latest_step <= old_step
        or start_offset < 0
        or transition_steps < 1
        or not np.isfinite(atol)
        or atol <= 0
    ):
        raise ValueError("Invalid independent transition parameters")
    old, latest = _poses(old_poses), _poses(latest_poses)
    horizon = len(latest)
    old_openings, latest_openings = _openings(old_openings, horizon), _openings(latest_openings, horizon)
    shift = latest_step - old_step
    if (
        old.shape != latest.shape
        or start_offset + transition_steps > horizon
        or shift + start_offset + transition_steps - 1 > horizon
    ):
        raise ValueError("Generated references do not overlap at the required endpoints")
    metadata = {
        "old_step": old_step,
        "latest_step": latest_step,
        "origin_step": latest_step,
        "start_offset": start_offset,
        "transition_steps": transition_steps,
    }
    if set(produced) != {*metadata, "poses", "openings", "weights"} or any(
        type(produced[key]) is not int or produced[key] != value for key, value in metadata.items()
    ):
        raise ValueError("Transition metadata differs from registered source times")
    poses, openings = _poses(produced["poses"]), _openings(produced["openings"], horizon)
    weights = np.asarray(produced["weights"], dtype=float)
    if poses.shape != latest.shape or weights.shape != (horizon,) or not np.isfinite(weights).all():
        raise ValueError("Saved transition horizon does not match")

    maximum_pose, maximum_opening = 0.0, 0.0
    for index in range(horizon):
        alpha = min(1.0, max(0.0, (index - start_offset + 1) / transition_steps))
        if weights[index] != alpha:
            raise ValueError("Saved blend weight differs from the fixed schedule")
        if alpha == 1:
            expected_pose, expected_opening = latest[index], latest_openings[index]
            if not np.array_equal(poses[index], expected_pose) or openings[index] != expected_opening:
                raise ValueError("The completed transition must use the exact latest reference")
        elif alpha == 0:
            expected_pose, expected_opening = old[index + shift], old_openings[index + shift]
            if not np.array_equal(poses[index], expected_pose) or openings[index] != expected_opening:
                raise ValueError("The preserved prefix must use the exact aligned old reference")
        else:
            earlier = old[index + shift]
            expected_pose = np.eye(4)
            expected_pose[:3, 3] = (1 - alpha) * earlier[:3, 3] + alpha * latest[index, :3, 3]
            keyframes = Rotation.from_matrix(np.stack((earlier[:3, :3], latest[index, :3, :3])))
            expected_pose[:3, :3] = Slerp([0, 1], keyframes)([alpha]).as_matrix()[0]
            expected_opening = (1 - alpha) * old_openings[index + shift] + alpha * latest_openings[index]
        pose_difference = float(np.max(np.abs(poses[index] - expected_pose)))
        opening_difference = float(abs(openings[index] - expected_opening))
        maximum_pose, maximum_opening = max(maximum_pose, pose_difference), max(maximum_opening, opening_difference)
        if pose_difference > atol or opening_difference > atol:
            raise ValueError("Saved transition differs from the independent same-endpoint reconstruction")
    return {
        "passed": True,
        "checked_endpoints": horizon,
        "maximum_pose_element_difference": maximum_pose,
        "maximum_opening_difference": maximum_opening,
    }
