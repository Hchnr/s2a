"""Build an aligned pose/opening transition from two already generated chunks."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, validate_transform


def build_reference_bridge(
    old_poses,
    old_openings,
    latest_poses,
    latest_openings,
    *,
    old_step,
    latest_step,
    start_offset,
    transition_steps,
):
    """Blend same-time endpoints without using any future observations.

    A reference at index j describes endpoint ``generated_step + j + 1``.
    Both input chunks must already exist at ``latest_step``. The result is a
    reference, not a feasibility or safety guarantee; actual tracking and the
    error to the latest unmodified reference must be evaluated independently.
    """
    if (
        any(type(value) is not int for value in (old_step, latest_step, start_offset, transition_steps))
        or old_step < 0
        or latest_step <= old_step
        or start_offset < 0
        or transition_steps < 1
    ):
        raise ValueError("Explicit causal generation steps and a positive transition are required")
    old_poses, latest_poses = validate_transform(old_poses), validate_transform(latest_poses)
    if old_poses.ndim != 3 or old_poses.shape != latest_poses.shape:
        raise ValueError("Both generated pose chunks must have the same finite horizon")
    horizon = len(latest_poses)
    old_openings, latest_openings = finite(old_openings, (horizon,)), finite(latest_openings, (horizon,))
    if np.any((old_openings < 0) | (old_openings > 1) | (latest_openings < 0) | (latest_openings > 1)):
        raise ValueError("Both opening chunks must be normalized")
    shift = latest_step - old_step
    old_points_required = start_offset + transition_steps - 1
    if start_offset + transition_steps > horizon or shift + old_points_required > horizon:
        raise ValueError("The generated chunks must overlap for the entire transition")

    poses, openings = latest_poses.copy(), latest_openings.copy()
    weights = np.ones(horizon, dtype=float)
    for index in range(start_offset + transition_steps - 1):
        old_index = shift + index
        alpha = max(0, index - start_offset + 1) / transition_steps
        weights[index] = alpha
        if alpha == 0:
            poses[index] = old_poses[old_index]
            openings[index] = old_openings[old_index]
            continue
        old_pose, latest_pose = old_poses[old_index], latest_poses[index]
        poses[index, :3, 3] = (1 - alpha) * old_pose[:3, 3] + alpha * latest_pose[:3, 3]
        relative = old_pose[:3, :3].T @ latest_pose[:3, :3]
        poses[index, :3, :3] = (
            old_pose[:3, :3] @ Rotation.from_rotvec(alpha * Rotation.from_matrix(relative).as_rotvec()).as_matrix()
        )
        openings[index] = (1 - alpha) * old_openings[old_index] + alpha * latest_openings[index]

    return {
        "old_step": old_step,
        "latest_step": latest_step,
        "origin_step": latest_step,
        "start_offset": start_offset,
        "transition_steps": transition_steps,
        "poses": poses,
        "openings": openings,
        "weights": weights,
    }
