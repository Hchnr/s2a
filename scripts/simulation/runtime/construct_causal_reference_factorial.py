"""Construct fixed same-time component interventions for physical diagnosis."""

import numpy as np

from s2a.geometry import finite, validate_transform

MODES = ("bridge_pose_old_opening", "old_pose_bridge_opening")


def construct_reference(
    old_poses,
    old_openings,
    bridge_poses,
    bridge_openings,
    *,
    old_step,
    origin_step,
    common_steps,
    continuation_steps,
    preview_steps,
    mode,
):
    """Mix only previously generated components aligned to identical endpoints.

    This is a diagnostic intervention, not a world-model-generated state or a
    control qualification. Beyond the old chunk, the bridge component is used.
    """
    if (
        any(type(x) is not int for x in (old_step, origin_step, common_steps, continuation_steps, preview_steps))
        or old_step < 0
        or origin_step <= old_step
        or not 0 < common_steps < continuation_steps
        or preview_steps < 1
        or mode not in MODES
    ):
        raise ValueError("Fixed causal generation times, horizon and registered intervention required")
    old, bridge = validate_transform(old_poses), validate_transform(bridge_poses)
    if old.ndim != 3 or old.shape != bridge.shape:
        raise ValueError("Equal complete generated pose horizons required")
    horizon, shift = len(bridge), origin_step - old_step
    old_openings, bridge_openings = finite(old_openings, (horizon,)), finite(bridge_openings, (horizon,))
    if np.any((old_openings < 0) | (old_openings > 1) | (bridge_openings < 0) | (bridge_openings > 1)):
        raise ValueError("Finite normalized opening references required")
    old_aligned_horizon = horizon - shift
    if old_aligned_horizon < continuation_steps + preview_steps - 1:
        raise ValueError("Old generated chunk must cover the complete executable preview")
    if not (
        np.array_equal(bridge[:common_steps], old[shift : shift + common_steps])
        and np.array_equal(bridge_openings[:common_steps], old_openings[shift : shift + common_steps])
    ):
        raise ValueError("The bridge must preserve the common old-reference prefix")
    poses, openings = bridge.copy(), bridge_openings.copy()
    if mode == "old_pose_bridge_opening":
        poses[:old_aligned_horizon] = old[shift:]
    else:
        openings[:old_aligned_horizon] = old_openings[shift:]
    return {
        "mode": mode,
        "old_step": old_step,
        "origin_step": origin_step,
        "common_steps": common_steps,
        "continuation_steps": continuation_steps,
        "preview_steps": preview_steps,
        "old_aligned_horizon": old_aligned_horizon,
        "poses": poses,
        "openings": openings,
    }
