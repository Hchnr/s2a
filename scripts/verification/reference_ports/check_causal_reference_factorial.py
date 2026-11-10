"""Independently check same-endpoint component interventions."""

import numpy as np

from s2a.geometry import finite, validate_transform


def verify_reference(
    old_poses,
    old_openings,
    bridge_poses,
    bridge_openings,
    produced,
    *,
    old_step,
    origin_step,
    common_steps,
    continuation_steps,
    preview_steps,
    mode,
):
    """Compare every saved point without importing the intervention producer."""
    if (
        any(type(x) is not int for x in (old_step, origin_step, common_steps, continuation_steps, preview_steps))
        or old_step < 0
        or origin_step <= old_step
        or not 0 < common_steps < continuation_steps
        or preview_steps < 1
        or mode not in ("bridge_pose_old_opening", "old_pose_bridge_opening")
    ):
        raise ValueError("Invalid registered component intervention")
    old, bridge = validate_transform(old_poses), validate_transform(bridge_poses)
    if old.ndim != 3 or bridge.shape != old.shape:
        raise ValueError("Equal generated pose horizons required")
    horizon, shift = len(old), origin_step - old_step
    old_openings, bridge_openings = finite(old_openings, (horizon,)), finite(bridge_openings, (horizon,))
    poses = validate_transform(produced["poses"])
    openings = finite(produced["openings"], (horizon,))
    if poses.shape != old.shape or any(np.any((v < 0) | (v > 1)) for v in (old_openings, bridge_openings, openings)):
        raise ValueError("Finite, normalized and aligned saved references required")
    available = horizon - shift
    if available < continuation_steps + preview_steps - 1:
        raise ValueError("The old reference does not cover the executable preview")
    expected_metadata = {
        "mode": mode,
        "old_step": old_step,
        "origin_step": origin_step,
        "common_steps": common_steps,
        "continuation_steps": continuation_steps,
        "preview_steps": preview_steps,
        "old_aligned_horizon": available,
    }
    if set(produced) != {*expected_metadata, "poses", "openings"} or any(
        type(produced[key]) is not type(value) or produced[key] != value for key, value in expected_metadata.items()
    ):
        raise ValueError("Saved intervention metadata differs from the registration")
    for index in range(horizon):
        if index < common_steps and (
            not np.array_equal(bridge[index], old[index + shift])
            or bridge_openings[index] != old_openings[index + shift]
        ):
            raise ValueError("The two input references do not share the common prefix")
        use_old_pose = mode == "old_pose_bridge_opening" and index < available
        use_old_opening = mode == "bridge_pose_old_opening" and index < available
        expected_pose = old[index + shift] if use_old_pose else bridge[index]
        expected_opening = old_openings[index + shift] if use_old_opening else bridge_openings[index]
        if not np.array_equal(poses[index], expected_pose) or openings[index] != expected_opening:
            raise ValueError("Saved component differs from its aligned source endpoint")
    return {"passed": True, "checked_endpoints": horizon, "old_aligned_horizon": available}
