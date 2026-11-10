"""Describe task-phase revisions at matched absolute generated endpoints."""

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform


def compare_replans(old_poses, old_openings, new_poses, new_openings, *, old_step, new_step, measured_tcp, opening):
    """Compare two H40 references available at a single eight-step replan.

    The returned opening events have no identity across predictions; matching
    first thresholds describes revisions, not a causal physical event.
    """
    if type(old_step) is not int or type(new_step) is not int or old_step < 0 or new_step != old_step + 8:
        raise ValueError("Two consecutive eight-period replans are required")
    old, new = validate_transform(old_poses), validate_transform(new_poses)
    current = validate_transform(measured_tcp)
    old_o, new_o = finite(old_openings, (40,)), finite(new_openings, (40,))
    if old.shape != (40, 4, 4) or new.shape != (40, 4, 4) or current.shape != (4, 4):
        raise ValueError("Two H40 TCP trajectories and one current TCP required")
    if np.any((old_o < 0) | (old_o > 1)) or np.any((new_o < 0) | (new_o > 1)):
        raise ValueError("Generated openings must be normalized")
    if not np.isfinite(opening) or not 0 <= opening <= 1:
        raise ValueError("Current measured opening must be normalized")

    old_overlap, new_overlap = old[8:], new[:32]
    error = pose_error(old_overlap, new_overlap)
    position, rotation = np.linalg.norm(error[:, :3], axis=1), np.linalg.norm(error[:, 3:], axis=1)
    previous_o, latest_o = old_o[8:], new_o[:32]
    current_error = pose_error(old_overlap[0], current)
    current_position = float(np.linalg.norm(current_error[:3]))
    current_rotation = float(np.linalg.norm(current_error[3:]))
    prefix_agrees = bool(np.all(position[:8] <= 0.02) and np.all(rotation[:8] <= 0.1))
    current_near_prefix = bool(current_position <= 0.03 and current_rotation <= 0.15)

    def first(values, direction):
        indices = np.flatnonzero(values < 0.35 if direction == "closure" else values > 0.8)
        return int(indices[0]) if len(indices) else None

    def event(direction):
        old_index, new_index = first(old_o, direction), first(new_o, direction)
        result = {"old_first_index": old_index, "new_first_index": new_index, "deferred_in_old": False}
        if old_index is None or old_index < 8:
            return result
        same_time = old_index - 8
        shift = None if new_index is None else new_index - same_time
        category = (
            "not_in_new_horizon" if shift is None else "earlier" if shift < 0 else "later" if shift > 0 else "same"
        )
        opposite = new_o[same_time] > 0.8 if direction == "closure" else new_o[same_time] < 0.35
        result.update(
            deferred_in_old=True,
            old_absolute_endpoint=old_step + old_index + 1,
            new_first_absolute_endpoint=None if new_index is None else new_step + new_index + 1,
            new_first_shift_steps=shift,
            first_event_revision=category,
            new_opening_at_old_event=float(new_o[same_time]),
            opposite_opening_at_old_event=bool(opposite),
            event_position_difference_m=float(position[same_time]),
            event_rotation_difference_rad=float(rotation[same_time]),
            event_pose_agreement=bool(position[same_time] <= 0.02 and rotation[same_time] <= 0.1),
            next_prefix_pose_agreement=prefix_agrees,
            current_near_old_prefix=current_near_prefix,
            currently_opposite_phase=bool(opening > 0.8 if direction == "closure" else opening < 0.35),
        )
        return result

    return {
        "old_step": old_step,
        "new_step": new_step,
        "current_opening": float(opening),
        "position_difference_m": position.tolist(),
        "rotation_difference_rad": rotation.tolist(),
        "opening_difference": (previous_o - latest_o).tolist(),
        "old_closed_new_open": int(np.count_nonzero((previous_o < 0.35) & (latest_o > 0.8))),
        "old_open_new_closed": int(np.count_nonzero((previous_o > 0.8) & (latest_o < 0.35))),
        "next_prefix_pose_agreement": prefix_agrees,
        "current_position_error_m": current_position,
        "current_rotation_error_rad": current_rotation,
        "current_near_old_prefix": current_near_prefix,
        "closure": event("closure"),
        "opening": event("opening"),
    }
