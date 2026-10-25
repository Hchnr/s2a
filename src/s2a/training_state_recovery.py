"""Privileged training-only reference schedules for physical recovery trials."""

import numpy as np

from s2a.geometry import validate_transform


def recovery_reference(source_poses, openings, *, base_x_offset_m, perturb_steps=16, return_steps=40):
    """Return target references, never observed-state supervision.

    Input endpoints include the initial pose at index0. The offset is in base
    coordinates. No orientation or opening is invented during perturb/return.
    Original native endpoints start at index1 after returning to initial pose.
    """
    poses = validate_transform(np.asarray(source_poses, dtype=float))
    opening = np.asarray(openings, dtype=float)
    if poses.ndim != 3 or len(poses) < 41 or opening.shape != (len(poses),):
        raise ValueError("Requires full source endpoints with one opening per endpoint")
    if not np.isfinite(opening).all() or np.any((opening < 0) | (opening > 1)):
        raise ValueError("Openings must be finite in [0,1]")
    if not np.isfinite(base_x_offset_m) or isinstance(base_x_offset_m, bool):
        raise ValueError("Finite explicit base offset required")
    if any(type(n) is not int or n <= 0 for n in (perturb_steps, return_steps)):
        raise ValueError("Positive integer physical step counts required")
    initial = poses[0]
    outbound = np.repeat(initial[None], perturb_steps, axis=0)
    returning = np.repeat(initial[None], return_steps, axis=0)
    outbound[:, 0, 3] += np.arange(1, perturb_steps + 1) / perturb_steps * base_x_offset_m
    returning[:, 0, 3] += (1 - np.arange(1, return_steps + 1) / return_steps) * base_x_offset_m
    targets = np.concatenate((outbound, returning, poses[1:]))
    target_opening = np.concatenate((np.full(perturb_steps + return_steps, opening[0]), opening[1:]))
    return targets, target_opening
