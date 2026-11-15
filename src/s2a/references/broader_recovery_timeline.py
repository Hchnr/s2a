"""Privileged training references for the registered broader recovery pilot."""

import numpy as np

from s2a.geometry import validate_transform
from s2a.references.oracle_timeline import HORIZON, STRIDE, OracleTimeline


def broader_recovery_timeline(source_poses, openings, source_from_tcp, *, variant, sign):
    """Construct the fixed base-frame recovery targets, never measured labels.

    Native input endpoints include the initial state. The first 16 desired
    endpoints reach the perturbation, the next 40 return to the initial TCP,
    and the remaining endpoints follow the measured native trajectory.
    """
    poses = validate_transform(np.asarray(source_poses, dtype=float))
    opening = np.asarray(openings, dtype=float)
    tool = validate_transform(np.asarray(source_from_tcp, dtype=float))
    if poses.ndim != 3 or len(poses) < 41 or opening.shape != (len(poses),):
        raise ValueError("Requires full source endpoints with one opening per endpoint")
    if tool.shape != (4, 4):
        raise ValueError("One fixed source-from-TCP transform is required")
    if not np.isfinite(opening).all() or np.any((opening < 0) | (opening > 1)):
        raise ValueError("Openings must be finite in [0,1]")
    if variant not in ("zero", "base_x", "base_y", "base_yaw"):
        raise ValueError("An explicitly registered recovery variant is required")
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("The registered demo sign must be integer -1 or 1")

    tcp = validate_transform(poses @ tool)
    initial = tcp[0]
    factors = np.concatenate((np.arange(1, 17) / 16, 1 - np.arange(1, 41) / 40))
    prefix = np.repeat(initial[None], len(factors), axis=0)
    if variant in ("base_x", "base_y"):
        axis = 0 if variant == "base_x" else 1
        prefix[:, axis, 3] += factors * sign * 0.06
    elif variant == "base_yaw":
        angles = factors * sign * 0.5
        rotations = np.repeat(np.eye(3)[None], len(factors), axis=0)
        rotations[:, 0, 0] = rotations[:, 1, 1] = np.cos(angles)
        rotations[:, 1, 0] = np.sin(angles)
        rotations[:, 0, 1] = -np.sin(angles)
        prefix[:, :3, :3] = rotations @ initial[:3, :3]

    targets = validate_transform(np.concatenate((prefix, tcp[1:])))
    target_openings = np.concatenate((np.full(len(prefix), opening[0]), opening[1:]))
    windows = [
        dict(
            step=step,
            effective_targets_tcp=targets[step : step + HORIZON],
            openings=target_openings[step : step + HORIZON],
        )
        for step in range(0, len(targets), STRIDE)
    ]
    timeline = OracleTimeline(windows, periods=len(targets))
    return timeline, dict(
        variant=variant,
        sign=sign,
        translation_magnitude_m=0.06,
        base_yaw_magnitude_rad=0.5,
        perturb_steps=16,
        return_steps=40,
        periods=timeline.periods,
        source_endpoints=len(poses),
        rotation_convention="base_z_left_multiply_about_current_tcp_origin",
        reference_kind="privileged_training_recovery_oracle",
        physical_labels_from_reference=False,
    )
