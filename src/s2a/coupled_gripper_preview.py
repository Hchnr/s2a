"""Experimental pose-coupled preview using measured and generated states only."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import validate_transform
from s2a.gripper_reference_preview import opening_preview


def coupled_opening_preview(
    poses,
    openings,
    current_index,
    measured_tcp,
    *,
    preview_steps,
    position_span_max_m,
    rotation_span_max_rad,
    measured_position_max_m,
    measured_rotation_max_rad,
):
    """Keep the current opening unless all whole-segment/current-state guards pass.

    Bounds are explicit hypotheses, not calibrated safety guarantees. Pose and
    opening must refer to the same already generated endpoints and TCP frame.
    """
    targets = validate_transform(np.asarray(poses, dtype=float))
    measured = validate_transform(np.asarray(measured_tcp, dtype=float))
    if targets.ndim != 3 or measured.shape != (4, 4) or len(targets) != len(openings):
        raise ValueError("Matching generated pose/opening endpoints and measured TCP required")
    limits = np.asarray(
        [position_span_max_m, rotation_span_max_rad, measured_position_max_m, measured_rotation_max_rad], dtype=float
    )
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("Explicit positive finite guard bounds required")
    selection = opening_preview(openings, current_index, preview_steps=preview_steps)
    stop = selection["selected_index"]
    segment = targets[current_index : stop + 1]
    anchor = targets[current_index]
    position_span = float(np.max(np.linalg.norm(segment[:, :3, 3] - anchor[:3, 3], axis=1)))
    rotation_span = float(np.max(Rotation.from_matrix(anchor[:3, :3].T @ segment[:, :3, :3]).magnitude()))
    measured_position = float(np.linalg.norm(measured[:3, 3] - anchor[:3, 3]))
    measured_rotation = float(Rotation.from_matrix(anchor[:3, :3].T @ measured[:3, :3]).magnitude())
    guards = dict(
        position_span=position_span <= position_span_max_m,
        rotation_span=rotation_span <= rotation_span_max_rad,
        measured_position=measured_position <= measured_position_max_m,
        measured_rotation=measured_rotation <= measured_rotation_max_rad,
    )
    accepted = selection["advanced"] and all(guards.values())
    return dict(
        reference=selection["reference"] if accepted else selection["original_reference"],
        original_reference=selection["original_reference"],
        advanced=accepted,
        selected_index=stop if accepted else current_index,
        preview_proposal=selection,
        guards=guards,
        position_span_m=position_span,
        rotation_span_rad=rotation_span,
        measured_position_error_m=measured_position,
        measured_rotation_error_rad=measured_rotation,
    )
