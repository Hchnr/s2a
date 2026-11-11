"""Causal fixed-interface feedback using an opt-in box-face target."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import finite, inverse, pose_error, validate_transform
from s2a.research.joint_control.tracking.box_face_reference_projection import project_box_faces_reference


def select_box_face_feedback(
    measured_tcp,
    target_tcp,
    site_from_tcp,
    vertices_tcp,
    slide_axis_tcp,
    current_joint,
    joint_range,
    boxes,
    previous_command,
    interface,
    *,
    position_uncertainty,
    rotation_uncertainty,
    additional_shape_uncertainty,
    clearance,
    numerical_margin,
    max_translation,
    command_step_limit,
):
    """Select from current measured geometry and one already-generated target.

    Joint-range and projection failures stop before dispatch. The fixed OSC
    interface and previous gripper command are preserved. This single-step
    rule does not guarantee future tracking, path clearance or task success.
    """
    measured, target, tool = (
        validate_transform(np.asarray(value, dtype=float)) for value in (measured_tcp, target_tcp, site_from_tcp)
    )
    previous, interval = finite(previous_command, (7,)).copy(), finite(joint_range, (2,))
    if (
        measured.shape != (4, 4)
        or target.shape != (4, 4)
        or tool.shape != (4, 4)
        or type(interface) is not OSCInterface
        or np.max(np.abs(previous)) > 1
        or interval[0] > interval[1]
        or isinstance(current_joint, bool)
        or not np.isscalar(current_joint)
        or not np.isfinite(current_joint)
        or isinstance(command_step_limit, bool)
        or not np.isscalar(command_step_limit)
        or not np.isfinite(command_step_limit)
        or command_step_limit <= 0
    ):
        raise ValueError("Current pose, joint range, fixed interface and bounded previous command are required")
    if not interval[0] <= current_joint <= interval[1]:
        return {
            "stop": {"reason": "finger_joint_outside_declared_range", "joint_position_m": float(current_joint)},
            "newly_selected": False,
        }
    projection = project_box_faces_reference(
        target,
        vertices_tcp,
        slide_axis_tcp,
        current_joint,
        *interval,
        boxes,
        position_uncertainty=position_uncertainty,
        rotation_uncertainty=rotation_uncertainty,
        additional_shape_uncertainty=additional_shape_uncertainty,
        clearance=clearance,
        numerical_margin=numerical_margin,
        max_translation=max_translation,
    )
    if not projection["accepted"]:
        return {
            "stop": {"reason": "projection_correction_exceeds_limit", "projection": projection},
            "newly_selected": False,
        }
    selected = projection["projected_tcp"].copy()
    tcp_to_site = inverse(tool)
    delta = pose_error(selected @ tcp_to_site, measured @ tcp_to_site)
    encoded, saturated = interface.encode(delta)
    arm = np.clip(encoded, previous[:6] - command_step_limit, previous[:6] + command_step_limit)
    if np.any(arm < interface.input_min) or np.any(arm > interface.input_max):
        raise ValueError("Previous command and rate limit must preserve the fixed controller interface")
    return {
        "command": np.r_[arm, previous[6]],
        "newly_selected": True,
        "original_target_tcp": target.copy(),
        "selected_target_tcp": selected,
        "reference_translation_m": float(np.linalg.norm(selected[:3, 3] - target[:3, 3])),
        "requested_site_error": delta,
        "encoded_before_rate_limit": encoded,
        "interface_saturated": saturated,
        "rate_limited": bool(np.any(np.abs(arm - encoded) > 1e-12)),
        "projection": projection,
    }
