"""Causal single-step feedback for a bounded tabletop projection experiment."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import finite, inverse, pose_error, validate_transform
from s2a.research.joint_control.tracking.tabletop_reference_projection import project_tabletop_reference


def select_tabletop_feedback(
    measured_tcp,
    target_tcp,
    site_from_tcp,
    vertices_tcp,
    table_pose,
    table_half_size,
    previous_command,
    interface,
    *,
    mode,
    clearance,
    numerical_margin,
    max_translation,
    command_step_limit,
):
    """Select from current measured geometry and one already-generated target.

    This deliberately simple experiment keeps the previous gripper command and
    clips each arm command against the actual last dispatch. It does not assess
    contacts, future physical observations, task success, or path feasibility.
    The caller must separately guard dispatch and capture completed feedback.
    """
    measured, target, tool = (
        validate_transform(np.asarray(value, dtype=float)) for value in (measured_tcp, target_tcp, site_from_tcp)
    )
    previous = finite(previous_command, (7,)).copy()
    if (
        measured.shape != (4, 4)
        or target.shape != (4, 4)
        or tool.shape != (4, 4)
        or type(interface) is not OSCInterface
        or np.max(np.abs(previous)) > 1
        or mode not in ("original", "nominal", "pose_bounded")
        or isinstance(command_step_limit, bool)
        or not np.isscalar(command_step_limit)
        or not np.isfinite(command_step_limit)
        or command_step_limit <= 0
    ):
        raise ValueError(
            "One measured pose, generated target, fixed interface and bounded previous command are required"
        )
    projection = None
    selected = target.copy()
    if mode != "original":
        position, rotation = (0.0, 0.0) if mode == "nominal" else (0.01, 0.05)
        projection = project_tabletop_reference(
            target,
            vertices_tcp,
            table_pose,
            table_half_size,
            position_uncertainty=position,
            rotation_uncertainty=rotation,
            shape_uncertainty=0.0,
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
    site_target = selected @ tcp_to_site
    current_site = measured @ tcp_to_site
    delta = pose_error(site_target, current_site)
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
