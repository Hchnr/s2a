"""Recompute the last completed ordered-correct arm prediction histories."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import validate_transform
from s2a.research.joint_control.preview.completed_history import CompletedHistory, restore_recomputed_history
from s2a.research.response_models.forward.current_coupled_response import build_response, nominal_prediction


def restore_ordered_arm_history(last_row, current_site, current_twist, contract, tracking_innovation, *, refiner=None):
    """Restore two histories only if their last predictions explain the measured endpoint."""
    if type(last_row["step"]) is not int or not 0 <= last_row["step"] < 600:
        raise ValueError("A completed original control period is required")
    site = validate_transform(np.asarray(last_row["site_pose"], dtype=float))
    endpoint = validate_transform(np.asarray(current_site, dtype=float))
    if site.shape != (4, 4) or endpoint.shape != (4, 4):
        raise ValueError("One previous and one current measured site pose required")
    twist = np.asarray(last_row["site_twist"], dtype=float)
    dynamics, refinement = last_row["dynamics"], last_row["refinement"]
    goal = np.asarray(refinement["expected_goal_ori"], dtype=float)
    if goal.shape != (3, 3) or not np.isfinite(goal).all():
        raise ValueError("The original controller rotation goal is required")
    relative = goal @ site[:3, :3].T
    model = build_response(dynamics["mass"], dynamics["jacobian"], dynamics["kp"], dynamics["kd"], relative)
    command = np.asarray(last_row["command"], dtype=float)
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    prediction = nominal_prediction(model, twist, interface.decode(command[:6])[:3])
    completed = CompletedHistory(
        previous_command=command,
        coupled_innovation=last_row["coupled_innovation_after"],
        tracking_innovation=tracking_innovation,
        last_plan=refinement["planned_commands"],
        completed_periods=last_row["step"] + 1,
    )
    return restore_recomputed_history(
        completed,
        site,
        twist,
        prediction,
        refinement["nominal_response"],
        endpoint,
        current_twist,
        refiner=refiner,
    )
