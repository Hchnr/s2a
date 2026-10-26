"""202 support-aware tracking-constrained final translation rate."""

import numpy as np

from s2a.joint_wrench_condition_forward import predict
from s2a.loading_phase_inverse import loading_phase_translation
from s2a.qualified_enumerated_inverse import branch
from s2a.tracking_guarded_rate import guarded_rate_command


def tracking_guarded_loading_translation(
    data,
    parameters,
    desired_motion,
    desired_velocity,
    *,
    support,
    route,
    physical_command,
    input_min,
    input_max,
    output_min,
    output_max,
    previous_command,
    predictor=predict,
):
    bounds = dict(input_min=input_min, input_max=input_max, output_min=output_min, output_max=output_max)
    result = loading_phase_translation(
        data,
        parameters,
        desired_motion,
        desired_velocity,
        support=support,
        route=route,
        physical_command=physical_command,
        previous_command=previous_command,
        predictor=predictor,
        **bounds,
    )
    previous = np.asarray(previous_command, dtype=float)
    before = np.asarray(result["command"]).copy()
    learned = result["learned_solution"]
    lo, hi, outlo, outhi = (
        np.asarray(bounds[k], dtype=float) for k in ("input_min", "input_max", "output_min", "output_max")
    )
    scale = (outhi - outlo) / (hi - lo)
    jacobian = np.asarray(learned["response_jacobian"])[:3]
    matrix = jacobian * scale[None, :]
    offset = np.asarray(learned["response_offset"])[:3] + jacobian @ (outlo - lo * scale)
    guard = dict(
        command=before.copy(),
        change=0.25,
        tracking_guard_feasible=None,
        predicted_error=float(np.linalg.norm(matrix @ before + offset - desired_motion)),
        reason="tracking_route_unchanged",
    )
    if route == "native_close_load":
        if result["support_decision"]["distance"] >= 2 * support.radius:
            guard["reason"] = "outside_training_support"
        else:
            guard = guarded_rate_command(matrix, offset, desired_motion, previous, before)
            guard["reason"] = "guard_evaluated"
    command = np.asarray(guard["command"]).copy()
    change = guard["change"]
    row = branch(
        data,
        command,
        *(np.asarray(bounds[k], dtype=float) for k in ("input_min", "input_max", "output_min", "output_max")),
    )
    full = np.asarray(predictor(row, parameters), dtype=float)
    if full.shape != (1, 12) or not np.isfinite(full).all():
        raise ValueError("One finite executed prediction required")
    physical = row["action_goal_delta"][0, :3].copy()
    learned = result["learned_solution"]
    np.testing.assert_allclose(
        full[0], learned["response_offset"] + learned["response_jacobian"] @ physical, atol=1e-9, rtol=1e-8
    )
    residual = full[0, [0, 1, 2, 6, 7, 8]] - np.r_[desired_motion, 0.05 * np.asarray(desired_velocity)]
    result.update(
        command=command,
        physical_delta=physical,
        full_predicted_response=full[0].copy(),
        predicted_motion=full[0, :3].copy(),
        predicted_linear_velocity=full[0, 6:9] / 0.05,
        squared_objective=float(residual @ residual),
        pre_rate_command=before,
        normalized_translation_rate=change / 0.05,
        tracking_guard=guard,
    )
    return result
