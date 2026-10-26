"""199 final translation-rate ablation during causal loading route."""

import numpy as np

from s2a.joint_wrench_condition_forward import predict
from s2a.loading_phase_inverse import loading_phase_translation
from s2a.qualified_enumerated_inverse import branch


def rate_limited_loading_translation(
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
    change = 0.125 if route == "native_close_load" else 0.25
    before = np.asarray(result["command"]).copy()
    command = np.minimum(
        np.minimum(input_max, previous + change), np.maximum(np.maximum(input_min, previous - change), before)
    )
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
    )
    return result
