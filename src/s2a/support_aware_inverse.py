"""166 blend learned and existing physical inverse commands using fit-only support."""

import numpy as np

from s2a.joint_wrench_condition_forward import predict, raw_condition
from s2a.qualified_enumerated_inverse import branch, inverse_translation
from s2a.training_support import blend_translation


def support_aware_translation(
    data,
    parameters,
    desired_motion,
    desired_velocity,
    *,
    support,
    physical_command,
    input_min,
    input_max,
    output_min,
    output_max,
    previous_command,
    predictor=predict,
):
    """Physical candidate must be generated causally from the same current state.

    Learned optimum and executed mixture are separate: the mixture need not be
    the optimum of the learned objective. Always re-predict the actual command.
    """
    raw = raw_condition(data)
    if raw.shape != (1, 89):
        raise ValueError("Exactly one current89 condition required")
    z = (raw[0] - np.asarray(parameters["state_mean"])) / np.asarray(parameters["state_scale"])
    decision = support.query(z)
    bounds = dict(input_min=input_min, input_max=input_max, output_min=output_min, output_max=output_max)
    learned = inverse_translation(
        data,
        parameters,
        desired_motion,
        desired_velocity,
        previous_command=previous_command,
        predictor=predictor,
        **bounds,
    )
    command = blend_translation(
        learned["command"],
        physical_command,
        previous_command,
        weight=decision["learned_weight"],
        lower=input_min,
        upper=input_max,
    )
    lo, hi, outlo, outhi = (np.asarray(v, dtype=float) for v in (input_min, input_max, output_min, output_max))
    actual_input = branch(data, command, lo, hi, outlo, outhi)
    actual = np.asarray(predictor(actual_input, parameters), dtype=float)
    if actual.shape != (1, 12) or not np.isfinite(actual).all():
        raise ValueError("One finite12 executed response required")
    physical = actual_input["action_goal_delta"][0, :3].copy()
    expected = learned["response_offset"] + learned["response_jacobian"] @ physical
    np.testing.assert_allclose(actual[0], expected, atol=1e-9, rtol=1e-8)
    target = np.r_[np.asarray(desired_motion), 0.05 * np.asarray(desired_velocity)]
    residual = actual[0, [0, 1, 2, 6, 7, 8]] - target
    return dict(
        command=command,
        physical_delta=physical,
        predicted_motion=actual[0, :3].copy(),
        predicted_linear_velocity=actual[0, 6:9] / 0.05,
        full_predicted_response=actual[0].copy(),
        squared_objective=float(residual @ residual),
        learned_solution=learned,
        support_decision=decision,
        physical_candidate=np.asarray(physical_command).copy(),
    )
