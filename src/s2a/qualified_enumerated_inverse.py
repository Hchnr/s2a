"""Extract physical affine translation response and solve actual bounded commands."""

import numpy as np

from s2a.enumerated_response_inverse import translation_command, vector
from s2a.joint_wrench_condition_forward import predict


def branch(data, command, input_min, input_max, output_min, output_max):
    result = {key: np.asarray(value).copy() for key, value in data.items()}
    if result["action_command"].shape != (1, 7) or result["action_goal_delta"].shape != (1, 6):
        raise ValueError("Exactly one current-state action row required")
    result["action_command"][0, :3] = command
    result["action_goal_delta"][0, :3] = output_min + (command - input_min) * (output_max - output_min) / (
        input_max - input_min
    )
    return result


def affine_response(data, parameters, *, input_min, input_max, output_min, output_max, predictor=predict):
    lo, hi, outlo, outhi = (vector(v, 3) for v in (input_min, input_max, output_min, output_max))
    if np.any(lo >= hi) or np.any(outlo >= outhi):
        raise ValueError("Ordered actual translation bounds required")
    baseline = vector(predictor(branch(data, lo, lo, hi, outlo, outhi), parameters)[0], 12)
    jacobian = np.empty((12, 3))
    for axis in range(3):
        command = lo.copy()
        command[axis] = hi[axis]
        response = vector(predictor(branch(data, command, lo, hi, outlo, outhi), parameters)[0], 12)
        jacobian[:, axis] = (response - baseline) / (outhi[axis] - outlo[axis])
    return dict(offset=baseline - jacobian @ outlo, jacobian=jacobian)


def inverse_translation(
    data,
    parameters,
    desired_motion,
    desired_velocity,
    *,
    input_min,
    input_max,
    output_min,
    output_max,
    previous_command,
    max_rate=5.0,
    predictor=predict,
):
    bounds = dict(input_min=input_min, input_max=input_max, output_min=output_min, output_max=output_max)
    response = affine_response(data, parameters, predictor=predictor, **bounds)
    rows = [0, 1, 2, 6, 7, 8]
    solved = translation_command(
        response["offset"][rows],
        response["jacobian"][rows],
        desired_motion,
        desired_velocity,
        dt=0.05,
        previous_command=previous_command,
        max_rate=max_rate,
        **bounds,
    )
    actual = vector(
        predictor(
            branch(
                data,
                solved["command"],
                *(np.asarray(bounds[k], dtype=float) for k in ("input_min", "input_max", "output_min", "output_max")),
            ),
            parameters,
        )[0],
        12,
    )
    affine = response["offset"] + response["jacobian"] @ solved["physical_delta"]
    if not np.allclose(actual, affine, rtol=1e-8, atol=1e-9):
        raise ValueError("Predictor is not affine in actual translation command")
    return dict(
        **solved,
        full_predicted_response=actual,
        response_offset=response["offset"],
        response_jacobian=response["jacobian"],
    )
