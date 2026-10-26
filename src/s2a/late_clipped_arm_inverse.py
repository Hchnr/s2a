"""216 add the frozen inverse residual before the single interface projection."""

import numpy as np

from s2a.direct_arm_inverse import feature_matrix, nominal_translation
from s2a.direct_arm_inverse_inputs import inverse_condition


def unconstrained_nominal_translation(data, desired_site, parameters):
    # Reuse validation only; the bounded result is deliberately not the base.
    nominal_translation(data, desired_site, parameters)
    current = np.asarray(data["x_site_pose"], dtype=float)
    desired = np.asarray(desired_site, dtype=float)
    velocity = np.asarray(data["x_site_twist"], dtype=float)
    response, momentum = (np.asarray(parameters["translation_" + key]) for key in ("response", "momentum"))
    lo, hi, outlo, outhi = (
        np.asarray(parameters["translation_" + key]) for key in ("input_min", "input_max", "output_min", "output_max")
    )
    physical = (desired[:, :3, 3] - current[:, :3, 3] - momentum * velocity[:, :3]) / response
    return lo + (physical - outlo) * (hi - lo) / (outhi - outlo)


def command_prediction(data, desired_site, *, gripper_command, parameters):
    values = inverse_condition(data, desired_site, gripper_command=gripper_command)
    nominal = unconstrained_nominal_translation(data, desired_site, parameters)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (218, 3) or not np.isfinite(coefficients).all():
        raise ValueError("Finite218x3 frozen inverse coefficients required")
    residual = feature_matrix(values, parameters) @ coefficients
    lo, hi = (np.asarray(parameters["translation_" + key]) for key in ("input_min", "input_max"))
    return dict(command=np.clip(nominal + residual, lo, hi), nominal_command=nominal, residual_command=residual)
