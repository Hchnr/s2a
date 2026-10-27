"""208 fixed-feature direct translation inverse with a physical position baseline."""

import numpy as np
from scipy.linalg import expm

from s2a.direct_arm_inverse_inputs import inverse_condition
from s2a.geometry import validate_transform


def physical_parameters(contract):
    if contract["control_freq"] != 20 or contract["control_frame"] != "MuJoCo world":
        raise ValueError("Fixed20Hz world-frame controller required")
    kp, kd = (np.asarray(contract[k], dtype=float) for k in ("kp", "kd"))
    if (
        kp.shape != (6,)
        or kd.shape != (6,)
        or not np.isfinite(kp).all()
        or not np.isfinite(kd).all()
        or np.any(kp <= 0)
        or np.any(kd <= 0)
    ):
        raise ValueError("Finite positive six-axis gains required")
    response = []
    momentum = []
    for p, d in zip(kp[:3], kd[:3], strict=True):
        evolution = expm(np.array([[0.0, 1.0], [-p, -d]]) * 0.05)
        response.append(float(1 - evolution[0, 0]))
        momentum.append(float(evolution[0, 1]))
    result = {
        f"translation_{k}": np.asarray(contract[k], dtype=float)[:3].tolist()
        for k in ("input_min", "input_max", "output_min", "output_max")
    }
    result.update(translation_response=response, translation_momentum=momentum, dt=0.05)
    if np.any(np.asarray(response) < 1e-5):
        raise ValueError("Insufficient physical inverse response")
    return result


def nominal_translation(data, desired_site, parameters):
    current = validate_transform(np.asarray(data["x_site_pose"], dtype=float))
    desired = validate_transform(np.asarray(desired_site, dtype=float))
    velocity = np.asarray(data["x_site_twist"], dtype=float)
    if (
        current.ndim != 3
        or desired.shape != current.shape
        or velocity.shape != (len(current), 6)
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("Aligned finite current state and desired reference required")
    response = np.asarray(parameters["translation_response"])
    momentum = np.asarray(parameters["translation_momentum"])
    lo, hi, outlo, outhi = (
        np.asarray(parameters["translation_" + k], dtype=float)
        for k in ("input_min", "input_max", "output_min", "output_max")
    )
    if (
        any(v.shape != (3,) or not np.isfinite(v).all() for v in (response, momentum, lo, hi, outlo, outhi))
        or np.any(response <= 0)
        or np.any(lo >= hi)
        or np.any(outlo >= outhi)
    ):
        raise ValueError("Valid frozen translation response and affine interface required")
    physical = (desired[:, :3, 3] - current[:, :3, 3] - momentum * velocity[:, :3]) / response
    return np.clip(lo + (physical - outlo) * (hi - lo) / (outhi - outlo), lo, hi)


def feature_matrix(values, parameters):
    values = np.asarray(values, dtype=float)
    mean, scale, frequencies = (
        np.asarray(parameters[k], dtype=float) for k in ("normalization_mean", "normalization_scale", "frequencies")
    )
    if (
        values.ndim != 2
        or values.shape[1] != 89
        or mean.shape != (89,)
        or scale.shape != (89,)
        or frequencies.shape != (64, 89)
        or not all(np.isfinite(v).all() for v in (values, mean, scale, frequencies))
        or np.any(scale <= 0)
    ):
        raise ValueError("Finite inverse89 features, frozen training scales and64 frequencies required")
    z = (values - mean) / scale
    phase = z @ frequencies.T
    return np.c_[np.ones(len(z)), z, np.cos(phase) / 8, np.sin(phase) / 8]


def command_prediction(data, desired_site, *, gripper_command, parameters):
    values = inverse_condition(data, desired_site, gripper_command=gripper_command)
    nominal = nominal_translation(data, desired_site, parameters)
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (218, 3) or not np.isfinite(coefficients).all():
        raise ValueError("Finite218x3 inverse coefficients required")
    residual = feature_matrix(values, parameters) @ coefficients
    lo, hi = (np.asarray(parameters["translation_" + k]) for k in ("input_min", "input_max"))
    return dict(command=np.clip(nominal + residual, lo, hi), nominal_command=nominal, residual_command=residual)
