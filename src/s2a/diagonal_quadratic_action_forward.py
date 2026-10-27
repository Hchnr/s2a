"""Original116 condition basis plus diagonal quadratic translation dependence."""

import numpy as np

from s2a.joint_wrench_condition_forward import design as linear_design
from s2a.joint_wrench_condition_forward import nominal, raw_condition, targets, with_history

__all__ = ["design", "nominal", "raw_condition", "targets", "with_history", "predict", "expand"]


def expand(linear):
    linear = np.asarray(linear, dtype=float)
    if linear.ndim != 2 or linear.shape[1] != 1128 or not len(linear) or not np.isfinite(linear).all():
        raise ValueError("Finite nonempty Nx1128 linear response design required")
    phi = linear[:, :282]
    if not np.allclose(phi[:, 0], 1, rtol=0, atol=1e-12):
        raise ValueError("Physical branch design with unit bias required, not differenced equations")
    action = linear[:, [282, 564, 846]]
    for i in range(3):
        if not np.allclose(linear[:, 282 * (i + 1) : 282 * (i + 2)], phi * action[:, i, None], rtol=1e-10, atol=1e-10):
            raise ValueError("Matching fixed condition basis and affine action blocks required")
    return np.c_[linear, *(phi * action[:, i, None] ** 2 for i in range(3))]


def design(data, parameters):
    return expand(linear_design(data, parameters))


def predict(data, parameters):
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    if coefficients.shape != (1974, 12) or not np.isfinite(coefficients).all():
        raise ValueError("Finite1974x12 coefficients required")
    return nominal(data, parameters) + design(data, parameters) @ coefficients
