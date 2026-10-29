"""Exact mass-metric projection shortcut for one strictly active face."""

import numpy as np

from s2a.joint_limit_preview import _project as general_projection


def project(mass, inverse_mass, value, constraints, bounds, initial):
    """Use a one-face KKT witness, otherwise preserve the general QP path.

    This does not approximate impacts or soften constraints. Strict inactive
    margins avoid treating a degenerate multi-face boundary as a smooth face.
    The caller computes inverse_mass once for its fixed mass matrix.
    """
    slack = bounds - constraints @ value
    if not len(bounds) or np.all(slack >= 0):
        return general_projection(mass, value, constraints, bounds, initial)
    violated = np.flatnonzero(slack < 0)
    if len(violated) == 1:
        k = int(violated[0])
        normal = constraints[k]
        direction = inverse_mass @ normal
        denominator = normal @ direction
        if denominator > 0:
            multiplier = -slack[k] / denominator
            candidate = value - multiplier * direction
            others = np.arange(len(bounds)) != k
            if multiplier > 1e-10 and np.all(bounds[others] - constraints[others] @ candidate > 1e-10):
                derivative = np.eye(len(value)) - np.outer(direction, normal) / denominator
                return candidate, derivative, False
    return general_projection(mass, value, constraints, bounds, initial)
