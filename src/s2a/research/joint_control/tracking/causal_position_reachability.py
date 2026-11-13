"""Floating-point outer bounds for a causally corrected first TCP position."""

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.stopped_pose_safety_bound import support_upper_bound


def causal_position_outer_bounds(linearization, translation_innovation, scopes, *, allowance_m=1e-9, on_support=None):
    """Lower-bound corrected position error over each relaxed command domain.

    The fixed direction is the corrected error at the linearization seed. The
    result uses the supplied rotation-exponential remainder and floating-point
    LP dual certificates; it is a bound for that frozen model, not hardware.
    """
    innovation = finite(translation_innovation, (3,))
    target = finite(linearization["target_position"], (3,))
    seed = finite(linearization["seed_tcp_position"], (3,))
    jacobian = finite(linearization["position_jacobian"], (3, 6))
    remainder = linearization["position_remainder_bound_m"]
    if not np.isfinite(remainder) or remainder < 0 or not np.isfinite(allowance_m) or allowance_m < 0:
        raise ValueError("Finite nonnegative linearization and numerical allowances are required")
    seed_error = target - seed + innovation
    norm = float(np.linalg.norm(seed_error))
    if norm <= 1e-15 or not isinstance(scopes, dict) or not scopes:
        raise ValueError("A nonzero corrected seed error and command domains are required")
    direction = seed_error / norm
    first_jacobian = direction @ jacobian
    results = {}
    for name, scope in scopes.items():
        width = len(scope["lower"])
        if width not in (6, 48):
            raise ValueError("A first-command or eight-command domain is required")
        support_direction = np.zeros(width)
        support_direction[:6] = first_jacobian
        if on_support is not None:
            on_support(name, "attempted")
        support = support_upper_bound(support_direction, scope)
        if on_support is not None:
            on_support(name, "completed")
        lower = float(direction @ seed_error - support["support_upper"] - remainder - allowance_m)
        results[name] = {
            "support_direction": support_direction,
            "support": support,
            "position_lower_bound_m": lower,
        }
    return {
        "corrected_seed_error_m": seed_error,
        "position_direction": direction,
        "linearization_remainder_m": float(remainder),
        "numerical_allowance_m": float(allowance_m),
        "scopes": results,
        "future_guarantee": False,
        "scope": "floating-point outer bound for a frozen first-period model",
    }
