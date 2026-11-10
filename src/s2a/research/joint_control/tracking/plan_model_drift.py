"""Separate a completed plan's state error from frozen-model prediction drift."""

import numpy as np

from s2a.geometry import finite


def shifted_completed_plan(period):
    """Shift only a plan whose first command was actually completed."""
    plan = finite(period["decision"]["refined_plan"], (8, 6))
    command = finite(period["command"], (7,))
    if not np.array_equal(plan[0], command[:6]):
        raise ValueError("The saved plan must start with the actual completed command")
    shifted = np.vstack((plan[1:], plan[-1]))
    if (
        np.max(abs(shifted)) > 1 + 1e-12
        or np.max(abs(np.diff(np.vstack((command[:6], shifted)), axis=0))) > 0.25 + 1e-12
    ):
        raise ValueError("The completed plan must retain the original box and rate")
    return shifted.copy()


def compare_joint_endpoint(predicted_q, predicted_velocity, actual_q, actual_velocity):
    """Record signed position and velocity model errors at an observed endpoint."""
    predicted_q, predicted_velocity, actual_q, actual_velocity = (
        finite(value, (7,)) for value in (predicted_q, predicted_velocity, actual_q, actual_velocity)
    )
    q_error, velocity_error = actual_q - predicted_q, actual_velocity - predicted_velocity
    return dict(
        joint_position_error_rad=q_error,
        joint_velocity_error_rad_s=velocity_error,
        maximum_joint_position_error_rad=float(np.max(abs(q_error))),
        maximum_joint_velocity_error_rad_s=float(np.max(abs(velocity_error))),
    )


def compare_next_pose_forecasts(original, position_reanchored, velocity_updated, dynamics_refreshed):
    """Decompose SO(3)-log error-vector changes across fixed diagnostic variants."""
    original, position_reanchored, velocity_updated, dynamics_refreshed = (
        finite(value, (6,)) for value in (original, position_reanchored, velocity_updated, dynamics_refreshed)
    )
    components = dict(
        position_and_site_reanchor=position_reanchored - original,
        velocity_update=velocity_updated - position_reanchored,
        mass_and_jacobian_refresh=dynamics_refreshed - velocity_updated,
    )
    total = dynamics_refreshed - original
    np.testing.assert_allclose(sum(components.values()), total, atol=1e-12, rtol=0)
    return dict(
        original=original.copy(),
        position_reanchored=position_reanchored.copy(),
        velocity_updated=velocity_updated.copy(),
        dynamics_refreshed=dynamics_refreshed.copy(),
        components=components,
        total_difference=total,
        component_norms={
            name: dict(position_m=float(np.linalg.norm(value[:3])), rotation_rad=float(np.linalg.norm(value[3:])))
            for name, value in components.items()
        },
        scope="frozen-model counterfactual error-vector decomposition; not a physical causal intervention",
    )
