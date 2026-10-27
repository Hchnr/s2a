"""Chain physical trajectory objectives to a normalized linear residual readout.

Only xyz and opening columns are optimized. Before-step rotations are fixed
external inputs; gradients through predicted rotation are deliberately absent.
"""

import numpy as np

from s2a.geometry import finite
from s2a.trajectory_objective import opening_loss, position_loss


def readout_objectives(
    features,
    coefficients,
    center,
    feature_scale,
    raw_clean,
    action_scale,
    action_bias,
    before_rotation,
    initial_position,
    desired_positions,
    desired_opening,
    endpoint_weights,
):
    """Return separate physical losses/parameter gradients for one trajectory.

    Effective action_scale/bias include identity (1,0) for unnormalized slots.
    This is ideal FP64 inverse normalization without clipping, not a derivative
    through the actual mixed-precision scheduler. The caller binds the transform
    to normalization31 and supplies independently frozen predicted rotations.
    """
    x, coef, center, feature_scale = (finite(a) for a in (features, coefficients, center, feature_scale))
    if x.ndim != 2 or not len(x) or x.shape[1] == 0:
        raise ValueError("Nonempty feature matrix required")
    width = x.shape[1]
    if coef.shape != (width + 1, 10) or center.shape != (width,) or feature_scale.ndim != 0 or feature_scale <= 0:
        raise ValueError("Readout dimensions or scalar scale differ")
    raw, scale, bias = (finite(a) for a in (raw_clean, action_scale, action_bias))
    if raw.shape != (len(x), 10) or scale.shape != (10,) or bias.shape != (10,) or (scale == 0).any():
        raise ValueError("Active10 values and invertible normalization required")
    phi = np.column_stack(((x - center) / feature_scale, np.ones(len(x))))
    physical = (raw + phi @ coef - bias) / scale
    pos_loss, pos_grad = position_loss(
        physical[:, :3],
        before_rotation,
        initial_position,
        desired_positions,
        endpoint_weights,
    )
    grip_loss, grip_grad = opening_loss(physical[:, 9], desired_opening, endpoint_weights)
    pos_coef_grad, grip_coef_grad = np.zeros_like(coef), np.zeros_like(coef)
    pos_coef_grad[:, :3] = phi.T @ (pos_grad / scale[:3])
    grip_coef_grad[:, 9] = phi.T @ (grip_grad / scale[9])
    return dict(position=pos_loss, opening=grip_loss), dict(position=pos_coef_grad, opening=grip_coef_grad)
