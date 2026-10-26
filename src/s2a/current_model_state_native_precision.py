"""Use the fixed controller's actual numeric orientation conversion, read only."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.current_model_state import current_model_state as original_reader


def rotation_goal_delta(delta, current, previous, *, matrix_decoder=None):
    delta = np.asarray(delta, dtype=float)
    current, previous = np.asarray(current, dtype=float), np.asarray(previous, dtype=float)
    if (
        delta.shape != (3,)
        or current.shape != (3, 3)
        or previous.shape != (3, 3)
        or not all(np.isfinite(v).all() for v in (delta, current, previous))
    ):
        raise ValueError("Finite rotation delta and measured/previous orientation required")
    if np.all(delta == 0):
        goal = previous
    else:
        if matrix_decoder is None:
            from robosuite.utils import transform_utils as transform

            def matrix_decoder(value):
                return transform.quat2mat(transform.axisangle2quat(value))

        goal = np.asarray(matrix_decoder(delta)) @ current
    return Rotation.from_matrix(goal @ current.T).as_rotvec()


def current_model_state(env, command, previous_command, *, previous_available, history):
    row, wrench = original_reader(
        env, command, previous_command, previous_available=previous_available, history=history
    )
    controller = env.robots[0].controller
    lo, hi, outlo, outhi = (
        np.asarray(getattr(controller, k), dtype=float) for k in ("input_min", "input_max", "output_min", "output_max")
    )
    delta = outlo + (np.asarray(command)[:6] - lo) * (outhi - outlo) / (hi - lo)
    row["action_goal_delta"][0, 3:] = rotation_goal_delta(delta[3:], row["x_site_pose"][0, :3, :3], controller.goal_ori)
    return row, wrench
