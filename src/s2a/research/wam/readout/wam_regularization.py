"""Explicit model-output regularization, separate from the S2A response inverse."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, matrix_to_rot6d, rot6d_to_matrix


def attenuate_wam_rotation(actions, *, alpha):
    """Exp(alpha Log(delta_R)) in the generated local delta convention.

    Translation and absolute aperture slots are unchanged; subsequent world
    positions can nevertheless change because local translations integrate
    through the modified orientations. No task labels, expert future, object
    state or current robot pose are inputs. Alpha1 preserves every numeric
    value in the validated floating action array. This changes the WAM
    reference itself, so results must be labeled regularized model outputs,
    never raw-checkpoint performance or pure execution compensation.
    """
    values = finite(actions)
    if values.ndim != 2 or values.shape[1] != 128 or not len(values) or np.any(values[:, 10:] != 0):
        raise ValueError("requires nonempty physical Action128 with ten active slots")
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("rotation attenuation alpha must be in [0,1]")
    rotations = rot6d_to_matrix(values[:, 3:9])
    result = values.copy()
    if alpha != 1:
        vectors = Rotation.from_matrix(rotations).as_rotvec()
        result[:, 3:9] = matrix_to_rot6d(Rotation.from_rotvec(alpha * vectors).as_matrix())
    return result
