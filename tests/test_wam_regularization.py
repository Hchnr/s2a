"""Regularization is explicit, local, and not an executor or task planner."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import integrate_local, matrix_to_rot6d, rot6d_to_matrix
from s2a.wam_regularization import attenuate_wam_rotation


def example():
    actions = np.zeros((2, 128))
    actions[:, :3] = [0.01, 0, 0]
    actions[:, 3:9] = matrix_to_rot6d(Rotation.from_rotvec([[0, 0, 0.8], [0.2, 0, 0]]).as_matrix())
    actions[:, 9] = [0.7, 0.2]
    return actions


def test_alpha_one_reproduces_raw_including_nonorthogonal_columns_and_no_alias():
    values = example()
    values[:, 3:9] *= 0.99
    raw = values.copy()
    result = attenuate_wam_rotation(values, alpha=1)
    np.testing.assert_array_equal(result, raw)
    result[0, 0] += 1
    np.testing.assert_array_equal(values, raw)


def test_local_rotation_attenuation_preserves_other_slots_but_changes_integrated_world_path():
    values = example()
    result = attenuate_wam_rotation(values, alpha=0.5)
    np.testing.assert_array_equal(result[:, :3], values[:, :3])
    np.testing.assert_array_equal(result[:, 9:], values[:, 9:])
    np.testing.assert_allclose(
        Rotation.from_matrix(rot6d_to_matrix(result[:, 3:9])).as_rotvec(), [[0, 0, 0.4], [0.1, 0, 0]], atol=1e-12
    )
    raw_path = integrate_local(np.eye(4), values[:, :9])
    calibrated_path = integrate_local(np.eye(4), result[:, :9])
    np.testing.assert_array_equal(raw_path[1, :3, 3], calibrated_path[1, :3, 3])
    assert not np.allclose(raw_path[2, :3, 3], calibrated_path[2, :3, 3])


def test_alpha_zero_is_rotation_hold_diagnostic():
    result = attenuate_wam_rotation(example(), alpha=0)
    np.testing.assert_array_equal(rot6d_to_matrix(result[:, 3:9]), np.repeat(np.eye(3)[None], 2, axis=0))


def test_invalid_mask_or_alpha_cannot_silently_change_the_model_contract():
    values = example()
    for alpha in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError):
            attenuate_wam_rotation(values, alpha=alpha)
    values[0, 30] = 0.01
    with pytest.raises(ValueError):
        attenuate_wam_rotation(values, alpha=0.5)
