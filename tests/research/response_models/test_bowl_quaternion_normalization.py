"""Only harmless free/ball quaternion normalization may change saved qpos."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tracking.response_models.early_box_face_bowl_effect import check_forward_qpos


def example():
    model = SimpleNamespace(nq=12, njnt=3, jnt_type=np.array([0, 1, 3]), jnt_qposadr=np.array([0, 7, 11]))
    before = np.array([1.0, 2.0, 3.0, 1.0 + 2e-16, 0, 0, 0, 1.0, 0, 0, 0, 0.2])
    after = before.copy()
    after[3:7] /= np.linalg.norm(after[3:7])
    return model, before, after


def test_roundoff_quaternion_normalization_preserves_orientation():
    model, before, after = example()
    result = check_forward_qpos(model, before, after, free_type=0, ball_type=1)
    assert result["changed_quaternions"] == 1
    assert result["maximum_qpos_change"] < 1e-12


def test_nonquaternion_changes_or_different_rotations_are_rejected():
    model, before, after = example()
    shifted = after.copy()
    shifted[0] += 1e-10
    with pytest.raises(AssertionError):
        check_forward_qpos(model, before, shifted, free_type=0, ball_type=1)
    rotated = after.copy()
    rotated[4] = 1e-5
    with pytest.raises(AssertionError):
        check_forward_qpos(model, before, rotated, free_type=0, ball_type=1)
    zero = before.copy()
    zero[7:11] = 0
    with pytest.raises(ValueError, match="nonzero"):
        check_forward_qpos(model, zero, after, free_type=0, ball_type=1)
