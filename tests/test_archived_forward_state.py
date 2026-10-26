import numpy as np
import pytest

from s2a.archived_forward_state import verify_forward_state


def fixture():
    state = np.zeros(16)
    state[0] = 0.25
    state[1] = 0.2
    state[5:9] = np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0]) * (1 + np.finfo(float).eps)
    forwarded = state.copy()
    forwarded[5:9] /= np.linalg.norm(forwarded[5:9])
    return state, forwarded, [dict(joint="object_free", qpos_start=4)]


def test_archived_unit_quaternion_normalization_is_recorded_with_equivalent_orientation():
    before, after, segments = fixture()
    changes = verify_forward_state(before, after, nq=8, quaternion_segments=segments)
    assert len(changes) == 1 and changes[0]["joint"] == "object_free"
    assert changes[0]["max_component_difference"] < 1e-15


def test_time_translation_hinge_or_velocity_changes_are_never_tolerated():
    before, after, segments = fixture()
    for index in (0, 1, 2, 9, 15):
        corrupted = after.copy()
        corrupted[index] += 1e-15
        with pytest.raises(AssertionError):
            verify_forward_state(before, corrupted, nq=8, quaternion_segments=segments)


def test_quaternion_motion_or_large_nonunit_correction_is_rejected():
    before, after, segments = fixture()
    moved = after.copy()
    moved[7] += 1e-6
    moved[5:9] /= np.linalg.norm(moved[5:9])
    with pytest.raises(AssertionError):
        verify_forward_state(before, moved, nq=8, quaternion_segments=segments)
    nonunit = before.copy()
    nonunit[5:9] *= 1.01
    with pytest.raises(AssertionError):
        verify_forward_state(nonunit, after, nq=8, quaternion_segments=segments)
