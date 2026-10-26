import numpy as np
import pytest

from s2a.paired_arm_response import paired_arm_commands


def test_only_one_translation_channel_at_anchor_changes_and_source_is_immutable():
    original = np.arange(35, dtype=float).reshape(5, 7) / 100
    saved = original.copy()
    result = paired_arm_commands(original, 2, 1, -0.1)
    expected = original[:3].copy()
    expected[2, 1] -= 0.1
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_array_equal(result[:, 3:], original[:3, 3:])


def test_interface_clipping_is_explicit_and_baseline_keeps_native_prefix():
    original = np.zeros((3, 7))
    original[1, 0] = 0.97
    result = paired_arm_commands(original, 1, 0, 0.1)
    assert result[1, 0] == 1 and result[1, 0] - original[1, 0] == pytest.approx(0.03)
    np.testing.assert_array_equal(paired_arm_commands(original, 1, None, 0), original[:2])


@pytest.mark.parametrize(
    "anchor,axis,offset", [(True, 0, 0.1), (3, 0, 0.1), (0, 3, 0.1), (0, True, 0.1), (0, None, 0.1), (0, 0, 0.2)]
)
def test_unregistered_interventions_are_rejected(anchor, axis, offset):
    with pytest.raises(ValueError):
        paired_arm_commands(np.zeros((3, 7)), anchor, axis, offset)
