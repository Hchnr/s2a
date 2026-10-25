import numpy as np
import pytest

from s2a.native_gripper_response import intervention_commands


def test_only_eight_gripper_commands_change_and_source_is_not_mutated():
    source = np.arange(40 * 7, dtype=float).reshape(40, 7) / 1000
    original = source.copy()
    changed = intervention_commands(source, 10, -1.0)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(changed[:, :6], source[:34, :6])
    np.testing.assert_array_equal(changed[:10], source[:10])
    np.testing.assert_array_equal(changed[18:], source[18:34])
    assert np.all(changed[10:18, -1] == -1)
    np.testing.assert_array_equal(intervention_commands(source, 10, None), source[:34])


@pytest.mark.parametrize("anchor,override", [(True, 0), (-1, 0), (20, 0), (0, True), (0, 0.5)])
def test_invalid_registration_cannot_silently_shorten_or_interpolate(anchor, override):
    with pytest.raises(ValueError):
        intervention_commands(np.zeros((40, 7)), anchor, override)
