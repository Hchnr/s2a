import numpy as np
import pytest

from s2a.gripper_motor_state import preserve_motor_target


@pytest.mark.parametrize("cache", [[0.0], [0.4], [0.7, -0.7]])
def test_preserves_native_broadcast_for_all_command_directions(cache):
    original = np.asarray(cache)
    expanded = preserve_motor_target(original)
    for command in (-1, 0, 1):
        native = np.clip(original + np.array([-1.0, 1.0]) * 0.01 * np.sign(command), -1, 1)
        actual = np.clip(expanded + np.array([-1.0, 1.0]) * 0.01 * np.sign(command), -1, 1)
        np.testing.assert_array_equal(actual, native)
    expanded[0] = -0.9
    np.testing.assert_array_equal(original, cache)


@pytest.mark.parametrize("cache", [[], [0, 0, 0], [float("nan")], [1.1]])
def test_invalid_cache_rejected(cache):
    with pytest.raises(ValueError):
        preserve_motor_target(cache)
