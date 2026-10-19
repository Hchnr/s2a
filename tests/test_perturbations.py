import numpy as np
import pytest

from s2a.perturbations import ActuatorDelay


def test_actuator_delay_is_causal_and_keeps_issued_commands_immutable():
    delay = ActuatorDelay(2, dimensions=1)
    command = np.array([1.0])
    np.testing.assert_array_equal(delay.update(command), [0])
    command[:] = 2
    np.testing.assert_array_equal(delay.update(command), [0])
    np.testing.assert_array_equal(delay.update([3]), [1])
    np.testing.assert_array_equal(delay.update([4]), [2])
    np.testing.assert_array_equal(ActuatorDelay(0, dimensions=1).update([5]), [5])
    with pytest.raises(ValueError, match="delay"):
        ActuatorDelay(-1)
    delay = ActuatorDelay(1, dimensions=2, initial_command=[0.5, -0.5])
    np.testing.assert_array_equal(delay.update([1, -1]), [0.5, -0.5])
    np.testing.assert_array_equal(delay.update([0, 0]), [1, -1])
