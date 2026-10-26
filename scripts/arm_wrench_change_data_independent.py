"""112 independent per-episode differences after sensor-frame reconstruction."""

import numpy as np
from arm_wrench_data_independent import attach_current_wrench as previous_attach


def attach_current_wrench(data, entry, config, e):
    output = previous_attach(data, entry, config, e)
    values = output["x_current_wrench_world"]
    flags = np.asarray(data["x_previous_command_available"])
    assert flags.shape == (len(values),) and not flags[0] and flags[1:].all()
    differences = np.zeros_like(values)
    for row in range(1, len(values)):
        for axis in range(6):
            differences[row, axis] = values[row, axis] - values[row - 1, axis]
    output["x_wrench_change_world"] = differences
    return output
