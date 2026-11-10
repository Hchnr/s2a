"""Only completed-period data may seed the refiner reconstruction."""

import numpy as np
import pytest

from scripts.simulation.runtime.reconstruct_early_refiner import completed_from_prefix


def completed_row():
    command = [0.1, 0, 0, 0, 0, 0, 0]
    return {
        "step": 2,
        "command": command,
        "coupled_response_history": {"innovation_after": [0.1] * 12, "completed_periods_after": 3},
        "tracking_history_after": [-0.2] * 12,
        "tracking_refinement": {"planned_commands": [command[:6], [0.2, 0, 0, 0, 0, 0]]},
    }


def test_restore_inputs_are_only_the_completed_prefix_values():
    row = completed_row()
    result = completed_from_prefix([{}, {}, row], np.asarray([[0] * 7, [0] * 7, row["command"]]))
    assert result.completed_periods == 3
    np.testing.assert_array_equal(result.previous_command, row["command"])
    np.testing.assert_array_equal(result.coupled_innovation, row["coupled_response_history"]["innovation_after"])
    np.testing.assert_array_equal(result.tracking_innovation, row["tracking_history_after"])


def test_reject_mismatch_of_completed_action_and_replay():
    row = completed_row()
    commands = np.zeros((3, 7))
    with pytest.raises(ValueError, match="previous command"):
        completed_from_prefix([{}, {}, row], commands)


def test_reject_unobserved_or_wrong_period_history():
    row = completed_row()
    row["coupled_response_history"]["completed_periods_after"] = 2
    with pytest.raises(ValueError, match="completed periods"):
        completed_from_prefix([{}, {}, row], np.asarray([[0] * 7, [0] * 7, row["command"]]))
