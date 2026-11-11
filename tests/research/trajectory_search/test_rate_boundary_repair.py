"""Only tiny numerical boundary corrections may enter independent checks."""

import numpy as np
import pytest

from s2a.research.trajectory_search.rate_boundary_repair import repair_nearby_rate_plan


def repair(plan, previous=None):
    return repair_nearby_rate_plan(
        plan,
        np.zeros(6) if previous is None else previous,
        -np.ones(6),
        np.ones(6),
        max_rate=0.25,
        interior=1e-10,
        max_correction=1e-8,
    )


def test_tiny_rate_violation_is_repaired_sequentially_inside_original_limits():
    plan = np.zeros((3, 6))
    plan[:, 0] = [0.250000000006, 0.500000000006, 0.750000000006]
    result = repair(plan)
    corrected = result["planned_commands"]
    assert result["maximum_command_change"] < 1e-8
    assert np.max(np.abs(np.diff(np.vstack((np.zeros(6), corrected)), axis=0))) < 0.25
    assert np.max(np.abs(corrected - plan)) > 0
    np.testing.assert_array_equal(corrected[:, 1:], plan[:, 1:])


def test_material_rate_violation_cannot_be_hidden_by_repair():
    plan = np.zeros((2, 6))
    plan[0, 0] = 0.3
    with pytest.raises(ValueError, match="not close enough"):
        repair(plan)


def test_previous_command_outside_box_is_rejected():
    with pytest.raises(ValueError, match="previous command"):
        repair(np.zeros((1, 6)), previous=np.full(6, 1.01))


def test_correction_is_deterministic_and_does_not_assert_pose_safety():
    plan = np.zeros((2, 6))
    plan[0, 0] = 0.25 + 5e-12
    first, second = repair(plan), repair(plan)
    np.testing.assert_array_equal(first["planned_commands"], second["planned_commands"])
    assert "independent recomputation" in first["scope"]
