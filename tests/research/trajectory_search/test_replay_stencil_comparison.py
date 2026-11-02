"""Exercise actual comparison shapes before expensive native evaluation."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values


def test_nested_tuple_slacks_and_boolean_difference_remain_visible():
    actual = dict(groups={"braking": (np.arange(7.0), 1e-10)}, feasible=True)
    rows = compare_values(actual, actual)
    assert all(row["exact"] and row["passed"] for row in rows)
    expected = dict(groups={"braking": (np.arange(7.0) + 1e-6, 1e-10)}, feasible=False)
    rows = compare_values(actual, expected)
    assert any(not row["passed"] for row in rows if "braking" in row["field"])
    assert next(row for row in rows if row["field"] == "/feasible")["passed"] is False


def test_original_torque_tolerance_does_not_spill_into_state_fields():
    actual = dict(requested_torque=np.zeros(7), joint_position=np.zeros(7))
    expected = dict(requested_torque=np.full(7, 1e-9), joint_position=np.full(7, 1e-9))
    rows = compare_values(actual, expected)
    assert rows[0]["passed"] is True
    assert rows[1]["passed"] is False


def test_missing_fields_or_different_shapes_are_not_compared_by_broadcasting():
    with pytest.raises(ValueError, match="fields"):
        compare_values(dict(state=0), {})
    with pytest.raises(ValueError, match="shape"):
        compare_values(np.zeros((7,)), np.zeros((1, 7)))
