"""Check the added margin and future-quality gates independently of SLSQP."""

import numpy as np
import pytest

from s2a.research.trajectory_search.refresh_margin_targeted_search import (
    candidate_is_eligible,
    targeted_constraints,
)


def test_targeted_constraints_only_tighten_second_pose_and_bound_the_suffix():
    values = np.arange(16, dtype=float) / 100
    jacobian = np.arange(16 * 3, dtype=float).reshape(16, 3)
    adjusted, actual_jacobian = targeted_constraints(values, jacobian, 0.08)
    np.testing.assert_allclose(adjusted[:4], values[:4] - [0.05, 0.06, 0.075, 0.09])
    np.testing.assert_allclose(adjusted[4:], values[4:] + 0.10)
    np.testing.assert_array_equal(actual_jacobian, jacobian)


@pytest.mark.parametrize("failure", [None, "first", "second_position", "second_rotation", "future", "safety"])
def test_candidate_must_pass_every_original_and_added_gate(failure):
    errors = np.zeros((8, 6))
    errors[0, 0], errors[0, 3] = 0.0094, 0.046
    errors[1, 0], errors[1, 3] = 0.0092, 0.0454
    future, safety = 0.099, {"accepted": True}
    if failure == "first":
        errors[0, 0] = 0.0096
    elif failure == "second_position":
        errors[1, 0] = 0.0093
    elif failure == "second_rotation":
        errors[1, 3] = 0.0456
    elif failure == "future":
        future = 0.101
    elif failure == "safety":
        safety["accepted"] = False
    assert candidate_is_eligible(errors, future, 0.08, safety) is (failure is None)


def test_constraints_reject_a_negative_or_nonfinite_baseline():
    for baseline in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            targeted_constraints(np.zeros(16), np.zeros((16, 48)), baseline)
