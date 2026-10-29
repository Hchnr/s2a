"""Compare the cheaper audit predicates with the original NumPy assertions."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparisons(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_tracking_v5_state_updated_trajectory_control_v2")


def accepted(check, *args, **kwargs):
    try:
        check(*args, **kwargs)
    except (AssertionError, ValueError):
        return False
    return True


def original_absolute_check(actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Original finite matching-shape condition")
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def optimized_absolute_check(comparisons, actual, expected, tolerance):
    difference = comparisons.finite_max_abs_difference(actual, expected, name="test quantity")
    comparisons.require_absolute_tolerance(difference, tolerance, name="test quantity")


@pytest.mark.parametrize("tolerance", [1e-10, 1e-8])
def test_adjacent_ulp_values_on_both_sides_of_absolute_tolerance(comparisons, tolerance):
    cases = (
        (np.nextafter(tolerance, 0.0), True),
        (tolerance, True),
        (np.nextafter(tolerance, np.inf), False),
    )
    for difference, expected_acceptance in cases:
        for sign in (-1.0, 1.0):
            actual, expected = np.array([0.0, sign * difference]), np.zeros(2)
            assert accepted(original_absolute_check, actual, expected, tolerance) is expected_acceptance
            assert accepted(optimized_absolute_check, comparisons, actual, expected, tolerance) is expected_acceptance


@pytest.mark.parametrize("tolerance", [1e-10, 1e-8])
def test_arrays_at_different_scales_keep_zero_relative_tolerance(comparisons, tolerance):
    rng = np.random.default_rng(42)
    outcomes = []
    for scale in (1e-12, 1.0, 1e4, 1e16):
        expected = scale * rng.uniform(-1.0, 1.0, (7, 7))
        for change in (0.0, 0.9 * tolerance, 1.1 * tolerance):
            actual = expected.copy()
            actual[3, 5] += change
            reference = accepted(original_absolute_check, actual, expected, tolerance)
            assert accepted(optimized_absolute_check, comparisons, actual, expected, tolerance) is reference
            outcomes.append(reference)
        next_representable = np.nextafter(expected, np.inf)
        reference = accepted(original_absolute_check, next_representable, expected, tolerance)
        assert accepted(optimized_absolute_check, comparisons, next_representable, expected, tolerance) is reference
    assert any(outcomes) and not all(outcomes)
    # At this magnitude an ULP is far outside either absolute tolerance;
    # a nonzero relative tolerance would silently accept the difference.
    expected = np.array([1e16])
    actual = np.nextafter(expected, np.inf)
    assert not accepted(original_absolute_check, actual, expected, tolerance)
    assert not accepted(optimized_absolute_check, comparisons, actual, expected, tolerance)


def test_equal_scalars_vectors_empty_arrays_and_shape_mismatches(comparisons):
    for expected in (np.array(2.0), np.array([-0.0, 2.0]), np.empty((0, 7)), np.eye(3, dtype=np.float32)):
        actual = expected.copy()
        assert comparisons.finite_max_abs_difference(actual, expected) == 0.0
        assert accepted(original_absolute_check, actual, expected, 0.0)
        assert accepted(optimized_absolute_check, comparisons, actual, expected, 0.0)
    for actual, expected in (
        (np.zeros((2, 1)), np.zeros((1, 2))),
        (np.empty((0, 7)), np.empty((0, 6))),
        (np.array(0.0), np.zeros(1)),
    ):
        assert not accepted(original_absolute_check, actual, expected, 1e-10)
        assert not accepted(optimized_absolute_check, comparisons, actual, expected, 1e-10)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_comparisons_are_rejected_even_when_both_arrays_match(comparisons, nonfinite):
    for actual, expected in (
        ([nonfinite, 1.0], [0.0, 1.0]),
        ([0.0, 1.0], [nonfinite, 1.0]),
        ([nonfinite, 1.0], [nonfinite, 1.0]),
    ):
        assert not accepted(original_absolute_check, actual, expected, 1e-10)
        assert not accepted(optimized_absolute_check, comparisons, actual, expected, 1e-10)


def test_overflowing_difference_of_finite_coordinates_is_rejected(comparisons):
    largest = np.finfo(np.float64).max
    actual, expected = np.array([largest]), np.array([-largest])
    with np.errstate(over="ignore", invalid="ignore"):
        assert not accepted(original_absolute_check, actual, expected, 1e-8)
        assert not accepted(optimized_absolute_check, comparisons, actual, expected, 1e-8)


def test_exact_query_checks_match_original_for_finite_arrays(comparisons):
    for actual, expected in (
        (np.array([-0.0, 1.0]), np.array([0.0, 1.0])),
        (np.arange(7, dtype=np.float64), np.arange(7, dtype=np.int32)),
        (np.zeros((0, 7)), np.zeros((0, 7))),
        (np.array([1.0, 2.0]), np.array([1.0, 3.0])),
        (np.zeros((2, 1)), np.zeros((1, 2))),
        (np.empty((0, 7)), np.empty((0, 6))),
    ):
        reference = accepted(np.testing.assert_array_equal, actual, expected)
        assert accepted(comparisons.require_finite_array_equal, actual, expected) is reference
    expected = np.array([1e-12, -1.0, 1e16])
    for index in range(len(expected)):
        actual = expected.copy()
        actual[index] = np.nextafter(actual[index], np.inf)
        assert not accepted(np.testing.assert_array_equal, actual, expected)
        assert not accepted(comparisons.require_finite_array_equal, actual, expected)


def test_exact_query_check_intentionally_rejects_matching_nonfinite_state(comparisons):
    for nonfinite in (np.nan, np.inf, -np.inf):
        value = np.array([0.0, nonfinite])
        # NumPy's old exact check accepted matching NaNs/infinities. The new
        # precondition rejects such invalid physical states before comparison.
        assert accepted(np.testing.assert_array_equal, value, value.copy())
        assert not accepted(comparisons.require_finite_array_equal, value, value.copy())


def test_invalid_tolerances_or_nonfinite_reduced_differences_never_pass(comparisons):
    for difference, tolerance in (
        (0.0, -1e-10),
        (0.0, np.nan),
        (0.0, np.inf),
        (np.nan, 1e-10),
        (np.inf, 1e-10),
        (-1e-12, 1e-10),
    ):
        assert not accepted(comparisons.require_absolute_tolerance, difference, tolerance)


def test_quaternion_normalization_tolerance_does_not_relax_scalar_state_equality(comparisons):
    quaternion_indices = np.arange(1, 5)
    scalar_indices = np.array([0, 5])
    before = np.array([2.816, 0.8, 0.2, -0.4, 0.4, 0.02])
    before[quaternion_indices] *= 1.0 + 4.0 * np.finfo(float).eps
    expected_quaternion = before[quaternion_indices] / np.linalg.norm(before[quaternion_indices])
    after = before.copy()
    after[quaternion_indices] = expected_quaternion
    after[1] = np.nextafter(after[1], np.inf)
    assert not np.array_equal(after[quaternion_indices], before[quaternion_indices])
    for query in (np.testing.assert_array_equal, comparisons.require_finite_array_equal):
        assert accepted(query, after[scalar_indices], before[scalar_indices])
    assert accepted(original_absolute_check, after[quaternion_indices], expected_quaternion, 1e-10)
    assert accepted(optimized_absolute_check, comparisons, after[quaternion_indices], expected_quaternion, 1e-10)

    perturbed = after.copy()
    perturbed[1] += 2e-10
    assert not accepted(original_absolute_check, perturbed[quaternion_indices], expected_quaternion, 1e-10)
    assert not accepted(
        optimized_absolute_check, comparisons, perturbed[quaternion_indices], expected_quaternion, 1e-10
    )

    perturbed = after.copy()
    perturbed[0] = np.nextafter(perturbed[0], np.inf)
    assert not accepted(np.testing.assert_array_equal, perturbed[scalar_indices], before[scalar_indices])
    assert not accepted(comparisons.require_finite_array_equal, perturbed[scalar_indices], before[scalar_indices])
