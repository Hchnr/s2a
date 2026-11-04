"""Support certificates distinguish one-cap impossibility from joint feasibility."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search.analyze_local_tracking_support import (
    audit_support,
    box_corrected_bound,
    solve_support,
    support_rows,
)

SPEC = dict(lp_time_limit_seconds=10, lp_max_iterations=10000, certificate_tolerance=1e-8)


def test_box_correction_covers_inexact_stationarity_and_negative_dual():
    A, b, c = np.array([[1.0], [-1.0]]), np.array([0.5, 1.0]), np.array([2.0])
    r = box_corrected_bound(A, b, c, np.array([1.9, -0.1]), np.array([-0.5]), np.array([1.0]))
    assert r["bound"] == pytest.approx(-1)
    assert r["residual_box_correction"] == pytest.approx(-0.05)


def test_signed_dual_and_original_unit_rescaling():
    A, b, c = np.array([[1000.0], [-1.0]]), np.array([500.0, 1.0]), np.array([2.0])
    r = solve_support(A, b, c, SPEC)
    assert r["success"]
    assert r["certificate"]["dual_bound"] == pytest.approx(-1)
    np.testing.assert_allclose(r["point"], [-0.5])
    np.testing.assert_allclose(r["dual"], [0.002, 0])


@pytest.mark.parametrize("fault", ["point", "negative_dual", "stationarity", "nan"])
def test_certificate_rejects_corruption(fault):
    A, b, c, x, y = (
        np.array([[1.0], [-1.0]]),
        np.array([0.5, 1.0]),
        np.array([2.0]),
        np.array([-0.5]),
        np.array([2.0, 0.0]),
    )
    if fault == "point":
        x[0] = -1
    elif fault == "negative_dual":
        y[1] = -1
    elif fault == "stationarity":
        y[0] = 1
    else:
        y[0] = np.nan
    with pytest.raises(ValueError):
        audit_support(A, b, c, x, y)


def test_support_excludes_only_when_optimistic_bound_above_one():
    A, b, c = np.array([[1.0], [-1.0]]), np.array([0.1, 0.1]), np.array([1.0])
    r = solve_support(A, b, c, SPEC)
    assert 1.2 + r["certificate"]["dual_bound"] > 1
    # Separate caps may each be satisfiable with incompatible directions.
    positive = solve_support(A, np.array([1.0, 1.0]), c, SPEC)
    negative = solve_support(A, np.array([1.0, 1.0]), -c, SPEC)
    assert positive["point"][0] < 0 < negative["point"][0]


def test_all_original_rows_terminal_weight_and_previous_command():
    entry = dict(commands=np.zeros((12, 6)), safety_slack=np.ones(4369) * 1e-3)
    cfg = dict(terminal_time_scale=0.1, safety_buffer=1e-8, reserve_target=1e-5)
    A, b = support_rows(entry, np.zeros((4369, 72)), np.ones(6) * 0.1, -np.ones((12, 6)), np.ones((12, 6)), cfg, 0.9)
    assert A.shape == (4657, 72) and b.shape == (4657,)
    assert b[0] == pytest.approx(0.001 - 1e-8 - 0.9e-5)
    assert b[4368] == pytest.approx(0.001 - 1e-8 - 0.9e-4)
    np.testing.assert_allclose(b[4369:4375], 0.15)
    np.testing.assert_allclose(b[4441:4447], 0.35)
    np.testing.assert_array_equal(A[-144:-72], np.eye(72))
