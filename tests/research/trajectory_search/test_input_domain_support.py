"""The domain diagnostic may alter offsets, never physical row coefficients."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search.analyze_input_domain_support import input_domain_rows
from scripts.tracking.trajectory_search.analyze_local_tracking_support import support_rows


def fixture():
    entry = dict(commands=np.zeros((12, 6)), safety_slack=np.full(4369, 0.001))
    cfg = dict(safety_buffer=1e-8, reserve_target=1e-5, terminal_time_scale=0.1)
    lower, upper = np.full((12, 6), -0.1), np.full((12, 6), 0.1)
    A, b = support_rows(entry, np.zeros((4369, 72)), np.zeros(6), lower, upper, cfg, 0.9)
    return A, b, entry, lower, upper, -np.ones(6), np.ones(6), cfg


@pytest.mark.parametrize("fraction", [0.0, 0.9])
def test_only_registered_offsets_change_without_mutating_inputs(fraction):
    args = fixture()
    A, b, lower, upper = input_domain_rows(*args, fraction)
    np.testing.assert_array_equal(A, args[0])
    np.testing.assert_array_equal(b[4369:4513], args[1][4369:4513])
    np.testing.assert_array_equal(lower, -np.ones(72))
    np.testing.assert_array_equal(upper, np.ones(72))
    assert b[4368] == pytest.approx(0.001 - 1e-8 - fraction * 1e-4)
    np.testing.assert_allclose(args[1][-144:], 0.1)
    np.testing.assert_allclose(b[-144:], 1.0)


@pytest.mark.parametrize("fault", ["shrink", "nan", "reversed", "partial", "matrix", "fraction"])
def test_invalid_relaxation_is_rejected(fault):
    args = list(fixture())
    fraction = 0.9
    if fault == "shrink":
        args[5][0] = 0
    elif fault == "nan":
        args[6][0] = np.nan
    elif fault == "reversed":
        args[5][0] = 2
    elif fault == "partial":
        args[6] = args[6][:-1]
    elif fault == "matrix":
        args[0][-1, -1] = 0
    else:
        fraction = 0.5
    with pytest.raises((ValueError, AssertionError)):
        input_domain_rows(*args, fraction)
