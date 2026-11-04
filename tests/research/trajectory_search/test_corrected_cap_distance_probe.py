"""Frozen full-objective trial admission without clipping or resampling."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from scripts.tracking.trajectory_search.probe_corrected_cap_distance import ALPHAS, fixed_trials


def data():
    x = np.zeros((12, 6))
    d = x.copy()
    d[-1, 0] = 0.005
    return x, d, np.zeros(6), -np.ones(6), np.ones(6), -np.ones((12, 6)) * 0.1, np.ones((12, 6)) * 0.1


def test_every_fixed_scale_is_retained_with_exact_repeat():
    x, d, p, lo, hi, lower, upper = data()
    seen = {b"earlier"}
    cases = fixed_trials(x, d, p, lo, hi, lower, upper, seen)
    assert len(cases) == 7
    assert [c["alpha"] for c in cases] == [0.0] + ALPHAS + [0.0]
    for case in cases:
        assert_array_equal(case["commands"], x + case["alpha"] * d)
    assert seen == {b"earlier"}
    assert_array_equal(d[-1], [0.005, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("fault", ["global", "rate", "trust", "zero", "duplicate", "nan", "partial"])
def test_failed_case_aborts_without_replacement(fault):
    x, d, p, lo, hi, lower, upper = data()
    seen = set()
    if fault == "global":
        upper[-1, 0] = 0.004
    elif fault == "rate":
        p[0] = 0.3
    elif fault == "trust":
        d[-1, 0] = 0.006
    elif fault == "zero":
        d[:] = 0
    elif fault == "duplicate":
        seen.add((x + d).tobytes())
    elif fault == "nan":
        d[-1, 0] = np.nan
    else:
        d = np.zeros((11, 6))
    before = d.copy()
    with pytest.raises(ValueError):
        fixed_trials(x, d, p, lo, hi, lower, upper, seen)
    assert_array_equal(d, before)
