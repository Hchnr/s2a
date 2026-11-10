"""Both arms share one starting point and retain actual numerical tolerances."""

from copy import deepcopy

import numpy as np
import pytest

from s2a.research.trajectory_search.whole_trajectory_refinement import common_start
from scripts.tracking.trajectory_search.check_whole_trajectory_refinement import arm_bounds, baseline_comparison


def test_matched_domains_share_exact_feasible_warm_and_original_first_rate():
    previous = np.zeros(6)
    lo, hi = -np.ones(6), np.ones(6)
    root_lo, root_hi = np.full((12, 6), -0.2), np.full((12, 6), 0.2)
    warm = np.zeros((12, 6))
    warm[0, 0] = 0.2 + 1e-15
    start = common_start(warm, previous, lo, hi, root_lo, root_hi)
    a, b = arm_bounds("root", previous, lo, hi, root_lo, root_hi)
    c, d = arm_bounds("full", previous, lo, hi, root_lo, root_hi)
    assert np.all(start >= a) and np.all(start <= b) and np.all(start >= c) and np.all(start <= d)
    np.testing.assert_array_equal(c[0], np.full(6, -0.25))
    np.testing.assert_array_equal(d[0], np.full(6, 0.25))
    assert np.all(c <= a) and np.all(d >= b)
    with pytest.raises(ValueError):
        arm_bounds("unregistered", previous, lo, hi, root_lo, root_hi)


def test_historical_warm_compares_strict_labels_and_original_tolerances():
    base = dict(
        rollout=dict(joint_position=np.zeros((12, 7)), torque=np.zeros((12, 7))),
        tcp_errors=np.zeros((12, 6)),
        normalized_squared_errors=np.zeros((12, 2)),
        safety_slack=np.ones(4369),
        audit=dict(feasible=True),
        prefix_length=12,
        feasible=True,
    )
    same = deepcopy(base)
    same["tcp_errors"][0, 0] = 1e-11
    assert baseline_comparison(same, base)["passed"]
    same["tcp_errors"][0, 0] = 2e-10
    with pytest.raises(ValueError):
        baseline_comparison(same, base)
    changed = deepcopy(base)
    changed["audit"]["feasible"] = False
    with pytest.raises(ValueError):
        baseline_comparison(changed, base)
    changed = deepcopy(base)
    changed["normalized_squared_errors"][0, 0] = 1.00000000001
    with pytest.raises((ValueError, AssertionError)):
        baseline_comparison(changed, base)
