"""Check OBB separation against analytic cases and independent linear feasibility."""

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from s2a.research.native_runtime.box_separation import box_separation


def test_parallel_boxes_and_degenerate_cross_axes():
    identity = np.eye(3)
    separated = box_separation([0, 0, 0], identity, [1, 1, 1], [3, 0, 0], identity, [1, 1, 1])
    assert separated["maximum_gap_m"] == pytest.approx(1)
    assert separated["tested_axes"] == 12
    overlapping = box_separation([0, 0, 0], identity, [1, 1, 1], [1.5, 0, 0], identity, [1, 1, 1])
    assert overlapping["maximum_gap_m"] == pytest.approx(-0.5)
    touching = box_separation([0, 0, 0], identity, [1, 1, 1], [2, 0, 0], identity, [1, 1, 1])
    assert touching["maximum_gap_m"] == pytest.approx(0)
    with pytest.raises(ValueError, match="proper orthogonal"):
        box_separation([0, 0, 0], -identity, [1, 1, 1], [2, 0, 0], identity, [1, 1, 1])


def test_random_boxes_match_independent_linear_feasibility():
    rng = np.random.default_rng(42)
    for _ in range(100):
        ca, cb = rng.normal(size=(2, 3)) * 0.08
        ra, rb = Rotation.random(random_state=rng).as_matrix(), Rotation.random(random_state=rng).as_matrix()
        ha, hb = rng.uniform(0.01, 0.08, size=(2, 3))
        before = tuple(item.copy() for item in (ca, ra, ha, cb, rb, hb))
        result = box_separation(ca, ra, ha, cb, rb, hb)
        matrices = np.vstack((ra.T, -ra.T, rb.T, -rb.T))
        limits = np.r_[ha + ra.T @ ca, ha - ra.T @ ca, hb + rb.T @ cb, hb - rb.T @ cb]
        feasible = linprog(np.zeros(3), A_ub=matrices, b_ub=limits, bounds=[(None, None)] * 3, method="highs")
        gap = result["maximum_gap_m"]
        if gap < -1e-7:
            assert feasible.success
        elif gap > 1e-7:
            assert feasible.status == 2
        for item, saved in zip((ca, ra, ha, cb, rb, hb), before, strict=True):
            np.testing.assert_array_equal(item, saved)
