"""Verify closing constraints and witnesses against independent box separation."""

import numpy as np
import pytest

from s2a.research.native_runtime.bilateral_box_feasibility import pair_constraints, refine_euclidean, solve_infinity
from s2a.research.native_runtime.box_separation import box_separation


def box(center):
    return dict(center_world_m=np.asarray(center, dtype=float), rotation_world=np.eye(3), half_size_m=np.full(3, 0.01))


def test_symmetric_closing_finds_a_witness_that_sat_independently_verifies():
    left, right = box([-0.1, 0, 0]), box([0.1, 0, 0])
    left_target, right_target = box([-0.05, 0, 0]), box([0.05, 0, 0])
    axis, margin = np.array([1, 0, 0]), 1e-5
    lc, rc = (
        pair_constraints(left, left_target, axis, 1, margin),
        pair_constraints(right, right_target, axis, -1, margin),
    )
    initial = solve_infinity(lc, rc, [0, 0, 0], 0.08)
    assert initial["success"] and initial["infinity_error_m"] == pytest.approx(0, abs=1e-10)
    witness = refine_euclidean(lc, rc, [0, 0, 0], 0.08, initial)
    assert witness["euclidean_error_m"] < 0.01 and witness["maximum_constraint_violation_m"] < 1e-8
    for pad, target, sign in ((left, left_target, 1), (right, right_target, -1)):
        center = pad["center_world_m"] + witness["translation_world_m"] + sign * witness["closing_m"] * axis
        check = box_separation(
            center,
            pad["rotation_world"],
            pad["half_size_m"],
            target["center_world_m"],
            target["rotation_world"],
            target["half_size_m"],
        )
        assert check["maximum_gap_m"] <= -margin + 1e-8


def test_joint_limit_can_make_a_pair_infeasible_and_bad_axes_fail():
    left, right = box([-0.1, 0, 0]), box([0.1, 0, 0])
    axis = [1, 0, 0]
    lc = pair_constraints(left, box([-0.05, 0, 0]), axis, 1, 1e-5)
    rc = pair_constraints(right, box([0.05, 0, 0]), axis, -1, 1e-5)
    result = solve_infinity(lc, rc, [0, 0, 0], 0)
    assert result["success"] is False and result["status"] == 2
    with pytest.raises(ValueError, match="unit jaw axis"):
        pair_constraints(left, right, [2, 0, 0], 1, 1e-5)


def test_random_linear_constraints_agree_with_independent_separating_axes():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(42)
    for _ in range(100):
        pad, target = box(rng.normal(0, 0.02, 3)), box(rng.normal(0, 0.02, 3))
        pad["rotation_world"] = Rotation.random(random_state=rng).as_matrix()
        target["rotation_world"] = Rotation.random(random_state=rng).as_matrix()
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        translation, closing = rng.normal(0, 0.01, 3), float(rng.uniform(0, 0.02))
        direction, margin = 1, 1e-5
        rows, bound = pair_constraints(pad, target, axis, direction, margin)
        residual = float(np.max(rows @ np.r_[translation, closing] - bound))
        moved = pad["center_world_m"] + translation + direction * closing * axis
        check = box_separation(
            moved,
            pad["rotation_world"],
            pad["half_size_m"],
            target["center_world_m"],
            target["rotation_world"],
            target["half_size_m"],
        )
        assert residual == pytest.approx(check["maximum_gap_m"] + margin, abs=1e-10)
