"""Independent audit algebra, geometric convention and fail-closed boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("rotation_drift_audit_test", root / "scripts/audit_rotation_drift.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_augmented_ridge_solution_and_radial_cap(audit):
    times = np.asarray([0.05, 0.1, 0.15, 0.2])
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    true_bias = np.asarray([0.1, -0.05, 0.02])
    bias, unconstrained = audit.augmented_solution(times[:, None] * true_bias, times, weights)
    scalar = np.dot(weights, times**2)
    expected = scalar / (scalar + 0.1) * true_bias
    np.testing.assert_allclose(bias, expected, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(bias, unconstrained)
    capped, unbounded = audit.augmented_solution(times[:, None] * true_bias * 100, times, weights)
    assert np.linalg.norm(unbounded) > 0.25
    np.testing.assert_allclose(capped, 0.25 * unbounded / np.linalg.norm(unbounded), atol=1e-15, rtol=0)


def test_independent_correction_uses_world_left_and_preserves_point_and_opening(audit):
    truth = Rotation.from_euler("xyz", [0.4, -0.7, 1.1]).as_matrix()
    bias = np.asarray([0.15, 0.04, -0.1])
    source = np.repeat(np.eye(4)[None], 2, axis=0)
    source[:, :3, 3] = [0.1, -0.2, 0.3]
    for i in range(2):
        source[i, :3, :3] = Rotation.from_rotvec((i + 1) / 20 * bias).as_matrix() @ truth
    openings = np.asarray([1.2, -0.1])
    corrected, actual_opening = audit.corrected_reference(source, openings, bias)
    np.testing.assert_allclose(corrected[:, :3, :3], [truth, truth], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(corrected[:, :3, 3], source[:, :3, 3])
    np.testing.assert_array_equal(actual_opening, openings)
    wrong_side = source[1, :3, :3] @ Rotation.from_rotvec(-0.1 * bias).as_matrix()
    assert Rotation.from_matrix(wrong_side @ truth.T).magnitude() > 1e-3


def test_episode_then_query_then_endpoint_equal_weights(audit):
    cases = [
        dict(name=f"q{i}", key=(ep, i, 42), episode=ep, count=n) for i, (ep, n) in enumerate(((0, 1), (0, 2), (1, 3)))
    ]
    records, weights = audit.records_and_weights([(case, case) for case in cases])
    np.testing.assert_allclose(weights, [0.25, 0.125, 0.125, 1 / 6, 1 / 6, 1 / 6], atol=1e-15, rtol=0)
    assert [record["query_total_weight"] for record in records] == [0.25, 0.25, 0.5]


def test_audit_rejects_missing_fields_shape_changes_and_large_numeric_disagreement(audit):
    checker = audit.Comparison()
    for saved, expected in (({}, {"x": 1}), ([1, 2], [[1, 2]]), (0.1 + 3e-12, 0.1), (False, True)):
        with pytest.raises(AssertionError):
            checker.check(saved, expected, "negative")
    checker.check([0.1 + 1e-13], [0.1], "within_existing_audit_tolerance")


def test_raw_improvement_does_not_override_better_control_or_bad_fold(audit):
    pooled = {
        str(prefix): {
            name: dict(rotation_rmse_rad=value)
            for name, value in dict(
                raw=1.0, corrected=0.85, hold=0.8, constant_velocity=0.95, no_wam=0.9, shuffled=0.99
            ).items()
        }
        for prefix in (8, 16, 40)
    }
    result = audit.decision(pooled, [pooled] * 6)
    assert result["short_prefix_criteria"]["8"]["at_least_ten_percent_raw_improvement"]
    assert not result["short_prefix_criteria"]["8"]["strictly_better_than_best_control_by_tolerance"]
    assert not result["eligible_for_further_validation"]
    assert result["full_candidate_excluded"] and not result["physical_promotion_authorized"]
