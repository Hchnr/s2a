"""Independent023 algebra and strict decision boundaries; synthetic CPU data only."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "generated_reference_audit_test", root / "scripts/audit_generated_reference_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_augmented_fit_matches_single_active_feature_ridge_and_episode_balancing(audit):
    anchor = np.eye(4)
    times = np.arange(1, 4) / 20
    truth = np.repeat(anchor[None], 3, axis=0)
    truth[:, :3, 3] = times[:, None] * [0.02, -0.01, 0.03]
    truth[:, :3, :3] = Rotation.from_rotvec(times[:, None] * [0.1, -0.2, 0.05]).as_matrix()
    case = dict(
        name="synthetic",
        key=(0, 0, 42),
        episode=0,
        count=3,
        anchor=anchor,
        anchor_opening=0.4,
        predicted=np.repeat(anchor[None], 3, axis=0),
        opening=np.full(3, 0.4),
        truth=truth,
        true_opening=0.4 + times * 0.1,
    )
    config = json.loads((audit.ROOT / audit.CONFIG).read_text())
    model, record = audit.augmented_fit([(case, case)], "paired", config)
    tau = times / 2
    denominator = np.mean(tau * tau) + 0.1
    expected = np.zeros((7, 6))
    y = times[:, None] * np.r_[np.array([0.02, -0.01, 0.03]) / 0.1, np.array([0.1, -0.2, 0.05]) / 0.5]
    expected[-1] = np.mean(tau[:, None] * y, axis=0) / denominator
    np.testing.assert_allclose(model["pose_weights"], expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(model["opening_weights"], [0, np.mean(tau * times * 0.1) / denominator], atol=1e-15)
    assert record["training_endpoints"] == 3 and record["labels_are_fit_only"]
    uneven = [
        dict(name=f"q{i}", key=(ep, i, 42), episode=ep, count=n) for i, (ep, n) in enumerate(((0, 1), (0, 2), (1, 3)))
    ]
    _, weights = audit.records_and_weights([(c, c) for c in uneven])
    np.testing.assert_allclose(weights, [0.25, 0.125, 0.125, 1 / 6, 1 / 6, 1 / 6], atol=1e-15)


def test_independent_body_rotation_addition_caps_and_raw_opening(audit):
    anchor = np.eye(4)
    anchor[:3, :3] = Rotation.from_euler("xyz", [0.5, -0.3, 0.7]).as_matrix()
    anchor[:3, 3] = [0.2, -0.1, 0.4]
    source = np.repeat(anchor[None], 2, axis=0)
    source[:, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec([[0.1, -0.2, 0.05], [0.2, 0.1, -0.1]]).as_matrix()
    weights = np.zeros((7, 6))
    weights[-1] = [2, -3, 4, -3, 4, 2]
    model = dict(pose_weights=weights, opening_weights=[100, 100])
    opening = np.asarray([1.2, -0.1])
    output, grip, detail = audit.corrected_reference(dict(anchor=anchor, anchor_opening=0.4), source, opening, model)
    times = np.arange(1, 3) / 20
    np.testing.assert_allclose(np.linalg.norm(detail["translation_correction_m"], axis=1), 0.05 * times, atol=1e-16)
    np.testing.assert_allclose(
        np.linalg.norm(detail["rotation_vector_correction_rad"], axis=1), 0.25 * times, atol=1e-16
    )
    expected = (
        anchor[:3, :3]
        @ Rotation.from_rotvec(
            detail["anchor_relative_rotation_vector_rad"] + detail["rotation_vector_correction_rad"]
        ).as_matrix()
    )
    np.testing.assert_allclose(output[:, :3, :3], expected, atol=1e-15)
    np.testing.assert_array_equal(detail["opening_features"][:, 0], opening - 0.4)
    assert np.all((grip >= 0) & (grip <= 1))
    bad = source.copy()
    bad[0, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec([np.pi - 0.5e-6, 0, 0]).as_matrix()
    with pytest.raises(AssertionError):
        audit.body_features(anchor, 0.4, bad, opening)


def test_decision_has_zero_score_slack_and_does_not_ignore_stronger_control_or_bad_fold(audit):
    pooled = {
        str(prefix): {
            method: dict(position_rmse_m=1.0, rotation_rmse_rad=1.0, opening_mae=1.0) for method in audit.METHODS
        }
        for prefix in (8, 16, 40)
    }
    scores = {
        str(prefix): {
            method: dict(endpoints=10, mean_squared_sum=1.0 if method == "raw" else 0.95) for method in audit.METHODS
        }
        for prefix in (8, 16, 40)
    }
    for prefix in (8, 16, 40):
        scores[str(prefix)]["paired"]["mean_squared_sum"] = 0.9
    folds = [dict(held_episode=episode, normalized_scores=scores) for episode in range(6)]
    assert audit.decision(pooled, scores, folds)["eligible_for_further_validation"]
    scores["8"]["paired"]["mean_squared_sum"] = np.nextafter(0.9, np.inf)
    assert not audit.decision(pooled, scores, folds)["eligible_for_further_validation"]
    scores["8"]["paired"]["mean_squared_sum"] = 0.9
    scores["16"]["no_wam"]["mean_squared_sum"] = 0.85
    assert not audit.decision(pooled, scores, folds)["eligible_for_further_validation"]
    scores["16"]["no_wam"]["mean_squared_sum"] = 0.95
    pooled["40"]["paired"]["opening_mae"] = 1 + 1e-15
    assert not audit.decision(pooled, scores, folds)["eligible_for_further_validation"]
    assert audit.decision(pooled, scores, folds)["added_comparison_tolerance"] == 0
    assert not audit.decision(pooled, scores, folds)["physical_promotion_authorized"]


def test_audit_comparison_rejects_schema_boolean_shape_and_numeric_errors(audit):
    checker = audit.Comparison()
    for actual, expected in (({}, {"x": 1}), (1, True), ([1, 2], [[1, 2]]), (0.1 + 3e-12, 0.1)):
        with pytest.raises(AssertionError):
            checker.check(actual, expected, "negative")
