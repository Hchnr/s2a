"""Causal feature inputs, grouped splits and the opening-event objective."""

import numpy as np
import pytest

from s2a.research.gripper.causal_opening_event import (
    apply_opening_event,
    event_features,
    fit_event_model,
    group_folds,
    loss_and_gradient,
    predict_probability,
)


def test_features_use_only_measured_anchor_and_generated_reference():
    anchor = np.eye(4)
    poses = np.repeat(anchor[None], 40, axis=0)
    poses[:, 0, 3] = np.linspace(0.001, 0.04, 40)
    opening = np.full(40, 0.8)
    before = poses.copy()
    values = event_features(anchor, 0.9, poses, opening)
    assert values.shape == (40, 44) and np.isfinite(values).all()
    np.testing.assert_array_equal(poses, before)
    changed = opening.copy()
    changed[-1] = 0.2
    # The complete generated chunk is already available at the query.
    assert not np.array_equal(values[0], event_features(anchor, 0.9, poses, changed)[0])
    with pytest.raises(ValueError):
        event_features(anchor, 0.9, poses[:39], opening[:39])


def test_logistic_gradient_matches_finite_differences():
    rng = np.random.default_rng(42)
    x, y, theta = rng.normal(size=(100, 44)), rng.integers(0, 2, 100), rng.normal(size=44) / 10
    _, gradient = loss_and_gradient(theta, x, y, positive_weight=2, ridge=0.003)
    for index in (0, 1, 11, 43):
        step = np.zeros(44)
        step[index] = 1e-6
        a = loss_and_gradient(theta + step, x, y, positive_weight=2, ridge=0.003)[0]
        b = loss_and_gradient(theta - step, x, y, positive_weight=2, ridge=0.003)[0]
        assert gradient[index] == pytest.approx((a - b) / 2e-6, abs=1e-8)


def test_fit_is_finite_and_closure_does_not_modify_raw_closed_values():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(400, 44))
    x[:, 0] = 1
    closed = (x[:, 1] - 0.4 * x[:, 2] > 0.5).astype(float)
    coefficients, info = fit_event_model(
        x, closed, positive_weight=2, ridge=0.003, max_iterations=400, gradient_tolerance=1e-6
    )
    assert info["max_abs_gradient"] <= 1e-6
    assert np.isfinite(predict_probability(x, coefficients)).all()
    raw = np.array([0.2, 0.6, 0.9])
    result = apply_opening_event(raw, np.array([0.99, 0.4, 0.9]), 0.8)
    np.testing.assert_allclose(result, [0.2, 0.6, 0.1])
    np.testing.assert_array_equal(raw, [0.2, 0.6, 0.9])


def test_grouped_folds_are_stable_balanced_and_episode_isolated():
    identities = [(task, f"demo_{index}") for task in range(3) for index in range(10)]
    assignment = group_folds(identities)
    assert assignment == group_folds(list(reversed(identities)))
    for task in range(3):
        for fold in range(5):
            assert sum(assignment[(task, f"demo_{index}")] == fold for index in range(10)) == 2
    with pytest.raises(ValueError):
        group_folds(identities[:-1] + identities[:1])
