"""Synthetic CPU checks of clean residual signs, layouts, and training-only fit."""

import json

import numpy as np
import pytest

from s2a.wam_clean_readout import (
    CleanResidualReadout,
    active10_from_full128,
    clean_estimate,
    clean_features,
    clean_residual_to_velocity,
    full128_from_active10,
)


def test_actual_fp32_clean_and_velocity_correction_sign_scale_and_terminal():
    sample = np.arange(20, dtype=np.float32).reshape(2, 10) / 8
    base = np.full((2, 10), 0.125, dtype=np.float32)
    sigma = np.asarray([0.2, 0.8], dtype=np.float32)
    saved = [x.copy() for x in (sample, base, sigma)]
    c0 = clean_estimate(sample, base, sigma, layout="active10")
    expected = sample.astype(np.float64) - sigma.astype(np.float64)[:, None] * base
    np.testing.assert_array_equal(c0, expected)
    delta_clean = np.full((2, 10), 0.25)
    delta_v = clean_residual_to_velocity(delta_clean, sigma[:, None], layout="active10")
    assert np.all(delta_v[0] < delta_v[1]) and np.all(delta_v < 0)
    changed_clean = sample.astype(np.float64) - sigma.astype(np.float64)[:, None] * (base + delta_v)
    np.testing.assert_allclose(changed_clean, c0 + delta_clean, atol=1e-15, rtol=0)
    for actual, original in zip((sample, base, sigma), saved, strict=True):
        np.testing.assert_array_equal(actual, original)
    with pytest.raises(ValueError, match="FP32"):
        clean_estimate(sample.astype(np.float64), base, sigma, layout="active10")
    for bad in (0, -0.1, np.nan, np.inf, [0.2, 0], [0.2, 0.5, 0.8]):
        with pytest.raises(ValueError, match="sigma"):
            clean_estimate(sample, base, bad, layout="active10")
        with pytest.raises(ValueError, match="sigma"):
            clean_residual_to_velocity(delta_clean, bad, layout="active10")


def test_explicit_action_layouts_reject_inactive_values_and_bad_dimensions():
    active = np.arange(20, dtype=np.float64).reshape(2, 10)
    full = full128_from_active10(active)
    assert full.shape == (2, 128) and not np.any(full[:, 10:])
    np.testing.assert_array_equal(active10_from_full128(full), active)
    output = clean_estimate(full.astype(np.float32), full, 0.5, layout="full128")
    np.testing.assert_array_equal(output, full / 2)
    correction = clean_residual_to_velocity(full, 0.5, layout="full128")
    np.testing.assert_array_equal(correction[:, :10], -2 * active)
    assert correction[:, 10:].tobytes() == np.zeros((2, 118), dtype=np.float64).tobytes()
    broken = full.copy()
    broken[:, 10] = 1e-30
    with pytest.raises(ValueError, match="inactive118"):
        active10_from_full128(broken)
    with pytest.raises(ValueError, match="inactive118"):
        clean_estimate(full.astype(np.float32), broken, 0.5, layout="full128")
    with pytest.raises(ValueError, match="inactive118"):
        clean_residual_to_velocity(broken, 0.5, layout="full128")
    for bad in (np.zeros((2, 9)), np.zeros((2, 128)), np.zeros(10)):
        with pytest.raises(ValueError, match="N x 10"):
            full128_from_active10(bad)
    with pytest.raises(ValueError, match="layout"):
        clean_residual_to_velocity(active, 0.5, layout="auto")
    with pytest.raises(ValueError, match="matching rows"):
        clean_estimate(np.zeros((2, 10), np.float32), np.zeros((1, 10)), 0.5, layout="active10")


def test_clean_fit_matches_independent_augmented_ls_and_preserves_training_state():
    rng = np.random.default_rng(42)
    hidden, c0, explicit = rng.normal(size=(16, 3)), rng.normal(size=(16, 10)), rng.normal(size=(16, 2))
    features = clean_features(hidden, c0, explicit_features=explicit)
    np.testing.assert_array_equal(features, np.column_stack((hidden, c0, explicit)))
    target = rng.normal(size=(16, 10))
    weights = np.arange(1, 17, dtype=np.float64)
    weights /= weights.sum()
    saved = [a.copy() for a in (hidden, c0, explicit, features, target, weights)]
    model = CleanResidualReadout.fit(features, target, weights, ridge=0.1)
    center = weights @ features
    scale = np.sqrt(np.sum(weights[:, None] * (features - center) ** 2))
    phi = np.column_stack(((features - center) / scale, np.ones(16)))
    expected = np.linalg.lstsq(
        np.vstack((np.sqrt(weights)[:, None] * phi, np.sqrt(0.1) * np.eye(16))),
        np.vstack((np.sqrt(weights)[:, None] * target, np.zeros((16, 10)))),
        rcond=None,
    )[0]
    np.testing.assert_allclose(model.center, center, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(model.scale, scale, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(model.coefficients, expected, rtol=1e-12, atol=1e-14)
    assert np.isclose(np.sum(weights[:, None] * ((features - model.center) / model.scale) ** 2), 1)
    for actual, original in zip((hidden, c0, explicit, features, target, weights), saved, strict=True):
        np.testing.assert_array_equal(actual, original)
    frozen = model.center.copy(), model.coefficients.copy(), model.scale
    validation = np.full((2, features.shape[1]), 1e9)
    clean = model.predict_clean_residual(validation)
    delta_v = model.predict_velocity_delta(validation, [0.25, 0.5], layout="full128")
    np.testing.assert_allclose(delta_v[:, :10], -clean / np.asarray([0.25, 0.5])[:, None])
    assert not np.any(delta_v[:, 10:])
    np.testing.assert_array_equal(model.center, frozen[0])
    np.testing.assert_array_equal(model.coefficients, frozen[1])
    assert model.scale == frozen[2]


def test_parameter_round_trip_copies_and_fit_contracts_reject_silent_changes():
    coefficients, center = np.arange(30, dtype=float).reshape(3, 10), np.asarray([1.0, 2.0])
    model = CleanResidualReadout(coefficients, center, 2.0, 0.1)
    original = json.loads(json.dumps(model.to_parameters()))
    restored = CleanResidualReadout.from_parameters(original)
    coefficients[:] = 0
    center[:] = 0
    original["coefficients"][0][0] = -100
    original["center"][0] = -100
    np.testing.assert_array_equal(restored.coefficients, model.coefficients)
    np.testing.assert_array_equal(restored.center, model.center)
    assert not restored.center.flags.writeable and not restored.coefficients.flags.writeable
    features, target, weights = np.asarray([[0.0], [1.0]]), np.zeros((2, 10)), np.full(2, 0.5)
    for invalid in (0, -1, np.inf, np.nan, True):
        with pytest.raises(ValueError, match="ridge"):
            CleanResidualReadout.fit(features, target, weights, ridge=invalid)
    for bad_weights in ([0.0, 1.0], [0.5, 0.4], [[0.5], [0.5]]):
        with pytest.raises(ValueError, match="weights"):
            CleanResidualReadout.fit(features, target, bad_weights, ridge=0.1)
    with pytest.raises(ValueError, match="global scale"):
        CleanResidualReadout.fit(np.ones((2, 2)), target, weights, ridge=0.1)
    with pytest.raises(ValueError, match="matching rows"):
        clean_features(features, np.zeros((1, 10)))
    with pytest.raises(ValueError, match="N x 2"):
        model.predict_clean_residual(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="registered fields"):
        CleanResidualReadout.from_parameters({**model.to_parameters(), "time_basis": "implicit"})
