"""Exercise the registered fit without reading any validation or native data."""

import numpy as np

from scripts.wam.training.native import causal_opening_event_study as study


def test_registered_identity_and_sources_are_explicit():
    ref, config = study.read_config()
    assert ref["path"].endswith("causal_opening_event_20261111.json")
    assert config["training_demos"] == 30 and config["folds"] == 5
    assert config["world_model_calls"] == config["physics_steps"] == config["fresh_cases"] == 0
    assert any(item["path"].endswith("verify_causal_opening_event.py") for item in study.source_refs())


def test_fold_fit_uses_only_selected_queries():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(12, 40, 44))
    features[:, :, 0] = 1
    truth = np.where(features[:, :, 1] > 0.2, 0.1, 0.9)
    data = {"features": features.copy(), "truth": truth.copy()}
    config = dict(closed_below=0.35, positive_weight=2.0, ridge=0.003, max_iterations=400, gradient_tolerance=1e-6)
    selected = np.arange(12) < 8
    first, diagnostic, independent = study.fit_one(data, config, selected)
    data["truth"][~selected] = 1 - data["truth"][~selected]
    second, _, _ = study.fit_one(data, config, selected)
    np.testing.assert_allclose(first, second, atol=0, rtol=0)
    assert diagnostic["max_abs_gradient"] <= 1e-6
    assert independent["max_finite_difference_error"] < 1e-8
