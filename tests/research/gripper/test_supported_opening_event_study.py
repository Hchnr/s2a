"""Check frozen study identity and holdout-only probability reconstruction."""

import copy

import numpy as np
import pytest

from scripts.wam.training.native import supported_opening_event_study as study


def frozen_folds():
    data = {"features": np.zeros((300, 40, 44)), "fold": np.repeat(np.arange(5), 60)}
    manifest_ref = {"path": "manifest", "sha256": "fixed"}
    folds = []
    for number in range(5):
        folds.append(
            dict(
                complete=True,
                passed=True,
                fold=number,
                training_manifest_ref=manifest_ref,
                train_queries=240,
                held_queries=60,
                validation_future_reads=0,
                held_indices=np.flatnonzero(data["fold"] == number),
                held_probability=np.full((60, 40), 0.5),
                coefficients=np.zeros(44),
            )
        )
    cv = {"fold_refs": list(range(5)), "oof_probability": np.full((300, 40), 0.5)}
    return data, manifest_ref, cv, folds


def test_registered_study_retains_original_budget_and_thresholds():
    _ref, config, old = study.read_config()
    assert config["support_future_steps"] == 16
    assert config["thresholds"] == old["thresholds"]
    assert config["new_fold_fits"] == config["world_model_calls"] == config["physics_steps"] == 0
    assert len(study.source_refs()) >= 10


def test_reconstruction_rejects_wrong_fold_or_unverified_probability():
    data, manifest_ref, cv, folds = frozen_folds()
    probability, difference = study.reconstruct_oof(data, manifest_ref, cv, lambda ref: folds[ref])
    assert probability.shape == (300, 40) and difference == 0
    changed = copy.deepcopy(folds)
    changed[0]["held_indices"][0] = 60
    with pytest.raises(ValueError, match="held-demo"):
        study.reconstruct_oof(data, manifest_ref, cv, lambda ref: changed[ref])
    changed = copy.deepcopy(folds)
    changed[1]["held_probability"][0, 0] = 0.6
    with pytest.raises(ValueError, match="probabilities"):
        study.reconstruct_oof(data, manifest_ref, cv, lambda ref: changed[ref])
