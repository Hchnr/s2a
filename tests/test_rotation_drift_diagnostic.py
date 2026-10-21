"""Actual admission boundaries and whole-episode grouping, with synthetic data."""

import copy
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "rotation_drift_driver_test", root / "scripts/diagnose_rotation_drift.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol():
    return dict(
        prefixes=[8, 16, 40],
        comparison_tolerance_rad=1e-10,
        minimum_short_raw_relative_improvement=0.1,
        minimum_nonregressing_short_episode_folds=5,
    )


def acceptable_metrics():
    pooled = {}
    for prefix in (8, 16, 40):
        pooled[str(prefix)] = {
            method: dict(rotation_rmse_rad=value)
            for method, value in dict(
                raw=1.0,
                corrected=0.85 if prefix < 40 else 1.0,
                hold=0.9,
                constant_velocity=1.0,
                no_wam=0.95,
                shuffled=1.1,
            ).items()
        }
    folds = [dict(metrics=copy.deepcopy(pooled)) for _ in range(6)]
    for prefix in (8, 16):
        folds[0]["metrics"][str(prefix)]["corrected"]["rotation_rmse_rad"] = 1.1
    return pooled, folds


def test_all_frozen_criteria_needed_and_full_candidate_cannot_promote(diagnostic):
    pooled, folds = acceptable_metrics()
    result = diagnostic.eligibility(pooled, folds, protocol())
    assert result["eligible_for_further_validation"]
    assert result["short_prefix_criteria"]["8"]["nonregressing_episode_folds"] == 5
    assert result["full_candidate_excluded"] and not result["physical_promotion_authorized"]
    assert not result["independent_acceptance_passed"]


@pytest.mark.parametrize("violation", ["best_control", "h40", "ten_percent", "four_folds", "control_margin"])
def test_apparent_short_gain_is_not_enough(diagnostic, violation):
    pooled, folds = acceptable_metrics()
    if violation == "best_control":
        pooled["8"]["no_wam"]["rotation_rmse_rad"] = 0.84
    elif violation == "h40":
        pooled["40"]["corrected"]["rotation_rmse_rad"] = 1 + 1e-9
    elif violation == "ten_percent":
        pooled["8"]["corrected"]["rotation_rmse_rad"] = 0.90000000001
    elif violation == "four_folds":
        folds[1]["metrics"]["16"]["corrected"]["rotation_rmse_rad"] = 1.1
    else:
        pooled["8"]["hold"]["rotation_rmse_rad"] = 0.85 + 5e-11
    assert not diagnostic.eligibility(pooled, folds, protocol())["eligible_for_further_validation"]


def test_entire_episode_seeds_stay_together_and_donor_lengths_define_balanced_weights(diagnostic):
    cases = [
        dict(name=f"ep{episode}_s{seed}", key=(episode, 0, seed), episode=episode, count=20 + episode)
        for episode in range(6)
        for seed in (42, 43)
    ]
    plans = diagnostic.fold_plans(cases, 42)
    seen = []
    for plan in plans:
        held = plan["held_episode"]
        assert len(plan["fit_pairs"]) == 10 and len(plan["test_pairs"]) == 2
        assert all(case["episode"] != held for case, _ in plan["fit_pairs"])
        assert all(case["episode"] == held and donor["episode"] != held for case, donor in plan["test_pairs"])
        episode_weight = defaultdict(float)
        for record in plan["fit_pair_records"]:
            assert record["common_endpoints"] == min(record["original_query_endpoints"], record["donor_endpoints"])
            episode_weight[record["episode"]] += record["query_total_weight"]
        np.testing.assert_allclose(list(episode_weight.values()), np.full(5, 0.2), atol=1e-15, rtol=0)
        seen.extend(case["key"] for case, _ in plan["test_pairs"])
    assert sorted(seen) == sorted(case["key"] for case in cases)
