"""Fit-only fold causality, raw opening supervision and all023 admission gates."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def driver(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "generated_reference_diagnostic_test", root / "scripts/diagnose_generated_reference_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol(driver):
    return json.loads((driver.REPO / driver.CONFIG).read_text())


def synthetic_cases():
    from s2a.reference_gains import relative_features

    result = []
    for episode in range(6):
        for seed in (42, 43):
            count = 12 + episode
            t = np.arange(1, count + 1) / 20
            anchor = np.eye(4)
            anchor[:3, :3] = Rotation.from_euler("xyz", [0.02 * episode, -0.1, 0.2]).as_matrix()
            anchor[:3, 3] = [0.2, -0.1, 0.5]
            predicted = np.repeat(anchor[None], count, axis=0)
            predicted[:, :3, 3] += (t[:, None] * np.array([0.01, 0.02, -0.01])) @ anchor[:3, :3].T
            truth = predicted.copy()
            truth[:, :3, 3] += [0.002, -0.001, 0]
            result.append(
                dict(
                    name=f"ep{episode}_s{seed}",
                    key=(episode, 0, seed),
                    episode=episode,
                    task=f"task{episode // 2}",
                    phase="one",
                    count=count,
                    anchor=anchor,
                    anchor_opening=0.6,
                    predicted=predicted,
                    opening=np.linspace(1.2, -0.2, count),
                    truth=truth,
                    true_opening=np.full(count, 0.7),
                    history=np.repeat(anchor[None], count, axis=0),
                    predicted_features=relative_features(anchor, predicted),
                )
            )
    return result


def test_whole_episode_weights_shared_donors_and_raw_opening_supervision(driver):
    cases, config = synthetic_cases(), protocol(driver)
    plans = driver.fold_plans(cases, 42)
    seen = []
    for plan in plans:
        held = plan["held_episode"]
        assert held not in plan["fit_episodes"]
        assert all(case["episode"] == held and donor["episode"] != held for case, donor in plan["test_pairs"])
        arrays, records = driver.fit_arrays(plan["fit_pairs"], "paired", config)
        per_episode = {}
        for record in records:
            per_episode[record["episode"]] = per_episode.get(record["episode"], 0) + record["query_total_weight"]
        np.testing.assert_allclose(list(per_episode.values()), np.full(5, 0.2), atol=1e-15, rtol=0)
        assert len(arrays["input_queries"]) == len(plan["fit_pairs"])
        np.testing.assert_allclose(arrays["endpoint_weights"].sum(), 1, atol=1e-15)
        first_case, first_donor = plan["fit_pairs"][0]
        count = min(first_case["count"], first_donor["count"])
        np.testing.assert_array_equal(arrays["opening_features"][:count, 0], first_case["opening"][:count] - 0.6)
        np.testing.assert_array_equal(arrays["opening_residuals"][:count], 0.7 - first_case["opening"][:count])
        assert arrays["opening_features"][0, 0] > 0.5
        seen.extend(tuple(case["key"]) for case, _ in plan["test_pairs"])
    assert sorted(seen) == sorted(tuple(case["key"]) for case in cases)


def test_held_labels_and_reporting_labels_cannot_change_fitted_parameters(driver):
    cases, config = synthetic_cases(), protocol(driver)
    plan = driver.fold_plans(cases, 42)[0]
    original = driver.score_fold(plan, config, [])
    altered = copy.deepcopy(cases)
    for case in altered:
        case["task"], case["phase"] = "arbitrary task", "another phase"
        if case["episode"] == plan["held_episode"]:
            case["truth"][:, :3, 3] += [0.1, -0.03, 0.05]
            case["true_opening"][:] = 0.1
    changed = driver.score_fold(driver.fold_plans(altered, 42)[0], config, [])
    for method in ("paired", "no_wam", "shuffled"):
        assert original["fitted_models"][method]["model"] == changed["fitted_models"][method]["model"]
    assert original["metrics"] != changed["metrics"]


def acceptable(driver):
    metrics, scores = {}, {}
    for h in (8, 16, 40):
        metrics[str(h)] = {
            method: dict(position_rmse_m=1.0, rotation_rmse_rad=1.0, opening_mae=1.0) for method in driver.METHODS
        }
        scores[str(h)] = {method: dict(endpoints=1, mean_squared_sum=1.0) for method in driver.METHODS}
        scores[str(h)]["paired"]["mean_squared_sum"] = 0.9
    folds = [dict(held_episode=i, normalized_scores=copy.deepcopy(scores)) for i in range(6)]
    return metrics, scores, folds


def test_all_components_controls_and_both_short_windows_per_episode_are_required(driver):
    metrics, scores, folds = acceptable(driver)
    config = protocol(driver)
    decision = driver.eligibility(metrics, scores, folds, config)
    assert decision["eligible_for_further_validation"] and decision["full_candidate_excluded"]
    assert decision["added_comparison_tolerance"] == 0 and not decision["physical_promotion_authorized"]
    violations = []
    changed = copy.deepcopy(metrics)
    changed["40"]["paired"]["opening_mae"] += 1e-15
    violations.append((changed, scores, folds))
    changed = copy.deepcopy(scores)
    changed["8"]["paired"]["mean_squared_sum"] = 0.900000000001
    violations.append((metrics, changed, folds))
    changed = copy.deepcopy(scores)
    changed["16"]["shuffled"]["mean_squared_sum"] = 0.9
    violations.append((metrics, changed, folds))
    changed = copy.deepcopy(folds)
    changed[0]["normalized_scores"]["8"]["paired"]["mean_squared_sum"] = 1.01
    assert driver.eligibility(metrics, scores, changed, config)["eligible_for_further_validation"]
    changed[1]["normalized_scores"]["16"]["paired"]["mean_squared_sum"] = 1.01
    violations.append((metrics, scores, changed))
    for values in violations:
        assert not driver.eligibility(*values, config)["eligible_for_further_validation"]


def test_score_uses_per_endpoint_normalized_si_sum_and_opening_metric_has_shared_clip(driver):
    output = dict(
        methods={
            method: dict(position_errors_m=[0.01] * 40, rotation_errors_rad=[0.05] * 40, opening_errors=[0.1] * 40)
            for method in driver.METHODS
        }
    )
    scores = driver.scores([output], [8, 16, 40])
    assert scores["8"]["paired"] == dict(endpoints=8, mean_squared_sum=3.0)
    case = synthetic_cases()[0]
    clipped = driver.shared_metrics(case, case["predicted"], case["opening"], case["count"])
    expected = driver.metrics(case, case["predicted"], np.clip(case["opening"], 0, 1), case["count"])
    assert clipped == expected
