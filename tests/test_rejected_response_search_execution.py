"""Isolated input binding and independent feedback eligibility, without a model."""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from s2a.interior_trajectory_search import DEFAULT_CONFIG


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return (
        importlib.import_module("probe_tracking_v5_rejected_response_search"),
        importlib.import_module("check_tracking_v5_rejected_response_search"),
    )


def bindings(producer, monkeypatch, tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    baseline = dict(
        complete=True, passed=True, sources_unchanged=True, total_response_calls=88, witness_indices=[], input_refs={}
    )
    diagnostic = dict(
        complete=True,
        passed=True,
        experiment="posthoc_rejected_ray_residual_diagnostic",
        independent_dynamics_evidence_added=False,
        new_activity=dict(qp_solves=0, response_calls=0),
        fit=dict(beta=1e90),
    )
    bundle = dict(base="the original3557", probes="the original72 signed probes")
    prior_path, new_path = tmp_path / "baseline.json", tmp_path / "variant.json"
    for path in (prior_path, new_path):
        path.write_text(json.dumps(config))
    monkeypatch.setattr(producer, "BASELINE_CONFIG", prior_path)
    monkeypatch.setattr(producer, "CONFIG", new_path)
    monkeypatch.setattr(producer, "load_baseline_inputs", lambda: (bundle, deepcopy(config), {}))
    monkeypatch.setattr(producer, "reference", lambda path: dict(path=str(path), sha256="a" * 64))
    monkeypatch.setattr(producer, "remember", lambda ref, refs: refs.setdefault(ref["path"], ref))
    monkeypatch.setattr(producer, "finish", lambda refs: None)

    def read(ref):
        if ref == producer.BASELINE_ANALYSIS:
            return baseline
        if ref == producer.DIAGNOSTIC:
            return diagnostic
        return json.loads(Path(ref["path"]).read_text())

    monkeypatch.setattr(producer, "read", read)
    return bundle, config, baseline, diagnostic, new_path


def test_numeric_configuration_is_identical_and_diagnostic_fit_never_preloads_the_model(modules, monkeypatch, tmp_path):
    producer, _ = modules
    bundle, config, _, diagnostic, _ = bindings(producer, monkeypatch, tmp_path)
    loaded, current, refs = producer.load_inputs()
    assert loaded is bundle and current == config
    diagnostic["fit"]["beta"] = -1e100
    second, second_config, _ = producer.load_inputs()
    assert second is bundle and second_config == current
    assert producer.BASELINE_ANALYSIS["path"] in refs and producer.DIAGNOSTIC["path"] in refs


@pytest.mark.parametrize("defect", ["budget", "cosmetic_config_change", "baseline_open", "new_diagnostic_call"])
def test_incomplete_prior_or_changed_budget_cannot_enter_the_experiment(modules, monkeypatch, tmp_path, defect):
    producer, _ = modules
    _, config, baseline, diagnostic, path = bindings(producer, monkeypatch, tmp_path)
    if defect == "budget":
        path.write_text(json.dumps(dict(config, max_responses=123)))
    elif defect == "cosmetic_config_change":
        path.write_text(path.read_text() + "\n")
    elif defect == "baseline_open":
        baseline["complete"] = False
    else:
        diagnostic["new_activity"]["response_calls"] = 1
    with pytest.raises(ValueError):
        producer.load_inputs()


def feedback_case():
    metrics = [
        dict(joint_feasible=True, safety_slack=np.full(4369, 1e-4), normalized_squared_errors=np.full((12, 2), 0.5))
        for _ in range(6)
    ]
    for metric in metrics[1:]:
        metric["normalized_squared_errors"][-1, 0] = 2.0
    attempts = [
        dict(
            response_index=i + 1,
            alpha=2.0**-i,
            accepted=False,
            acceptance_reason="actual_admission_failed",
            actual_step=np.full((12, 6), 1e-4),
        )
        for i in range(5)
    ]
    answer = dict(
        evaluated_candidates=[{}] * 6,
        pose_ceiling=np.ones(24),
        rejected_update_count=1,
        stages=[dict(accepted_response_index=None, attempts=attempts, rejected_update=dict(source_response_index=5))],
    )
    return answer, metrics


def test_independent_feedback_check_selects_the_smallest_actually_safe_alpha(modules):
    _, checker = modules
    answer, metrics = feedback_case()
    checked = checker.verify_fresh_feedback(answer, metrics, DEFAULT_CONFIG)
    assert checked["passed"] and checked["source_response_indices"] == [5]
    assert checked["new_response_calls"] == 0 and not checked["control_qualified"]


@pytest.mark.parametrize(
    "defect", ["unsafe", "audit", "inside_ceiling", "tiny", "uncomputed", "wrong_index", "counter"]
)
def test_fresh_values_can_invalidate_a_saved_feedback_choice(modules, defect):
    _, checker = modules
    answer, metrics = feedback_case()
    if defect == "unsafe":
        metrics[5]["safety_slack"][0] = np.nextafter(1e-8, -np.inf)
    elif defect == "audit":
        metrics[5]["joint_feasible"] = False
    elif defect == "inside_ceiling":
        metrics[5]["normalized_squared_errors"][-1, 0] = 1.0
    elif defect == "tiny":
        answer["stages"][0]["attempts"][-1]["actual_step"] *= 1e-8
    elif defect == "uncomputed":
        answer["stages"][0]["attempts"][-1]["response_index"] = None
    elif defect == "wrong_index":
        answer["stages"][0]["rejected_update"]["source_response_index"] = 1
    else:
        answer["rejected_update_count"] = 0
    with pytest.raises(ValueError):
        checker.verify_fresh_feedback(answer, metrics, DEFAULT_CONFIG)


def test_accepted_stage_cannot_also_consume_a_rejected_feedback_update(modules):
    _, checker = modules
    answer, metrics = feedback_case()
    answer["stages"][0]["accepted_response_index"] = 1
    with pytest.raises(ValueError):
        checker.verify_fresh_feedback(answer, metrics, DEFAULT_CONFIG)
    answer["stages"][0]["rejected_update"] = None
    answer["rejected_update_count"] = 0
    assert checker.verify_fresh_feedback(answer, metrics, DEFAULT_CONFIG)["update_count"] == 0
