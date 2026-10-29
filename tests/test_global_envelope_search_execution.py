"""Bind the global-envelope experiment to unchanged inputs and closed evidence."""

import json
from copy import deepcopy
from pathlib import Path

import probe_tracking_v5_global_envelope_search as producer
import pytest


@pytest.fixture
def bound_inputs(monkeypatch):
    config = json.loads(producer.BASELINE_CONFIG.read_text())
    prior_config = deepcopy(config)
    bundle = {"base": {"historical_index": 3557}, "probes": ["original_72_signed_fd"]}
    prior = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        experiment="rejected_response_search_saved_comparison",
        costs={"variant": {"response_calls_per_route": 52}},
        variant={"witness_indices": [], "selected_response_index": None},
        source_refs_unchanged=[dict(path="/retained/closed/evidence", sha256="0" * 64)],
    )
    seen = []
    diagnostic = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        experiment="posthoc_fixed_global_envelope_counterfactual",
        hypothetical_search_generated=False,
        new_activity={"numerical_responses": 0},
        source_ref=dict(path="/retained/diagnostic/script", sha256="0" * 64),
    )
    owner = dict(
        complete=True,
        passed=True,
        actual_wait=True,
        reaped=True,
        returncode=0,
        signal_sent=False,
        automatic_retry=False,
        result_ref=producer.COUNTERFACTUAL,
        log_ref=dict(path="/retained/diagnostic/log", sha256="0" * 64),
    )

    def read(path):
        if Path(path) == producer.CONFIG:
            return deepcopy(config)
        if str(path) == producer.COUNTERFACTUAL["path"]:
            return diagnostic
        if str(path) == producer.COUNTERFACTUAL_OWNER["path"]:
            return owner
        return prior

    monkeypatch.setattr(producer, "load_prior_inputs", lambda: (bundle, deepcopy(prior_config), {}))
    monkeypatch.setattr(producer, "read", read)
    monkeypatch.setattr(producer, "reference", lambda path: dict(path=str(path), sha256="0" * 64))

    def remember(ref, refs):
        seen.append(ref)
        return ref["path"]

    monkeypatch.setattr(producer, "remember", remember)
    monkeypatch.setattr(producer, "finish", lambda refs: None)
    return bundle, config, prior, seen


def test_new_policy_keeps_original_bundle_and_numeric_configuration(bound_inputs):
    bundle, config, _, seen = bound_inputs
    actual, configured, _ = producer.load_inputs()
    assert actual is bundle
    assert configured == config
    assert producer.CONFIG.read_bytes() == producer.BASELINE_CONFIG.read_bytes()
    assert producer.POSE_CEILING_POLICY == "fixed_initial_global_maximum"
    assert any(ref["path"] == "/retained/closed/evidence" for ref in seen)


@pytest.mark.parametrize("key", ["complete", "passed", "sources_unchanged"])
def test_prior_must_be_closed_and_unchanged(bound_inputs, key):
    bound_inputs[2][key] = False
    with pytest.raises(ValueError, match="Closed prior"):
        producer.load_inputs()


@pytest.mark.parametrize("field,value", [("witness_indices", [40]), ("selected_response_index", 40)])
def test_an_adopted_prior_trajectory_cannot_be_silently_ignored(bound_inputs, field, value):
    bound_inputs[2]["variant"][field] = value
    with pytest.raises(ValueError, match="Closed prior"):
        producer.load_inputs()


def test_changed_response_budget_is_rejected(bound_inputs):
    bound_inputs[1]["max_responses"] += 1
    with pytest.raises(ValueError, match="numeric gates and budgets"):
        producer.load_inputs()


def test_equal_numbers_with_different_config_bytes_are_rejected(bound_inputs, monkeypatch, tmp_path):
    path = tmp_path / "reformatted.json"
    path.write_text(json.dumps(bound_inputs[1], separators=(",", ":")))
    monkeypatch.setattr(producer, "CONFIG", path)
    with pytest.raises(ValueError, match="numeric gates and budgets"):
        producer.load_inputs()


def test_saved_search_responses_and_posthoc_fits_are_not_preloaded(bound_inputs):
    bundle, _, prior, _ = bound_inputs
    prior["evaluated_candidates"] = [{"commands": "not_training_data"}]
    prior["posthoc_beta"] = 1e99
    actual, _, _ = producer.load_inputs()
    assert actual is bundle
    assert set(actual) == {"base", "probes"}
