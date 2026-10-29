"""Keep the stage-budget comparison bound to its original inputs and response cap."""

import json
from copy import deepcopy
from pathlib import Path

import probe_tracking_v5_stage_budget_search as producer
import pytest


@pytest.fixture
def bound_inputs(monkeypatch):
    prior_config = json.loads(producer.BASELINE_CONFIG.read_text())
    config = dict(prior_config, max_stages=60)
    bundle = {"base": {"historical_index": 3557}, "probes": ["original_72_signed_fd"]}
    prior = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        costs={"variant": {"response_calls_per_route": 35, "stage_count": 24}},
        variant={
            "witness_indices": [],
            "selected_response_index": None,
            "termination_reason": "stage_budget_exhausted",
        },
        source_refs_unchanged=[dict(path="/retained/closed/evidence", sha256="0" * 64)],
    )
    seen = []
    monkeypatch.setattr(producer, "load_prior_inputs", lambda: (bundle, deepcopy(prior_config), {}))
    monkeypatch.setattr(producer, "read", lambda path: deepcopy(config) if Path(path) == producer.CONFIG else prior)
    monkeypatch.setattr(producer, "reference", lambda path: dict(path=str(path), sha256="0" * 64))

    def remember(ref, refs):
        seen.append(ref)
        return ref["path"]

    monkeypatch.setattr(producer, "remember", remember)
    monkeypatch.setattr(producer, "finish", lambda refs: None)
    return bundle, config, prior, seen


def test_only_stage_budget_changes_without_loading_old_anchor(bound_inputs):
    bundle, config, _, seen = bound_inputs
    actual, configured, _ = producer.load_inputs()
    assert actual is bundle and configured == config
    assert configured["max_stages"] == 60 and configured["max_responses"] == 122
    assert producer.CONFIG.read_bytes().replace(b'"max_stages": 60', b'"max_stages": 24') == (
        producer.BASELINE_CONFIG.read_bytes()
    )
    assert any(ref["path"] == "/retained/closed/evidence" for ref in seen)


@pytest.mark.parametrize("key", ["complete", "passed", "sources_unchanged"])
def test_prior_must_be_closed_and_unchanged(bound_inputs, key):
    bound_inputs[2][key] = False
    with pytest.raises(ValueError, match="Closed 24-stage"):
        producer.load_inputs()


@pytest.mark.parametrize("field,value", [("witness_indices", [33]), ("selected_response_index", 33)])
def test_existing_qualified_trajectory_cannot_be_silently_ignored(bound_inputs, field, value):
    bound_inputs[2]["variant"][field] = value
    with pytest.raises(ValueError, match="Closed 24-stage"):
        producer.load_inputs()


@pytest.mark.parametrize("value", [24, 59, 61, True])
def test_exact_registered_stage_budget_is_required(bound_inputs, value):
    bound_inputs[1]["max_stages"] = value
    with pytest.raises(ValueError, match="Only stage cap changes"):
        producer.load_inputs()


@pytest.mark.parametrize("key,value", [("max_responses", 123), ("accept_ratio", 0.05), ("safety_buffer", 1e-9)])
def test_other_gates_and_response_budget_cannot_change(bound_inputs, key, value):
    bound_inputs[1][key] = value
    with pytest.raises(ValueError, match="Only stage cap changes"):
        producer.load_inputs()


def test_numerically_equal_but_other_bytes_changed_is_not_the_single_token_comparison(
    bound_inputs, monkeypatch, tmp_path
):
    path = tmp_path / "reformatted.json"
    path.write_text(json.dumps(bound_inputs[1], separators=(",", ":")))
    monkeypatch.setattr(producer, "CONFIG", path)
    with pytest.raises(ValueError, match="Only stage cap changes"):
        producer.load_inputs()


def test_saved_responses_and_posthoc_fits_are_not_search_initialization(bound_inputs):
    bundle, _, prior, _ = bound_inputs
    prior["evaluated_candidates"] = [{"commands": "old_anchor33"}]
    prior["posthoc_beta"] = 1e99
    actual, _, _ = producer.load_inputs()
    assert actual is bundle and set(actual) == {"base", "probes"}
