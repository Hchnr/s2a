"""Keep equivalent solver-form comparison bound to the original response inputs."""

import json
from copy import deepcopy
from pathlib import Path

import probe_tracking_v5_state_vector_objective_scale as producer
import pytest


@pytest.fixture
def bound_inputs(monkeypatch):
    config = json.loads(producer.BASELINE_CONFIG.read_text())
    bundle = {"base": {"historical_index": 3557}, "probes": ["original_72_signed_fd"]}
    prior = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        costs={"variant": {"response_calls_per_route": 3, "stage_count": 7}},
        variant=dict(witness_indices=[], selected_response_index=None, termination_reason="radius_below_minimum"),
        source_refs_unchanged=[],
    )
    diagnostic = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        new_activity={"numerical_responses": 0, "qp_solves": 0},
        original_producer_qualified_witness_count=0,
        accepted_stage0_secant_metadata_exact=True,
        all_six_saved_pose_failures=[dict(solver_status=8, solver_success=False, qualified=False) for _ in range(6)],
        prospective_equivalent_scaling=dict(executed=False, objective_divisor=1e99),
        accepted_model_hashes="do_not_load_prior_model",
        source_refs_unchanged=[],
    )
    objects = {
        producer.CONFIG: config,
        Path(producer.CLOSED_PRIOR_ANALYSIS["path"]): prior,
        Path(producer.DIAGNOSTIC_REF["path"]): diagnostic,
    }
    prior_config = deepcopy(config)
    monkeypatch.setattr(producer, "load_prior_inputs", lambda: (bundle, deepcopy(prior_config), {}))
    monkeypatch.setattr(producer, "read", lambda path: objects[Path(path)])
    monkeypatch.setattr(producer, "reference", lambda path: dict(path=str(path), sha256="0" * 64))
    monkeypatch.setattr(producer, "remember", lambda ref, refs: ref["path"])
    monkeypatch.setattr(producer, "finish", lambda refs: None)
    return bundle, config, prior, diagnostic


def test_same_original_inputs_without_loading_final_fit_or_diagnostic_scale(bound_inputs):
    bundle, config, *_ = bound_inputs
    actual, configured, _ = producer.load_inputs()
    assert actual is bundle and configured == config
    assert set(actual) == {"base", "probes"}
    assert producer.CONFIG.read_bytes() == producer.BASELINE_CONFIG.read_bytes()


@pytest.mark.parametrize(
    "position,key,value",
    [
        (2, "passed", False),
        (2, "sources_unchanged", False),
        (3, "passed", False),
        (3, "sources_unchanged", False),
        (3, "accepted_stage0_secant_metadata_exact", False),
        (3, "original_producer_qualified_witness_count", 1),
    ],
)
def test_prior_evidence_must_be_closed_and_failed(bound_inputs, position, key, value):
    bound_inputs[position][key] = value
    with pytest.raises(ValueError):
        producer.load_inputs()


@pytest.mark.parametrize("value", [1, False])
def test_diagnostic_has_no_new_activity_and_boolean_is_not_a_count(bound_inputs, value):
    bound_inputs[3]["new_activity"]["qp_solves"] = value
    with pytest.raises(ValueError, match="zero-activity"):
        producer.load_inputs()


def test_scaled_solve_must_not_have_been_executed_in_motivation_diagnostic(bound_inputs):
    bound_inputs[3]["prospective_equivalent_scaling"]["executed"] = True
    with pytest.raises(ValueError, match="zero-activity"):
        producer.load_inputs()


@pytest.mark.parametrize("key,value", [("solver_status", 0), ("solver_success", True), ("qualified", True)])
def test_old_status8_cannot_be_relabelled_as_qualified(bound_inputs, key, value):
    bound_inputs[3]["all_six_saved_pose_failures"][0][key] = value
    with pytest.raises(ValueError, match="original failures"):
        producer.load_inputs()


@pytest.mark.parametrize("field,value", [("witness_indices", [1]), ("selected_response_index", 1)])
def test_existing_execution_witness_cannot_be_ignored(bound_inputs, field, value):
    bound_inputs[2]["variant"][field] = value
    with pytest.raises(ValueError, match="Closed seven-stage"):
        producer.load_inputs()


@pytest.mark.parametrize(
    "key,value", [("max_stages", 61), ("max_responses", 123), ("accept_ratio", 0.05), ("safety_buffer", 1e-9)]
)
def test_numeric_configuration_and_both_budgets_match(bound_inputs, key, value):
    bound_inputs[1][key] = value
    with pytest.raises(ValueError, match="All numeric gates"):
        producer.load_inputs()


def test_byte_comparison_preserves_config_format_as_registered(bound_inputs, monkeypatch, tmp_path):
    path = tmp_path / "reformatted.json"
    path.write_text(json.dumps(bound_inputs[1], separators=(",", ":")))
    # The config contents remain identical, but registration binds the exact bytes.
    original_read = producer.read
    monkeypatch.setattr(
        producer, "read", lambda value: bound_inputs[1] if Path(value) == path else original_read(value)
    )
    monkeypatch.setattr(producer, "CONFIG", path)
    with pytest.raises(ValueError, match="All numeric gates"):
        producer.load_inputs()
