"""Bind the prospective vector experiment and independently classify fresh merit."""

import json
from copy import deepcopy
from pathlib import Path

import check_tracking_v5_state_vector_gn_search as checker
import numpy as np
import probe_tracking_v5_state_vector_gn_search as producer
import pytest


@pytest.fixture
def bound_inputs(monkeypatch):
    config = json.loads(producer.BASELINE_CONFIG.read_text())
    bundle = {"base": {"historical_index": 3557}, "probes": ["original_72_signed_fd"]}
    prior = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        costs={"variant": {"response_calls_per_route": 90, "stage_count": 60}},
        variant=dict(witness_indices=[], selected_response_index=None, termination_reason="stage_budget_exhausted"),
        source_refs_unchanged=[],
    )
    diagnostic = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        new_activity={"responses": 0, "qp_solves": 0},
        control_qualified=False,
        source_refs_unchanged=[],
        final_Je="must_not_initialize_search",
    )
    owner = dict(
        complete=True,
        passed=True,
        actual_wait=True,
        reaped=True,
        returncode=0,
        signal_sent=False,
        automatic_retry=False,
        source_unchanged=True,
        input_hashes_unchanged=True,
        result_ref=producer.DIAGNOSTIC_REF,
    )
    objects = {
        producer.CONFIG: config,
        Path(producer.CLOSED_PRIOR_ANALYSIS["path"]): prior,
        Path(producer.DIAGNOSTIC_REF["path"]): diagnostic,
        Path(producer.DIAGNOSTIC_OWNER_REF["path"]): owner,
    }
    prior_config = deepcopy(config)
    monkeypatch.setattr(producer, "load_prior_inputs", lambda: (bundle, deepcopy(prior_config), {}))
    monkeypatch.setattr(producer, "read", lambda path: objects[Path(path)])
    monkeypatch.setattr(producer, "reference", lambda path: dict(path=str(path), sha256="0" * 64))
    monkeypatch.setattr(producer, "remember", lambda ref, refs: ref["path"])
    monkeypatch.setattr(producer, "finish", lambda refs: None)
    return bundle, config, prior, diagnostic, owner


def test_same_inputs_and_bytes_without_saved_final_model(bound_inputs):
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
        (3, "control_qualified", True),
        (4, "actual_wait", False),
        (4, "reaped", False),
        (4, "returncode", 1),
        (4, "returncode", False),
        (4, "automatic_retry", True),
        (4, "source_unchanged", False),
    ],
)
def test_all_prior_evidence_must_be_closed(bound_inputs, position, key, value):
    bound_inputs[position][key] = value
    with pytest.raises(ValueError):
        producer.load_inputs()


def test_motivation_diagnostic_may_not_execute_new_activity(bound_inputs):
    bound_inputs[3]["new_activity"]["responses"] = 1
    with pytest.raises(ValueError, match="zero-activity"):
        producer.load_inputs()


@pytest.mark.parametrize("field,value", [("witness_indices", [88]), ("selected_response_index", 88)])
def test_prior_witness_cannot_be_silently_discarded(bound_inputs, field, value):
    bound_inputs[2]["variant"][field] = value
    with pytest.raises(ValueError, match="Closed 60-stage"):
        producer.load_inputs()


@pytest.mark.parametrize(
    "key,value", [("max_stages", 61), ("max_responses", 123), ("accept_ratio", 0.05), ("safety_buffer", 1e-9)]
)
def test_no_numeric_gate_or_budget_change(bound_inputs, key, value):
    bound_inputs[1][key] = value
    with pytest.raises(ValueError, match="All numeric gates"):
        producer.load_inputs()


def decision_fixture(monkeypatch, base_q, trial_q, predicted, reason, accepted):
    config = json.loads(producer.CONFIG.read_text())
    safety = np.full(4369, 0.01)

    def metric(q):
        q = np.asarray(q).reshape(12, 2)
        return dict(
            normalized_squared_errors=q, safety_slack=safety.copy(), joint_feasible=True, feasible=bool(np.all(q <= 1))
        )

    metrics = [metric(base_q), metric(trial_q), metric(base_q)]
    actual = 0.5 * (np.sum(base_q) - np.sum(trial_q))
    ratio = actual / predicted
    commands = np.zeros((12, 6))
    step = np.zeros((12, 6))
    step[0, 0] = 0.005
    radius = 0.005
    if not accepted or ratio < config["shrink_ratio"]:
        radius *= 0.5
    elif ratio >= config["expand_ratio"]:
        radius *= 2
    attempt = dict(
        response_index=1,
        predicted_reduction=predicted,
        actual_reduction=actual,
        ratio=ratio,
        commands=commands + step,
        accepted=accepted,
        acceptance_reason=reason,
    )
    answer = dict(
        pose_ceiling=np.full(24, max(1, np.max(base_q)) * 1.01),
        evaluated_candidates=[dict(commands=commands), dict(commands=commands + step), dict(commands=commands)],
        stages=[dict(anchor_before_index=0, mode="pose", radius_before=0.005, radius_after=radius, attempts=[attempt])],
        initial_comparison=dict(exact=True),
        repeatability=dict(exact=True),
        qualified_witness_indices=[1] if metrics[1]["feasible"] else [],
    )
    monkeypatch.setattr(checker, "_comparison_report", lambda *args: dict(exact=True))
    return answer, [{}, {}, {}], metrics, config


def test_fresh_loss_decrease_accepts_even_when_worst_error_increases(monkeypatch):
    base = np.full(24, 0.2)
    base[0] = 4
    trial = np.full(24, 0.1)
    trial[0] = 4.02
    args = decision_fixture(monkeypatch, base, trial, 1.0, "accepted_reduction", True)
    assert checker.verify_fresh_decisions(*args)["passed"]


def test_fresh_loss_increase_rejects_even_when_worst_error_decreases(monkeypatch):
    base = np.full(24, 0.1)
    base[0] = 4
    trial = np.full(24, 0.3)
    trial[0] = 3.9
    args = decision_fixture(monkeypatch, base, trial, 1.0, "actual_reduction_insufficient", False)
    assert checker.verify_fresh_decisions(*args)["passed"]


def test_fresh_original_witness_keeps_priority_over_loss(monkeypatch):
    base = np.full(24, 0.001)
    base[0] = 1.01
    trial = np.full(24, 0.9)
    args = decision_fixture(monkeypatch, base, trial, 0.01, "original_witness", True)
    assert checker.verify_fresh_decisions(*args)["qualified_witness_indices"] == [1]


def test_fresh_near_cap_threshold_uses_F_not_sum_loss(monkeypatch):
    base = np.full(24, 0.9)
    base[0] = 1.00001
    trial = base.copy()
    trial[1] -= 4e-6
    args = decision_fixture(monkeypatch, base, trial, 2e-6, "accepted_reduction", True)
    assert checker.verify_fresh_decisions(*args)["passed"]


@pytest.mark.parametrize("failure", ["joint", "buffer", "ceiling"])
def test_fresh_admission_failure_precedes_improvement_and_ratio(monkeypatch, failure):
    base = np.full(24, 0.2)
    base[0] = 4
    trial = np.full(24, 0.1)
    trial[0] = 5 if failure == "ceiling" else 4.02
    args = decision_fixture(monkeypatch, base, trial, 100, "actual_admission_failed", False)
    if failure == "joint":
        args[2][1]["joint_feasible"] = False
    elif failure == "buffer":
        args[2][1]["safety_slack"][0] = args[3]["safety_buffer"] * 0.5
    assert checker.verify_fresh_decisions(*args)["passed"]


@pytest.mark.parametrize(
    "ratio,accepted,radius",
    [
        (0.099, False, 0.0025),
        (0.1, True, 0.0025),
        (0.249, True, 0.0025),
        (0.25, True, 0.005),
        (0.749, True, 0.005),
        (0.75, True, 0.01),
    ],
)
def test_fresh_ratio_and_radius_boundaries(monkeypatch, ratio, accepted, radius):
    base = np.full(24, 0.2)
    base[0] = 4
    trial = base.copy()
    trial[0] = 3.9
    actual = 0.5 * (np.sum(base) - np.sum(trial))
    args = decision_fixture(
        monkeypatch, base, trial, actual / ratio, "accepted_reduction" if accepted else "ratio_insufficient", accepted
    )
    assert args[0]["stages"][0]["radius_after"] == radius
    assert checker.verify_fresh_decisions(*args)["passed"]


@pytest.mark.parametrize(
    "key,value",
    [("actual_reduction", 10), ("ratio", 10), ("acceptance_reason", "original_witness"), ("accepted", False)],
)
def test_fresh_classification_rejects_tampered_values(monkeypatch, key, value):
    base = np.full(24, 0.2)
    base[0] = 4
    trial = np.full(24, 0.1)
    trial[0] = 4.02
    args = decision_fixture(monkeypatch, base, trial, 1, "accepted_reduction", True)
    args[0]["stages"][0]["attempts"][0][key] = value
    with pytest.raises(ValueError):
        checker.verify_fresh_decisions(*args)
