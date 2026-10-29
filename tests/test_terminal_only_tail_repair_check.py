"""Independent fixed-grid bookkeeping and fresh strict gates without native dynamics."""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_terminal_only_tail_repair import case

from s2a.directional_derivative_probe import _canonical
from s2a.terminal_only_tail_repair import probe_terminal_only_tail_repair


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_tracking_v5_terminal_only_tail_repair")


@pytest.fixture
def archive():
    arguments = case(base=-1e-6)
    arguments["historical_anchor"]["trace_response_index"] = 3543
    answer = probe_terminal_only_tail_repair(**arguments)
    return json.loads(json.dumps(_canonical(answer))), arguments["historical_anchor"]


def replay_fields(answer):
    rollouts = [deepcopy(row["rollout"]) for row in answer["evaluated_candidates"]]
    metrics = [
        dict(
            joint_feasible=row["audit"]["feasible"],
            safety_slack=deepcopy(row["safety_slack"]),
            normalized_squared_errors=deepcopy(row["normalized_squared_errors"]),
        )
        for row in answer["evaluated_candidates"]
    ]
    return rollouts, metrics


def test_complete_json_ledger_and_fresh_gates_handle_real_boolean_arrays(checker, archive):
    answer, history = archive
    flags = np.asarray(history["rollout"]["periods"][0]["clipped"])
    assert flags.dtype.kind == "b"
    observed = []

    def compare(name, actual, saved, tolerance=checker.NUMERIC_TOLERANCE):
        observed.append((name, np.asarray(actual).dtype.kind))
        return checker.semantic_compare(name, actual, saved, tolerance)

    checked = checker.audit_repair_ledger(answer, historical_anchor=history, compare=compare)
    fresh = checker.verify_replayed_repair(answer, *replay_fields(answer))
    assert checked["passed"] and fresh["passed"]
    assert checked["selected_response_index"] == fresh["selected_response_index"] == 1
    assert checked["witness_indices"] == []
    assert checked["control_qualified"] is fresh["control_qualified"] is False
    assert any(name.endswith("periods[0].clipped") and kind == "b" for name, kind in observed)
    assert not any("sufficient" in field or "prior_max_abs" in field for field in checked)


def test_changed_historical_boolean_cannot_be_hidden_by_float_tolerance(checker, archive):
    answer, history = archive
    history["rollout"]["periods"][0]["clipped"][0, 0] = True
    with pytest.raises(ValueError):
        checker.audit_repair_ledger(answer, historical_anchor=history)


@pytest.mark.parametrize("defect", ["ranking", "selected", "grid", "history_fields", "unreported_history", "empirical"])
def test_independent_reconstruction_rejects_forged_metadata(checker, archive, defect):
    answer, history = archive
    if defect == "ranking":
        answer["eligible_ranking"] = answer["eligible_ranking"][::-1]
    elif defect == "selected":
        answer["selected_response_index"] = 3
    elif defect == "grid":
        answer["grid"]["grid_commands"][0]["commands"][10][0] = 1e-6
    elif defect == "history_fields":
        answer["historical_anchor_comparison"]["compared_fields"].remove("rollout")
    elif defect == "unreported_history":
        answer["historical_anchor_comparison"]["ignored_historical_fields"] = []
    else:
        answer["empirical_error_bound_used"] = True
    with pytest.raises(ValueError):
        checker.audit_repair_ledger(answer, historical_anchor=history)


@pytest.mark.parametrize("field", ["safety", "pose", "joint", "causal", "repeat"])
def test_fresh_strict_classifications_do_not_inherit_saved_success(checker, archive, field):
    answer, _ = archive
    rollouts, metrics = replay_fields(answer)
    if field == "safety":
        metrics[1]["safety_slack"][0] = np.nextafter(1e-8, -np.inf)
    elif field == "pose":
        metrics[1]["normalized_squared_errors"][0][0] = np.nextafter(1.0, np.inf)
    elif field == "joint":
        metrics[1]["joint_feasible"] = False
    elif field == "causal":
        rollouts[1]["periods"][0]["clipped"][0][0] = True
    else:
        rollouts[-1]["periods"][-1]["clipped"][0][0] = True
    with pytest.raises(ValueError):
        checker.verify_replayed_repair(answer, rollouts, metrics)


def test_strict_selection_uses_actual_displacement_then_request_index(checker):
    arguments = case(base=-1e-6, terminal=lambda commands: -1e-6 if not np.any(commands) else 1e-4)
    answer = probe_terminal_only_tail_repair(**arguments)
    checked = checker.audit_repair_ledger(answer, historical_anchor=arguments["historical_anchor"])
    assert checked["selected_response_index"] == 3
    assert checked["eligible_ranking"][:6] == [3, 4, 7, 8, 11, 12]


def test_all_failed_repairs_finish_without_invented_selection(checker):
    arguments = case(base=-1e-6, terminal=lambda commands: -1e-6)
    answer = probe_terminal_only_tail_repair(**arguments)
    checked = checker.audit_repair_ledger(answer, historical_anchor=arguments["historical_anchor"])
    fresh = checker.verify_replayed_repair(answer, *replay_fields(answer))
    assert checked["selected_response_index"] is fresh["selected_response_index"] is None
    assert checked["improvement_found"] is fresh["improvement_found"] is False
