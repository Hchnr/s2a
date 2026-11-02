"""Observe strict fresh-decision failures without changing the comparison gates.

These classified-response unit fixtures exercise no QP or native response engine.
The full response/repeat reconstruction remains the native checker's separate gate.
"""

import json
from copy import deepcopy

import numpy as np
import pytest

import scripts.tracking.trajectory_search.check_tracking_v5_state_vector_gn_search as checker
from s2a.research.trajectory_search.interior_trajectory_search import DEFAULT_CONFIG


def _assert_unchanged(actual, saved):
    if isinstance(saved, np.ndarray):
        np.testing.assert_array_equal(actual, saved, strict=True)
    elif isinstance(saved, dict):
        assert actual.keys() == saved.keys()
        for key in saved:
            _assert_unchanged(actual[key], saved[key])
    elif isinstance(saved, (tuple, list)):
        assert type(actual) is type(saved) and len(actual) == len(saved)
        for value, before in zip(actual, saved, strict=True):
            _assert_unchanged(value, before)
    else:
        assert type(actual) is type(saved) and actual == saved


def _verify(args, diagnostics=None, *, error=None):
    before = deepcopy(args)
    try:
        if error is None:
            return checker.verify_fresh_decisions(*args, diagnostics=diagnostics)
        with pytest.raises(ValueError, match=error) as raised:
            checker.verify_fresh_decisions(*args, diagnostics=diagnostics)
        return str(raised.value)
    finally:
        _assert_unchanged(args, before)


def _entry(commands, q, safety):
    return dict(
        commands=commands.copy(),
        normalized_squared_errors=np.asarray(q).reshape(12, 2).copy(),
        safety_slack=safety.copy(),
        audit=dict(feasible=True),
        feasible=bool(np.all(np.asarray(q) <= 1)),
    )


def _merit(entry, mode, config):
    if mode == "pose":
        return float(0.5 * np.sum(entry["normalized_squared_errors"]))
    weighted = entry["safety_slack"] - config["safety_buffer"]
    weighted = weighted.copy()
    weighted[-1] *= config["terminal_time_scale"]
    return max(0.0, 1.0 - float(np.min(weighted) / config["reserve_target"]))


def _stage(entries, base, trial, mode, predicted, radius_before, radius_after, config, *, accepted=True):
    actual = _merit(entries[base], mode, config) - _merit(entries[trial], mode, config)
    return dict(
        stage_index=base,
        anchor_before_index=base,
        mode=mode,
        radius_before=radius_before,
        radius_after=radius_after,
        attempts=[
            dict(
                alpha_index=0,
                alpha=1.0,
                response_index=trial,
                commands=entries[trial]["commands"].copy(),
                predicted_reduction=predicted,
                actual_reduction=actual,
                ratio=actual / predicted,
                accepted=accepted,
                acceptance_reason="accepted_reduction" if accepted else "ratio_insufficient",
            )
        ],
    )


def _inputs(monkeypatch, entries, stages, config):
    # Isolate decision diagnostics; still require the fixture's repeat metrics exact.
    def compare_repeat(initial, repeat, index):
        assert index == len(entries) - 1
        _assert_unchanged(initial["metrics"], repeat["metrics"])
        return dict(exact=True)

    monkeypatch.setattr(checker, "_comparison_report", compare_repeat)
    metrics = [
        dict(
            normalized_squared_errors=entry["normalized_squared_errors"].copy(),
            safety_slack=entry["safety_slack"].copy(),
            joint_feasible=entry["audit"]["feasible"],
            feasible=entry["feasible"],
        )
        for entry in entries
    ]
    answer = dict(
        evaluated_candidates=entries,
        pose_ceiling=np.full(24, max(1.0, float(np.max(entries[0]["normalized_squared_errors"]))) * 1.01),
        stages=stages,
        initial_comparison=dict(exact=True),
        repeatability=dict(exact=True),
        qualified_witness_indices=[],
    )
    return answer, [{} for _ in entries], metrics, config


def _pose_inputs(monkeypatch, *, accepted=True):
    config = deepcopy(DEFAULT_CONFIG)
    base = np.full(24, 0.2)
    base[0] = 4.0
    trial = base.copy()
    trial[0] = 3.9
    commands = np.zeros((12, 6))
    step = commands.copy()
    step[0, 0] = 0.005
    safety = np.full(4369, 0.01)
    entries = [_entry(commands, base, safety), _entry(step, trial, safety), _entry(commands, base, safety)]
    stage = _stage(
        entries,
        0,
        1,
        "pose",
        1e-4 if accepted else 1.0,
        0.005,
        0.01 if accepted else 0.0025,
        config,
        accepted=accepted,
    )
    return _inputs(monkeypatch, entries, [stage], config)


def _assert_ratio_failure(event):
    assert event["event"] == "attempt"
    assert event["numeric_check"] == "fresh_ratio"
    assert event["fresh_actual_reduction_passed"] is True
    assert event["fresh_ratio_passed"] is False
    assert event["numeric_check_passed"] is False
    assert 0 < abs(event["actual_reduction_difference"]) < checker.NUMERIC_TOLERANCE
    assert abs(event["ratio_difference"]) > checker.NUMERIC_TOLERANCE
    assert event["predicted_reduction"] == 1e-4
    assert event["fresh_gate_inputs"]["admitted"] is True
    assert event["fresh_gate_inputs"]["witness"] is False
    assert json.loads(json.dumps(event, allow_nan=False)) == event


def test_small_predicted_reduction_exposes_fresh_ratio_without_relaxing_gate(monkeypatch):
    args = _pose_inputs(monkeypatch)
    assert _verify(args)["passed"] is True
    saved_q = args[0]["evaluated_candidates"][1]["normalized_squared_errors"]
    args[2][1]["normalized_squared_errors"][0, 0] += 2.0**-40
    assert not np.array_equal(saved_q, args[2][1]["normalized_squared_errors"])
    without_diagnostics = _verify(args, error="fresh_ratio")
    events = []
    assert _verify(args, events, error="fresh_ratio") == without_diagnostics
    assert [event["event"] for event in events] == ["stage", "attempt"]
    assert events[0]["numeric_check_passed"] is True
    last = events[-1]
    _assert_ratio_failure(last)
    assert (last["stage_index"], last["anchor_before_index"], last["response_index"], last["alpha_index"]) == (
        0,
        0,
        1,
        0,
    )
    assert last["alpha"] == 1.0 and last["merit_field"] == "state_loss"
    assert last["fresh_base_merit"] == last["saved_base_merit"]
    assert last["fresh_actual_merit"] > last["saved_actual_merit"]
    assert last["actual_reduction_difference"] < 0 and last["ratio_difference"] < 0
    assert last["thresholds"]["minimum_actual_reduction"] == args[3]["minimum_actual_reduction"]


def test_diagnostics_preserve_legitimate_acceptance_and_ratio_rejection(monkeypatch):
    for accepted in (True, False):
        args = _pose_inputs(monkeypatch, accepted=accepted)
        without_diagnostics = _verify(args)
        events = []
        assert _verify(args, events) == without_diagnostics
        assert without_diagnostics["passed"] is True
        assert without_diagnostics["control_qualified"] is False
        assert without_diagnostics["qualified_witness_indices"] == []
        attempt = args[0]["stages"][0]["attempts"][0]
        assert attempt["accepted"] is accepted
        assert attempt["acceptance_reason"] == ("accepted_reduction" if accepted else "ratio_insufficient")
        last = events[-1]
        assert last["numeric_check"] == "acceptance_priority" and last["numeric_check_passed"] is True
        assert last["fresh_actual_reduction_passed"] is True and last["fresh_ratio_passed"] is True
        assert last["fresh_accepted"] is last["saved_accepted"] is accepted
        assert last["fresh_acceptance_reason"] == last["saved_acceptance_reason"] == attempt["acceptance_reason"]
        assert last["saved_radius_after"] == args[0]["stages"][0]["radius_after"]
        assert last["numeric_tolerance"] == checker.NUMERIC_TOLERANCE == 1e-10
        assert events[0]["radius_check_passed"] is True
        assert events[0]["fresh_radius_after"] == events[0]["saved_radius_after"]
        assert last["actual_reduction_difference"] == last["ratio_difference"] == 0
        assert last["fresh_gate_inputs"] == last["saved_gate_inputs"]
        assert last["fresh_gate_inputs"]["admitted"] is True
        assert (last["boundary_distances"]["ratio_minus_accept"] >= 0) is accepted
        assert json.loads(json.dumps(events, allow_nan=False)) == events


def test_restore_diagnostics_track_weighted_minimum_row_and_changing_anchor(monkeypatch):
    config = deepcopy(DEFAULT_CONFIG)
    buffer, target, weight = (config[key] for key in ("safety_buffer", "reserve_target", "terminal_time_scale"))
    q = np.full(24, 0.2)
    q[0] = 4.0
    entries = []
    for displacement, ordinary_reserve, terminal_reserve in ((0.0, 0.2, 0.25), (0.005, 0.3, 0.25), (0.015, 0.45, 0.4)):
        commands = np.zeros((12, 6))
        commands[0, 0] = displacement
        safety = np.full(4369, 0.01)
        safety[11] = buffer + ordinary_reserve * target
        safety[-1] = buffer + terminal_reserve * target / weight
        entries.append(_entry(commands, q, safety))
    entries.append(deepcopy(entries[0]))
    stages = [
        _stage(entries, 0, 1, "restore", 1e-4, 0.005, 0.01, config),
        _stage(entries, 1, 2, "restore", 1e-4, 0.01, 0.02, config),
    ]
    args = _inputs(monkeypatch, entries, stages, config)
    events = []
    assert _verify(args, events)["passed"] is True
    assert [(row["event"], row["stage_index"]) for row in events] == [
        ("stage", 0),
        ("attempt", 0),
        ("stage", 1),
        ("attempt", 1),
    ]
    first, second = events[1], events[3]
    assert first["fresh_base_reserve_row"]["flat_index"] == 11
    assert first["fresh_base_reserve_row"]["block"] == "inner_lower"
    assert first["fresh_base_reserve_row"]["local_index"] == [1, 4]
    assert first["fresh_actual_reserve_row"]["flat_index"] == 4368
    assert second["anchor_before_index"] == 1 and second["response_index"] == 2
    assert second["fresh_base_reserve_row"] == first["fresh_actual_reserve_row"]
    terminal = second["fresh_actual_reserve_row"]
    assert terminal["block"] == "terminal" and terminal["local_index"] == [0]
    assert terminal["weight"] == weight == 0.1
    assert terminal["raw_slack"] > entries[2]["safety_slack"][11]
    assert terminal["weighted_slack"] == weight * terminal["raw_slack"]
    assert terminal["weighted_buffered_slack"] == weight * (terminal["raw_slack"] - buffer)
    assert terminal["reserve"] == pytest.approx(0.4)
    assert second["merit_field"] == "violation"
    assert second["fresh_base_merit"] == pytest.approx(0.75)
    assert second["fresh_actual_merit"] == pytest.approx(0.6)
    args[2][2]["safety_slack"][-1] += 2.0**-55
    without_diagnostics = _verify(args, error="fresh_ratio")
    perturbed = []
    assert _verify(args, perturbed, error="fresh_ratio") == without_diagnostics
    last = perturbed[-1]
    _assert_ratio_failure(last)
    assert (last["stage_index"], last["anchor_before_index"], last["response_index"]) == (1, 1, 2)
    assert perturbed[1]["fresh_ratio_passed"] is True
    assert last["fresh_actual_reserve_row"]["flat_index"] == last["saved_actual_reserve_row"]["flat_index"] == 4368
    assert last["fresh_actual_reserve_row"]["reserve"] > last["saved_actual_reserve_row"]["reserve"]
    assert last["actual_reduction_difference"] > 0 and last["ratio_difference"] > 0
    assert last["fresh_actual_metrics"]["state_loss"] == last["saved_actual_metrics"]["state_loss"]


def test_mode_mismatch_retains_stage_context_before_any_numeric_attempt(monkeypatch):
    args = _pose_inputs(monkeypatch)
    config = args[3]
    args[2][0]["safety_slack"][-1] = (
        config["safety_buffer"] + 0.2 * config["reserve_target"] / config["terminal_time_scale"]
    )
    error = "Independent actual reserve changes the stage mode"
    without_diagnostics = _verify(args, error=error)
    events = []
    assert _verify(args, events, error=error) == without_diagnostics
    assert len(events) == 1
    stage = events[0]
    assert stage["event"] == "stage" and stage["stage_index"] == stage["anchor_before_index"] == 0
    assert stage["numeric_check"] == "stage_mode" and stage["numeric_check_passed"] is False
    assert stage["fresh_mode"] == "restore"
    assert stage["saved_mode"] == stage["saved_mode_from_metrics"] == "pose"
    assert stage["fresh_base_reserve_row"]["flat_index"] == 4368
    assert stage["fresh_base_metrics"]["reserve"] == pytest.approx(0.2)
    assert stage["boundary_distances"]["fresh_reserve_minus_entry"] < 0
    assert stage["boundary_distances"]["saved_reserve_minus_entry"] > 0
    assert "response_index" not in stage and "fresh_ratio_passed" not in stage
    assert json.loads(json.dumps(stage, allow_nan=False)) == stage
