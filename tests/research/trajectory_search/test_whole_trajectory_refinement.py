"""Full-trajectory protection, fresh derivatives, and durable interruption evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.trajectory_search import whole_trajectory_refinement as core
from s2a.research.trajectory_search.cap_distance_objective import cap_loss
from scripts.tracking.trajectory_search.whole_trajectory_refinement_audit import audit_refinement

PREVIOUS = np.zeros(6)
LOWER, UPPER = -np.ones((12, 6)), np.ones((12, 6))


def response(command, *, bad_other=False):
    error = np.zeros((12, 6))
    error[0, 0] = 0.01 * (1.2 - command[0, 0])
    if bad_other:
        error[4, 0] = 0.01 * (0.5 + 4 * command[0, 0])
    q = np.column_stack((np.sum(error[:, :3] ** 2, axis=1) / 0.01**2, np.sum(error[:, 3:] ** 2, axis=1) / 0.05**2))
    safety = np.ones(4369)
    safety[0] = 0.1 - command[0, 0] ** 2
    rate = np.all(np.abs(np.diff(np.vstack((PREVIOUS, command)), axis=0)) <= 0.25 + 1e-12)
    return dict(
        commands=command.copy(),
        tcp_errors=error,
        normalized_squared_errors=q,
        safety_slack=safety,
        audit=dict(feasible=bool(rate and np.min(safety) >= -1e-10)),
    )


def run(*, budget=640, bad_other=False):
    entries, events = [], []

    def evaluate(command):
        item = response(command, bad_other=bad_other)
        entries.append(item)
        return item

    ledger = core.refine(
        evaluate, np.zeros((12, 6)), PREVIOUS, LOWER, UPPER, max_evaluations=budget, event_sink=events.append
    )
    return ledger, entries, events


def test_full_trajectory_refinement_improves_without_losing_original_passes():
    ledger, entries, events = run()
    assert audit_refinement(ledger, entries, PREVIOUS, LOWER, UPPER, events)["passed"]
    assert ledger["selected_index"] is not None
    baseline = entries[0]["normalized_squared_errors"]
    selected = entries[ledger["selected_index"]]["normalized_squared_errors"]
    assert cap_loss(selected.ravel()) < cap_loss(baseline.ravel())
    assert np.all(selected <= np.maximum(1.0, baseline))
    assert np.sum(selected <= 1) >= np.sum(baseline <= 1)
    assert np.all(entries[ledger["selected_index"]]["safety_slack"] >= 1e-8)


def test_cap_gradient_is_in_squared_error_units():
    q = np.linspace(0.4, 3.0, 24)
    g = core.loss_weights(q)
    for i in range(24):
        step = np.zeros(24)
        step[i] = 1e-6
        numerical = (cap_loss(q + step) - cap_loss(q - step)) / 2e-6
        assert g[i] == pytest.approx(numerical, abs=1e-9)
    assert core.loss_weights(np.ones(24)).max() == 0


def test_success_flag_cannot_admit_a_new_previously_passing_violation(monkeypatch):
    def fake_minimize(objective, start, *, constraints, **kwargs):
        trial = start.copy()
        trial[0] = 0.2
        objective(trial)
        constraints[1]["fun"](trial)
        return SimpleNamespace(success=True, status=0, message="claimed", nit=1, x=trial)

    monkeypatch.setattr(core, "minimize", fake_minimize)
    ledger, entries, events = run(bad_other=True)
    assert audit_refinement(ledger, entries, PREVIOUS, LOWER, UPPER, events)["passed"]
    assert ledger["solver_success"] and ledger["selected_index"] == 0
    assert entries[-1]["normalized_squared_errors"][4, 0] > 1


def test_budget_interrupt_keeps_incomplete_derivative_and_paid_prefix():
    ledger, entries, events = run(budget=12)
    assert ledger["status"] == "response_budget" and len(entries) == 12
    assert not ledger["derivatives"][0]["complete"]
    assert audit_refinement(ledger, entries, PREVIOUS, LOWER, UPPER, events)["passed"]


def test_interrupted_evaluation_keeps_started_but_not_completed_event(monkeypatch):
    def fake_minimize(objective, start, *, constraints, **kwargs):
        constraints[1]["jac"](start)

    monkeypatch.setattr(core, "minimize", fake_minimize)
    events = []
    ledger = {}
    calls = []

    def interrupted(command):
        calls.append(command)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return response(command)

    with pytest.raises(RuntimeError):
        core.refine(interrupted, np.zeros((12, 6)), PREVIOUS, LOWER, UPPER, ledger=ledger, event_sink=events.append)
    assert (
        len(ledger["requests"]) == 2 and ledger["requests"][0]["completed"] and not ledger["requests"][1]["completed"]
    )
    assert sum(e["kind"] == "request" for e in events) == 2
    assert sum(e["kind"] == "response_completed" for e in events) == 1
    assert not any(e["kind"] == "solver_closed" for e in events)


@pytest.mark.parametrize("fault", ["q_ceiling", "count", "derivative", "events", "selected"])
def test_independent_audit_rejects_changed_tracking_or_provenance(fault):
    ledger, entries, events = run(budget=74)
    ledger, events = deepcopy(ledger), deepcopy(events)
    if fault == "q_ceiling":
        ledger["q_ceiling"][0] += 1
    elif fault == "count":
        ledger["response_evaluations"] -= 1
    elif fault == "derivative":
        ledger["derivatives"][0]["probes"][0]["nominal_delta"] *= -1
    elif fault == "events":
        events = [e for e in events if e["kind"] != "response_completed"]
    else:
        ledger["selected_index"] = None
    with pytest.raises((ValueError, AssertionError)):
        audit_refinement(ledger, entries, PREVIOUS, LOWER, UPPER, events)


def test_common_warm_is_identical_and_only_roundoff_repair_allowed():
    warm = np.zeros((12, 6))
    warm[0, 0] = 0.2 + 1e-15
    rootlo, roothi = np.full((12, 6), -0.2), np.full((12, 6), 0.2)
    start = core.common_start(warm, PREVIOUS, -np.ones(6), np.ones(6), rootlo, roothi)
    assert start[0, 0] == 0.2 and warm[0, 0] > 0.2
    warm[0, 0] = 0.201
    with pytest.raises(ValueError):
        core.common_start(warm, PREVIOUS, -np.ones(6), np.ones(6), rootlo, roothi)
