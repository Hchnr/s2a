"""Actual curved safety, finite cache budgets and strict full-trajectory eligibility."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.trajectory_search import nonlinear_feasibility_projection as core
from scripts.tracking.trajectory_search.nonlinear_projection_audit import audit_projection

PREVIOUS, LO, HI = np.zeros(6), -np.ones(6), np.ones(6)


def evaluator(calls, curved=False, pose_fail=False):
    def evaluate(command):
        calls.append(command.copy())
        safety = np.ones(4369)
        safety[0] = 0.04 - command[0, 0] ** 2 if curved else 0.1 - command[0, 0]
        rate = np.all(np.abs(np.diff(np.vstack((PREVIOUS, command)), axis=0)) <= 0.25 + 1e-12)
        return dict(
            commands=command.copy(),
            safety_slack=safety,
            audit=dict(feasible=bool(rate and np.min(safety) >= -1e-10)),
            normalized_squared_errors=np.full((12, 2), 2.0 if pose_fail else 0.0),
        )

    return evaluate


@pytest.mark.parametrize("curved", [False, True])
def test_fresh_derivatives_restore_actual_safety_with_original_buffer(curved):
    calls = []
    entries = []
    evaluate = evaluator(calls, curved)

    def capture(command):
        e = evaluate(command)
        entries.append(e)
        return e

    warm = np.zeros((12, 6))
    warm[0, 0] = 0.24
    answer = core.project(capture, warm, PREVIOUS, LO, HI)
    assert audit_projection(answer, entries, PREVIOUS, LO, HI)["passed"]
    assert answer["response_evaluations"] == len(calls) <= 320
    if curved:
        assert len(answer["derivatives"]) > 1
        assert len({r["jacobian_sha256"] for r in answer["derivatives"] if r["complete"]}) > 1
        # Four tangent/Newton steps approach the curved boundary from outside.
        # Positive raw slack still does not meet the original buffered gate.
        best = max(e["safety_slack"][0] for e in entries)
        assert 0 < best < 1e-8
        assert not answer["safe_indices"] and answer["selected_index"] is None
    else:
        assert answer["safe_indices"]
        selected = entries[answer["selected_index"]]
        assert min(selected["safety_slack"]) >= 1e-8


def test_partial_derivative_budget_is_charged_and_auditable():
    entries = []
    calls = []
    evaluate = evaluator(calls)

    def capture(command):
        e = evaluate(command)
        entries.append(e)
        return e

    warm = np.zeros((12, 6))
    warm[0, 0] = 0.24
    a = core.project(capture, warm, PREVIOUS, LO, HI, max_evaluations=12)
    assert a["status"] == "response_budget" and len(entries) == 12 and not a["derivatives"][0]["complete"]
    assert audit_projection(a, entries, PREVIOUS, LO, HI)["passed"]


def test_solver_success_does_not_grant_safety_or_pose_qualification(monkeypatch):
    monkeypatch.setattr(
        core,
        "minimize",
        lambda fun, x, **kwargs: SimpleNamespace(success=True, status=0, message="claimed", nit=1, x=x),
    )
    calls = []
    warm = np.zeros((12, 6))
    warm[0, 0] = 0.24
    a = core.project(evaluator(calls), warm, PREVIOUS, LO, HI)
    assert a["solver_success"] and not a["safe_indices"] and a["selected_index"] is None
    b = core.project(evaluator([], pose_fail=True), np.zeros((12, 6)), PREVIOUS, LO, HI)
    assert b["safe_indices"] and not b["witness_indices"]


@pytest.mark.parametrize("fault", ["count", "derivative", "selected", "buffer", "domain", "rate"])
def test_independent_audit_rejects_ledger_corruption(fault):
    entries = []
    evaluate = evaluator([])

    def capture(command):
        e = evaluate(command)
        entries.append(e)
        return e

    warm = np.zeros((12, 6))
    warm[0, 0] = 0.24
    a = deepcopy(core.project(capture, warm, PREVIOUS, LO, HI))
    if fault == "count":
        a["response_evaluations"] -= 1
    elif fault == "derivative":
        a["derivatives"][0]["probes"][0]["nominal_delta"] = -1e-4
    elif fault == "selected":
        a["selected_index"] = 0
    elif fault == "buffer":
        a["safe_indices"].append(0)
    elif fault == "domain":
        a["upper"][1] = 0.9
    else:
        a["requests"][0]["rate_admitted"] = False
    with pytest.raises((ValueError, AssertionError)):
        audit_projection(a, entries, PREVIOUS, LO, HI)


def test_warm_repair_is_only_roundoff_and_larger_changes_rejected():
    warm = np.zeros((12, 6))
    warm[0, 0] = 0.25 + 1e-15
    a = core.project(evaluator([]), warm, PREVIOUS, LO, HI, max_evaluations=1)
    assert 0 < np.max(np.abs(a["warm_roundoff"])) < 1e-12
    warm[0, 0] = 0.25 + 1e-6
    with pytest.raises(ValueError):
        core.project(evaluator([]), warm, PREVIOUS, LO, HI)
