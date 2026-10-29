"""A fixed global numerical envelope never replaces original execution caps."""

from copy import deepcopy

import numpy as np
import pytest
from test_interior_trajectory_search import arguments, entry
from test_rejected_response_search import failed_case, fake_qp

import s2a.global_envelope_search as planning
import s2a.rejected_response_search as baseline
from s2a.directional_derivative_probe import _canonical


@pytest.mark.parametrize("maximum", [0.0, 0.5, 1.0, 4.0])
def test_initial_global_envelope_is_one_constant_bound_to_the_initial_errors(maximum):
    q = np.linspace(0, maximum, 24)
    result = planning.initial_global_envelope(q, {"pose_ceiling_fraction": 0.01})
    np.testing.assert_array_equal(result, np.full(24, max(1, maximum) * 1.01))
    q[:] = 1000
    assert np.all(result == max(1, maximum) * 1.01)


@pytest.mark.parametrize("defect", ["nan", "infinite", "negative", "matrix", "short"])
def test_envelope_rejects_nonfinite_negative_or_incomplete_squared_errors(defect):
    q = np.ones(24)
    if defect == "nan":
        q[0] = np.nan
    elif defect == "infinite":
        q[-1] = np.inf
    elif defect == "negative":
        q[0] = -np.nextafter(0.0, 1.0)
    elif defect == "matrix":
        q = q.reshape(12, 2)
    else:
        q = q[:-1]
    with pytest.raises(ValueError):
        planning.initial_global_envelope(q, {"pose_ceiling_fraction": 0.01})


def test_uniform_initial_errors_make_the_entire_search_equivalent_to_the_frozen_variant(monkeypatch):
    def response(command):
        return entry(command, q=np.full((12, 2), 4.0 - 10 * command[0, 0]))

    inputs = arguments(response, max_stages=1)
    inputs["Jq"][:, 0] = -10
    fake_qp(monkeypatch, [0.005], module=baseline)
    expected = baseline.search(**deepcopy(inputs))
    fake_qp(monkeypatch, [0.005], module=planning)
    observed = planning.search(**deepcopy(inputs))
    assert observed.pop("pose_ceiling_policy") == "fixed_initial_global_maximum"
    assert _canonical(observed) == _canonical(expected)


def test_safe_progress_inside_global_envelope_can_be_an_anchor_without_being_an_execution_solution(monkeypatch):
    fake_qp(monkeypatch, [0.005], module=baseline)
    old = baseline.search(**failed_case())
    fake_qp(monkeypatch, [0.005], module=planning)
    new = planning.search(**failed_case())
    assert old["final_anchor_index"] == 0 and old["rejected_update_count"] == 1
    assert new["final_anchor_index"] == 1 and new["rejected_update_count"] == 0
    observed = new["evaluated_candidates"][1]
    assert observed["normalized_squared_errors"][0, 0] > old["pose_ceiling"][0]
    assert np.all(observed["normalized_squared_errors"].ravel() <= new["pose_ceiling"])
    assert new["stages"][0]["attempts"][0]["acceptance_reason"] == "accepted_reduction"
    assert new["stages"][0]["secant_update"] is not None
    assert new["selected"] is None and new["qualified_witness_indices"] == []
    assert observed["prefix_length"] == 0 and not new["control_qualified"]


def test_all_later_qps_keep_the_original_envelope_after_the_anchor_improves(monkeypatch):
    calls = fake_qp(monkeypatch, [0.005, 0.0025], module=planning)
    inputs = failed_case(max_stages=2)
    answer = planning.search(**inputs)
    initial_q = inputs["historical_initial"]["normalized_squared_errors"].ravel()
    expected = np.full(24, np.max(initial_q) * 1.01)
    np.testing.assert_array_equal(answer["pose_ceiling"], expected)
    assert [stage["anchor_before_index"] for stage in answer["stages"]] == [0, 1]
    first_anchor_q = answer["evaluated_candidates"][1]["normalized_squared_errors"].ravel()
    np.testing.assert_array_equal(calls[1]["b"][4369:4393], expected - first_anchor_q)
    assert not np.array_equal(expected, planning.initial_global_envelope(first_anchor_q, inputs["config"]))
    assert answer["final_anchor_index"] == 2 and answer["selected"] is None


@pytest.mark.parametrize("defect", ["unsafe", "joint", "ratio"])
def test_larger_numerical_envelope_preserves_safety_audit_and_actual_improvement_gates(monkeypatch, defect):
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**failed_case(defect=defect))
    stage = answer["stages"][0]
    assert stage["accepted_response_index"] is None and stage["anchor_after_index"] == 0
    assert answer["rejected_update_count"] == 0 and answer["selected"] is None
    assert all(not attempt["accepted"] for attempt in stage["attempts"])
    assert stage["radius_after"] == stage["radius_before"] / 2


def test_feedback_still_requires_a_failure_of_the_fixed_global_envelope(monkeypatch):
    def response(command):
        x = command[0, 0]
        q = np.full((12, 2), 0.25)
        q[0, 0], q[9, 0] = (2.0 if x else 0.999), 1.5 - 10 * x
        return entry(command, q=q, reserve=2 + 10 * x)

    inputs = arguments(response, max_stages=1)
    inputs["Jq"][0, 0] = 0.2
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**inputs)
    update = answer["stages"][0]["rejected_update"]
    assert update["alpha"] == 0.0625 and update["source_response_index"] == 5
    assert answer["rejected_update_count"] == 1 and answer["final_anchor_index"] == 0
    assert answer["selected"] is None and answer["prediction_calls"] == 7


def test_response_budget_keeps_the_charged_repeat_with_the_new_envelope(monkeypatch):
    calls = fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**failed_case(max_stages=2, max_responses=3))
    assert len(calls) == 1 and answer["termination_reason"] == "response_budget_exhausted"
    assert [row["role"] for row in answer["request_ledger"]] == ["initial", "trial", "initial_repeat"]
    assert answer["prediction_calls"] == answer["response_evaluations"] == 3
    assert answer["unique_command_count"] == 2 and answer["repeatability"]["exact"]


def test_original_complete_caps_still_determine_a_selected_witness(monkeypatch):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.00005 - 0.02 * command[0, 0]
        return entry(command, q=q)

    inputs = arguments(response, max_stages=1)
    inputs["Jq"][18, 0] = -0.02
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**inputs)
    assert answer["selected_response_index"] == 1 and answer["qualified_witness_indices"] == [1]
    assert np.all(answer["selected"]["normalized_squared_errors"] <= 1)
    assert answer["selected"]["audit"]["feasible"] and np.all(answer["selected"]["safety_slack"] >= 1e-8)
    assert answer["initial_comparison"]["exact"] and answer["repeatability"]["exact"]
    assert not answer["control_qualified"]


def test_final_exact_repeat_remains_necessary_for_witness_qualification(monkeypatch):
    calls = []

    def response(command):
        calls.append(None)
        return entry(command, trace=int(len(calls) > 2))

    inputs = arguments(response)
    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("Initial witness does not solve"))
    answer = planning.search(**inputs)
    assert answer["initial_comparison"]["exact"] and not answer["repeatability"]["exact"]
    assert answer["witness_indices"] == [0, 1] and answer["qualified_witness_indices"] == []
    assert answer["selected"] is None and answer["termination_reason"] == "repeat_mismatch"


def test_callback_failures_share_the_original_exception_and_preserve_billing(monkeypatch):
    assert planning.SearchEvaluationError is baseline.SearchEvaluationError
    inputs = failed_case()
    original = inputs["evaluate"]

    def interrupted(command):
        if np.any(command):
            raise RuntimeError("Synthetic callback interruption")
        return original(command)

    inputs["evaluate"] = interrupted
    fake_qp(monkeypatch, [0.005], module=planning)
    with pytest.raises(baseline.SearchEvaluationError) as caught:
        planning.search(**inputs)
    answer = caught.value.answer
    assert answer["prediction_calls"] == 2 and answer["response_evaluations"] == 1
    assert answer["request_ledger"][-1]["charged"] and not answer["request_ledger"][-1]["completed"]
