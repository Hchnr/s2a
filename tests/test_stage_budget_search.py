"""Only the explicit stage ceiling changes; responses and qualification do not."""

import ast
import inspect
from copy import deepcopy

import numpy as np
import pytest
from test_interior_trajectory_search import arguments, entry
from test_rejected_response_search import failed_case, fake_qp

import s2a.global_envelope_search as baseline
import s2a.stage_budget_search as planning
from s2a.directional_derivative_probe import _canonical


def test_search_state_machine_and_all_shared_policies_are_the_frozen_ones():
    expected = ast.parse(inspect.getsource(baseline.search))
    observed = ast.parse(inspect.getsource(planning.search))
    assert ast.dump(observed) == ast.dump(expected)
    assert planning.POSE_CEILING_POLICY == baseline.POSE_CEILING_POLICY
    assert planning.initial_global_envelope is baseline.initial_global_envelope
    assert planning.FEEDBACK_POLICY == baseline.FEEDBACK_POLICY
    assert planning.SearchEvaluationError is baseline.SearchEvaluationError
    assert planning.causal_secant_update is baseline.causal_secant_update
    assert planning.choose_rejected_source is baseline.choose_rejected_source


@pytest.mark.parametrize("stages", [1, 24, 25, 60])
def test_explicit_budget_up_to_sixty_is_preserved_without_mutating_the_config(stages):
    config = arguments(max_stages=stages)["config"]
    saved = deepcopy(config)
    checked = planning.validate_config(config)
    assert checked == saved and checked["max_stages"] == stages and config == saved
    checked["alphas"][0] = 0.25
    assert config["alphas"][0] == 1.0


@pytest.mark.parametrize("stages", [True, False, 0, -1, 61, 60.0, np.int64(60)])
def test_boolean_noninteger_and_out_of_range_stages_are_rejected_before_any_response(stages):
    inputs = arguments(max_stages=stages)
    calls = []
    inputs["evaluate"] = lambda command: calls.append(command)
    with pytest.raises(ValueError, match="integer stage budget"):
        planning.search(**inputs)
    assert not calls


@pytest.mark.parametrize("defect", ["response_budget", "seed", "alpha", "missing", "radius"])
def test_extended_stages_do_not_bypass_the_original_configuration_validation(defect):
    config = arguments(max_stages=60)["config"]
    if defect == "response_budget":
        config["max_responses"] = 123
    elif defect == "seed":
        config["seed"] = 43
    elif defect == "alpha":
        config["alphas"][-1] = 0.03125
    elif defect == "missing":
        del config["safety_buffer"]
    else:
        config["radius_min"] = config["radius_max"] * 2
    with pytest.raises(ValueError):
        planning.validate_config(config)


def test_twenty_four_stage_config_has_exactly_the_original_complete_output(monkeypatch):
    fake_qp(monkeypatch, [0.001] * 24, module=baseline)
    expected = baseline.search(**arguments(max_stages=24))
    fake_qp(monkeypatch, [0.001] * 24, module=planning)
    observed = planning.search(**arguments(max_stages=24))
    assert _canonical(observed) == _canonical(expected)
    assert observed["total_stages"] == 24 and observed["response_evaluations"] == 26


def test_stages_after_twenty_four_really_run_and_every_response_is_charged(monkeypatch):
    calls = fake_qp(monkeypatch, [0.001] * 26, module=planning)
    answer = planning.search(**arguments(max_stages=26))
    assert answer["config"]["max_stages"] == 26
    assert len(calls) == answer["total_stages"] == 26 and answer["total_qp_iterations"] == 78
    assert answer["termination_reason"] == "stage_budget_exhausted"
    assert answer["final_anchor_index"] == 26 and answer["response_evaluations"] == answer["prediction_calls"] == 28
    assert answer["unique_command_count"] == 27
    assert [row["stage_index"] for row in answer["request_ledger"][1:-1]] == list(range(26))
    assert answer["request_ledger"][25]["stage_index"] == 24
    assert answer["request_ledger"][26]["stage_index"] == 25
    assert all(row["charged"] and row["completed"] for row in answer["request_ledger"])
    assert answer["request_ledger"][-1]["role"] == "initial_repeat" and answer["repeatability"]["exact"]
    assert answer["selected"] is None and answer["qualified_witness_indices"] == []


def test_larger_stage_budget_keeps_the_response_limit_and_reserves_the_repeat(monkeypatch):
    calls = fake_qp(monkeypatch, [0.001], module=planning)
    answer = planning.search(**arguments(max_stages=60, max_responses=3))
    assert len(calls) == 1 and answer["termination_reason"] == "response_budget_exhausted"
    assert answer["config"]["max_stages"] == 60 and answer["config"]["max_responses"] == 3
    assert [row["role"] for row in answer["request_ledger"]] == ["initial", "trial", "initial_repeat"]
    assert answer["prediction_calls"] == answer["response_evaluations"] == 3
    assert answer["unique_command_count"] == 2


def test_the_original_radius_stop_still_terminates_without_restarting_to_consume_sixty_stages(monkeypatch):
    calls = fake_qp(monkeypatch, [0] * 6, module=planning, qualified=False)
    answer = planning.search(**arguments(max_stages=60))
    assert len(calls) == 6 and answer["total_stages"] == 6
    assert answer["termination_reason"] == "radius_below_minimum"
    assert answer["response_evaluations"] == 2 and answer["unique_command_count"] == 1
    assert answer["final_anchor_index"] == 0 and answer["selected"] is None


def test_unsafe_progress_still_cannot_move_the_anchor_or_supply_feedback(monkeypatch):
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**failed_case(defect="unsafe"))
    assert answer["final_anchor_index"] == 0 and answer["selected"] is None
    assert answer["rejected_update_count"] == 0
    assert all(attempt["acceptance_reason"] == "actual_admission_failed" for attempt in answer["stages"][0]["attempts"])


def test_original_full_witness_stops_early_even_when_sixty_stages_are_available(monkeypatch):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.00005 - 0.02 * command[0, 0]
        return entry(command, q=q)

    inputs = arguments(response, max_stages=60)
    inputs["Jq"][18, 0] = -0.02
    calls = fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**inputs)
    assert len(calls) == 1 and answer["termination_reason"] == "qualified_witness"
    assert answer["selected_response_index"] == 1 and answer["qualified_witness_indices"] == [1]
    assert answer["response_evaluations"] == 3 and np.all(answer["selected"]["normalized_squared_errors"] <= 1)
    assert answer["selected"]["audit"]["feasible"] and not answer["control_qualified"]


def test_exact_final_repeat_is_still_required_for_witness_qualification(monkeypatch):
    calls = []

    def response(command):
        calls.append(None)
        return entry(command, trace=int(len(calls) > 2))

    inputs = arguments(response, max_stages=60)
    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("Initial witness requires no QP"))
    answer = planning.search(**inputs)
    assert answer["initial_comparison"]["exact"] and not answer["repeatability"]["exact"]
    assert answer["witness_indices"] == [0, 1] and answer["qualified_witness_indices"] == []
    assert answer["selected"] is None and answer["termination_reason"] == "repeat_mismatch"


def test_shared_callback_exception_preserves_the_actual_charged_prefix(monkeypatch):
    inputs = arguments(max_stages=60)
    original = inputs["evaluate"]

    def interrupted(command):
        if np.any(command):
            raise RuntimeError("Synthetic interrupted response")
        return original(command)

    inputs["evaluate"] = interrupted
    fake_qp(monkeypatch, [0.001], module=planning)
    with pytest.raises(baseline.SearchEvaluationError) as caught:
        planning.search(**inputs)
    answer = caught.value.answer
    assert answer["prediction_calls"] == 2 and answer["response_evaluations"] == 1
    assert answer["request_ledger"][-1]["charged"] and not answer["request_ledger"][-1]["completed"]
    assert isinstance(caught.value.__cause__, RuntimeError)
