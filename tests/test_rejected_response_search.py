"""Charged rejection feedback changes the model, never acceptance or the anchor."""

from copy import deepcopy

import numpy as np
import pytest
from test_interior_trajectory_search import arguments, entry

import s2a.interior_trajectory_search as baseline
import s2a.rejected_response_search as planning
from s2a.directional_derivative_probe import _canonical


def fake_qp(monkeypatch, steps, *, module=planning, qualified=True):
    calls, steps = [], iter(steps)

    def solve(B, g, A, b, initial_t, **kwargs):
        calls.append(dict(B=B.copy(), g=g.copy(), A=A.copy(), b=b.copy(), initial_t=initial_t, kwargs=kwargs))
        point = np.zeros(73)
        point[0] = next(steps)
        return dict(qualified=qualified, point=point, solver_nit=3)

    monkeypatch.setattr(module, "solve_epigraph_qp", solve)
    return calls


def failed_case(*, defect=None, max_stages=1, max_responses=122):
    def response(command):
        x = command[0, 0]
        q = np.full((12, 2), 0.25)
        q[0, 0] = 1.02 if x != 0 and defect != "ratio" else 0.999
        q[9, 0] = 1.5 - (0 if defect == "ratio" else 10) * x
        reserve = -0.1 if x != 0 and defect == "unsafe" else 2.0 + 10 * x
        return entry(command, q=q, reserve=reserve, joint=not (defect == "joint" and x != 0))

    inputs = arguments(response, max_stages=max_stages, max_responses=max_responses)
    inputs["Jq"][0, 0], inputs["Js"][0, 0] = 0.2, 1e-5
    return inputs


def strip_feedback(answer):
    answer = deepcopy(answer)
    del answer["feedback_policy"], answer["rejected_update_count"]
    for stage in answer["stages"]:
        del stage["rejected_update"]
    return answer


def test_first_stage_is_identical_to_baseline_until_the_rejected_update(monkeypatch):
    fake_qp(monkeypatch, [0.005], module=baseline)
    expected = baseline.search(**failed_case())
    fake_qp(monkeypatch, [0.005])
    observed = planning.search(**failed_case())
    assert _canonical(strip_feedback(observed)) == _canonical(expected)
    assert observed["feedback_policy"] == planning.FEEDBACK_POLICY
    assert observed["rejected_update_count"] == 1


def test_minimum_alpha_feedback_is_charged_safe_and_never_becomes_an_anchor(monkeypatch):
    calls = fake_qp(monkeypatch, [0.005])
    answer = planning.search(**failed_case())
    stage = answer["stages"][0]
    update = stage["rejected_update"]
    assert update["source_response_index"] == 5 and update["alpha"] == 0.0625
    source = answer["evaluated_candidates"][5]
    np.testing.assert_array_equal(update["step"], source["commands"] - answer["evaluated_candidates"][0]["commands"])
    assert source["audit"]["feasible"] and np.all(source["safety_slack"] >= 1e-8)
    assert np.any(source["normalized_squared_errors"].ravel() > answer["pose_ceiling"])
    assert stage["anchor_before_index"] == stage["anchor_after_index"] == answer["final_anchor_index"] == 0
    assert stage["anchor_metrics_before"] == stage["anchor_metrics_after"]
    assert stage["radius_after"] == stage["radius_before"] / 2
    assert stage["secant_update"] is None and answer["selected"] is None
    assert answer["qualified_witness_indices"] == answer["witness_indices"] == []
    assert len(calls) == 1 and answer["prediction_calls"] == answer["response_evaluations"] == 7
    assert answer["unique_command_count"] == 6 and answer["request_ledger"][-1]["role"] == "initial_repeat"
    for name, rows in (("safety", 4369), ("pose", 24)):
        metadata = update[name]
        assert all(value.shape == (rows,) for value in metadata.values())
        assert np.all((metadata["theta"] >= 0) & (metadata["theta"] <= 1))
        assert np.any(metadata["updated"])


def test_next_qp_uses_causal_damped_feedback_while_its_anchor_and_mode_stay_unchanged(monkeypatch):
    calls = fake_qp(monkeypatch, [0.005, -0.002])
    inputs = failed_case(max_stages=2)
    before_Js, before_Jq = inputs["Js"].copy(), inputs["Jq"].copy()
    answer = planning.search(**inputs)
    first, second = answer["stages"]
    update = first["rejected_update"]
    source = answer["evaluated_candidates"][update["source_response_index"]]
    anchor = answer["evaluated_candidates"][0]
    sp, qp = baseline.causal_periods()
    Js, safety = baseline.causal_secant_update(
        before_Js, update["step"].ravel(), source["safety_slack"] - anchor["safety_slack"], sp, inputs["config"]
    )
    Jq, pose = baseline.causal_secant_update(
        before_Jq,
        update["step"].ravel(),
        (source["normalized_squared_errors"] - anchor["normalized_squared_errors"]).ravel(),
        qp,
        inputs["config"],
    )
    assert _canonical(safety) == _canonical(update["safety"])
    assert _canonical(pose) == _canonical(update["pose"])
    assert 0 < pose["theta"][0] < 1  # Large observed ceil miss is bounded by the registered damping.
    assert Jq[0, 0] == pytest.approx(0.6)
    for matrix, periods in ((Js, sp), (Jq, qp)):
        mask = np.repeat(np.arange(12), 6)[None, :] <= periods[:, None]
        assert not np.any(matrix[~mask])
    expected = baseline.build_stage_problem(
        anchor,
        Js,
        Jq,
        inputs["previous"],
        answer["global_lower"],
        answer["global_upper"],
        answer["pose_ceiling"],
        second["radius_before"],
        second["mode"],
        inputs["config"],
    )
    np.testing.assert_array_equal(calls[1]["A"], expected["A"])
    np.testing.assert_array_equal(calls[1]["b"], expected["b"])
    assert second["anchor_before_index"] == 0 and second["mode"] == first["mode"] == "pose"
    assert calls[0]["initial_t"] == calls[1]["initial_t"]
    assert answer["response_evaluations"] == 7 and answer["rejected_update_count"] == 1
    np.testing.assert_array_equal(inputs["Js"], before_Js)
    np.testing.assert_array_equal(inputs["Jq"], before_Jq)


@pytest.mark.parametrize("defect", ["unsafe", "joint", "ratio"])
def test_unsafe_joint_audit_or_reduction_failures_do_not_supply_feedback(monkeypatch, defect):
    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**failed_case(defect=defect))
    assert answer["stages"][0]["rejected_update"] is None
    assert answer["rejected_update_count"] == 0 and answer["final_anchor_index"] == 0
    assert answer["selected"] is None


@pytest.mark.parametrize("qualified,step", [(False, 0.005), (True, 0.0)])
def test_no_actual_response_means_no_feedback(monkeypatch, qualified, step):
    fake_qp(monkeypatch, [step], qualified=qualified)
    answer = planning.search(**failed_case())
    assert answer["rejected_update_count"] == 0 and answer["stages"][0]["rejected_update"] is None
    assert answer["response_evaluations"] == 2


def test_budget_termination_uses_the_smallest_completed_alpha_and_still_bills_repeat(monkeypatch):
    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**failed_case(max_responses=4))
    stage = answer["stages"][0]
    assert stage["termination"] == answer["termination_reason"] == "response_budget_exhausted"
    assert stage["attempts"][-1]["response_index"] is None
    assert stage["rejected_update"]["source_response_index"] == 2
    assert stage["rejected_update"]["alpha"] == 0.5
    assert answer["prediction_calls"] == answer["response_evaluations"] == 4
    assert answer["request_ledger"][-1]["role"] == "initial_repeat"


def test_an_accepted_stage_only_uses_its_accepted_response_even_after_eligible_rejection(monkeypatch):
    inputs = failed_case()
    original = inputs["evaluate"]

    def response(command):
        value = original(command)
        if 0 < command[0, 0] <= 0.004:
            value["tcp_errors"][0, 0] = 0.005
            value["normalized_squared_errors"][0, 0] = 0.25
            value["prefix_length"] = 9
            value["pose_loss"] = float(np.sum(value["normalized_squared_errors"]))
        return value

    inputs["evaluate"] = response
    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**inputs)
    stage = answer["stages"][0]
    assert stage["attempts"][0]["acceptance_reason"] == "actual_admission_failed"
    assert stage["accepted_response_index"] == 2 and stage["secant_update"] is not None
    assert stage["rejected_update"] is None and answer["rejected_update_count"] == 0
    assert answer["final_anchor_index"] == 2 and answer["selected"] is None


def source_case():
    inputs = failed_case()
    initial = inputs["historical_initial"]
    command = initial["commands"].copy()
    command[0, 0] = 0.001
    observed = inputs["evaluate"](command)
    attempt = dict(
        accepted=False,
        acceptance_reason="actual_admission_failed",
        response_index=1,
        actual_step=command - initial["commands"],
        alpha=0.5,
    )
    stage = dict(accepted_response_index=None, attempts=[attempt])
    ceiling = np.maximum(initial["normalized_squared_errors"].ravel(), 1) * 1.01
    return stage, [initial, observed], ceiling, inputs["config"]


def test_tiny_step_and_nonqualifying_actual_reasons_are_not_feedback_sources():
    stage, entries, ceiling, config = source_case()
    assert planning.choose_rejected_source(stage, entries, ceiling, config) is stage["attempts"][0]
    stage["attempts"][0]["actual_step"][:] = 0
    stage["attempts"][0]["actual_step"][0, 0] = 1e-7
    assert planning.choose_rejected_source(stage, entries, ceiling, config) is None
    stage["attempts"][0]["actual_step"][0, 0] = 1e-6
    assert planning.choose_rejected_source(stage, entries, ceiling, config) is stage["attempts"][0]
    stage["attempts"][0]["acceptance_reason"] = "ratio_insufficient"
    assert planning.choose_rejected_source(stage, entries, ceiling, config) is None


def test_source_rank_uses_alpha_then_response_index_instead_of_residual_size():
    stage, entries, ceiling, config = source_case()
    entries.append(deepcopy(entries[1]))
    stage["attempts"].append(deepcopy(stage["attempts"][0]))
    stage["attempts"][0]["response_index"] = 2
    assert planning.choose_rejected_source(stage, entries, ceiling, config)["response_index"] == 1
    stage["attempts"][0]["alpha"] = 0.25
    assert planning.choose_rejected_source(stage, entries, ceiling, config)["response_index"] == 2
    stage["accepted_response_index"] = 1
    assert planning.choose_rejected_source(stage, entries, ceiling, config) is None


def test_variant_shares_the_exact_callback_failure_type_and_retains_charged_prefix(monkeypatch):
    assert planning.SearchEvaluationError is baseline.SearchEvaluationError
    inputs = failed_case()
    original = inputs["evaluate"]

    def response(command):
        if np.any(command):
            raise RuntimeError("Synthetic response failure")
        return original(command)

    inputs["evaluate"] = response
    fake_qp(monkeypatch, [0.005])
    with pytest.raises(baseline.SearchEvaluationError) as caught:
        planning.search(**inputs)
    answer = caught.value.answer
    assert answer["prediction_calls"] == 2 and answer["response_evaluations"] == 1
    assert answer["stages"][0]["rejected_update"] is None and answer["rejected_update_count"] == 0
    assert not answer["request_ledger"][-1]["completed"]


def test_original_full_witness_gates_and_global_exact_repeat_are_unchanged(monkeypatch):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.00005 - 0.02 * command[0, 0]
        return entry(command, q=q)

    inputs = arguments(response, max_stages=1)
    inputs["Jq"][18, 0] = -0.02
    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**inputs)
    assert answer["selected_response_index"] == 1 and answer["qualified_witness_indices"] == [1]
    assert answer["initial_comparison"]["exact"] and answer["repeatability"]["exact"]
    assert np.all(answer["selected"]["normalized_squared_errors"] <= 1)
    assert answer["selected"]["audit"]["feasible"] and not answer["control_qualified"]
    assert answer["rejected_update_count"] == 0
