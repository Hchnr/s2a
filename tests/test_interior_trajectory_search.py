"""Synthetic response and fake-QP tests for the finite causal search policy."""

from copy import deepcopy

import numpy as np
import pytest

import s2a.interior_trajectory_search as planning


def entry(commands, *, reserve=2.0, q=None, joint=True, trace=0):
    q = np.full((12, 2), 0.25) if q is None else np.asarray(q).reshape(12, 2)
    errors = np.zeros((12, 6))
    errors[:, 0], errors[:, 3] = 0.01 * np.sqrt(q[:, 0]), 0.05 * np.sqrt(q[:, 1])
    squared = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    safety = np.full(4369, 0.001)
    safety[0] = 1e-8 + reserve * 1e-5
    failures = np.flatnonzero(np.any(squared > 1, axis=1))
    prefix = int(failures[0]) if len(failures) else 12
    return dict(
        commands=commands.copy(),
        rollout=dict(trace=trace),
        audit=dict(feasible=joint),
        tcp_errors=errors,
        normalized_squared_errors=squared,
        safety_slack=safety,
        prefix_length=prefix,
        maximum_normalized_squared_error=float(np.max(squared)),
        maximum_pose_cap_violation=float(max(0, np.sqrt(np.max(squared)) - 1)),
        pose_loss=float(np.sum(squared)),
        feasible=bool(joint and prefix == 12),
    )


def arguments(response=None, **config_overrides):
    def default(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.5 - 10 * command[0, 0]
        return entry(command, q=q)

    response = default if response is None else response
    initial = response(np.zeros((12, 6)))
    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))
    Jq[18, 0] = -10
    config = deepcopy(planning.DEFAULT_CONFIG)
    config.update(config_overrides)
    return dict(
        evaluate=response,
        historical_initial=initial,
        Js=Js,
        Jq=Jq,
        previous=np.zeros(6),
        input_min=-np.ones(6),
        input_max=np.ones(6),
        config=config,
    )


def fake_qp(monkeypatch, steps, *, qualified=True):
    calls, steps = [], iter(steps)

    def solve(B, g, A, b, initial_t, **kwargs):
        calls.append(dict(B=B.copy(), g=g.copy(), A=A.copy(), b=b.copy(), initial_t=initial_t, kwargs=kwargs))
        step = next(steps)
        point = np.zeros(73)
        point[0] = step
        return dict(qualified=qualified, point=point, solver_nit=3)

    monkeypatch.setattr(planning, "solve_epigraph_qp", solve)
    return calls


def test_periods_and_future_columns_cover_every_original_output_without_projection(monkeypatch):
    safety, pose = planning.causal_periods()
    assert safety.shape == (4369,) and pose.shape == (24,)
    assert list(safety[[0, 174, 175, 2099, 2100, 4199, 4200, 4283, 4284, 4367, 4368]]) == [
        0,
        0,
        1,
        11,
        0,
        11,
        0,
        11,
        0,
        11,
        11,
    ]
    np.testing.assert_array_equal(pose, np.repeat(np.arange(12), 2))
    inputs = arguments()
    inputs["Js"][0, 6] = np.nextafter(0.0, 1.0)
    calls = []
    inputs["evaluate"] = lambda command: calls.append(command)
    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("No QP before input admission"))
    with pytest.raises(ValueError, match="Future command"):
        planning.search(**inputs)
    assert not calls


@pytest.mark.parametrize("mode", ["restore", "pose"])
def test_stage_qp_has_all_pose_rows_original_hard_safety_and_unit_scaled_reserve(mode):
    inputs = arguments()
    anchor, config = inputs["historical_initial"], inputs["config"]
    inputs["Js"][0, 0], inputs["Js"][-1, 71] = 2, 3
    inputs["Jq"][23, 71] = 7
    q = anchor["normalized_squared_errors"].ravel()
    ceiling = np.maximum(q, 1) * 1.01
    problem = planning.build_stage_problem(
        anchor,
        inputs["Js"],
        inputs["Jq"],
        inputs["previous"],
        -np.ones((12, 6)),
        np.ones((12, 6)),
        ceiling,
        0.005,
        mode,
        config,
    )
    starts = problem["row_blocks"]
    assert problem["A"].shape == (9051 if mode == "restore" else 4706, 73)
    p0, p1 = starts["pose_ceiling"]
    np.testing.assert_array_equal(problem["A"][p0:p1, :72], -inputs["Jq"])
    np.testing.assert_array_equal(problem["b"][p0:p1], ceiling - q)
    assert problem["A"][p1 - 1, 71] == -7  # Last rotation endpoint is constrained too.
    s0, s1 = starts["safety"]
    np.testing.assert_array_equal(problem["A"][s0:s1, :72], inputs["Js"])
    e0, e1 = starts["mode_epigraph"]
    assert np.all(problem["A"][e0:e1, -1] == 1)
    other = problem["A"][:-1, -1].copy()
    other[e0:e1] = 0
    assert not np.any(other)
    if mode == "restore":
        assert problem["A"][e0, 0] == pytest.approx(2 / 1e-5)
        assert problem["A"][e1 - 1, 71] == pytest.approx(0.1 * 3 / 1e-5)
        assert np.all(problem["safety_floor"] == 1e-8)
    else:
        np.testing.assert_array_equal(problem["A"][e0:e1, :72], -inputs["Jq"])
        np.testing.assert_array_equal(problem["b"][e0:e1], 1 - q)
        assert problem["safety_floor"][0] == pytest.approx(1e-8 + 0.5e-5)
        assert problem["safety_floor"][-1] == pytest.approx(1e-8 + 0.5e-5 / 0.1)


def test_restore_can_temporarily_lose_a_prefix_without_selecting_it_for_execution(monkeypatch):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[0, 0], q[9, 0] = 0.999 + command[0, 0], 1.5 - 10 * command[0, 0]
        return entry(command, q=q, reserve=0.1 + 200 * command[0, 0])

    inputs = arguments(response, max_stages=1)
    inputs["Js"][0, 0] = 200e-5
    inputs["Jq"][0, 0] = 1
    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**inputs)
    stage = answer["stages"][0]
    assert stage["mode"] == "restore" and stage["accepted_response_index"] == 1
    assert answer["evaluated_candidates"][1]["prefix_length"] == 0
    assert answer["selected"] is None and answer["qualified_witness_indices"] == []
    assert answer["response_evaluations"] == 3 and answer["unique_command_count"] == 2
    assert stage["radius_after"] == 0.01 and stage["secant_update"] is not None


def test_pose_progress_can_trigger_another_restore_stage_without_resetting_the_anchor(monkeypatch):
    def response(command):
        x = command[0, 0]
        reserve = 0.1 + 200 * x if x <= 0.005 else (0.6 if x > 0.014 else 1.0)
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.5 - 10 * x
        return entry(command, q=q, reserve=reserve)

    inputs = arguments(response, max_stages=3)
    inputs["Js"][0, 0] = 200e-5
    calls = fake_qp(monkeypatch, [0.005, 0.01, -0.002])
    answer = planning.search(**inputs)
    assert [stage["mode"] for stage in answer["stages"]] == ["restore", "pose", "restore"]
    assert [stage["anchor_before_index"] for stage in answer["stages"]] == [0, 1, 2]
    assert [stage["anchor_after_index"] for stage in answer["stages"]] == [1, 2, 3]
    assert len(calls) == 3 and answer["total_qp_iterations"] == 9
    assert answer["final_anchor_index"] == 3 and answer["selected"] is None
    assert all(stage["secant_update"] is not None for stage in answer["stages"])


@pytest.mark.parametrize("actual_gain,accepted", [(0.2, True), (0.05, False)])
def test_acceptance_ratio_controls_rejection_and_trust_radius(monkeypatch, actual_gain, accepted):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.5 - actual_gain * 10 * command[0, 0]
        return entry(command, q=q)

    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**arguments(response, max_stages=1))
    stage = answer["stages"][0]
    assert stage["attempts"][0]["ratio"] == pytest.approx(actual_gain)
    assert stage["attempts"][0]["accepted"] is accepted
    assert stage["radius_after"] == 0.0025
    assert len(stage["attempts"]) == (1 if accepted else 5)
    assert (stage["secant_update"] is not None) is accepted


def test_already_tried_commands_are_never_evaluated_again_after_rejection(monkeypatch):
    def response(command):
        return entry(command, q=np.r_[np.full(18, 0.25), 1.5, np.full(5, 0.25)])

    fake_qp(monkeypatch, [0.005, 0.005])
    answer = planning.search(**arguments(response, max_stages=2))
    assert answer["response_evaluations"] == 7  # Initial, five trials, charged repeat.
    assert len(answer["stages"][1]["attempts"]) == 5
    assert all(attempt["response_index"] is None for attempt in answer["stages"][1]["attempts"])
    assert not answer["stages"][1]["attempts"][1]["admission"]["unique"]
    assert answer["request_ledger"][-1]["role"] == "initial_repeat"


def test_response_budget_always_reserves_one_charged_initial_repeat(monkeypatch):
    calls = fake_qp(monkeypatch, [0.005])
    answer = planning.search(**arguments(max_stages=2, max_responses=3))
    assert len(calls) == 1 and answer["termination_reason"] == "response_budget_exhausted"
    assert [row["role"] for row in answer["request_ledger"]] == ["initial", "trial", "initial_repeat"]
    assert answer["prediction_calls"] == answer["response_evaluations"] == 3
    assert answer["unique_command_count"] == 2 and answer["repeatability"]["exact"]


def test_initial_exact_mismatch_costs_initial_and_repeat_but_zero_qps(monkeypatch):
    inputs = arguments()
    inputs["historical_initial"]["rollout"]["trace"] = 1
    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("Historical mismatch stops QP"))
    answer = planning.search(**inputs)
    assert answer["termination_reason"] == "historical_initial_mismatch"
    assert answer["response_evaluations"] == 2 and not answer["stages"]
    assert not answer["initial_comparison"]["exact"] and answer["repeatability"]["exact"]
    assert answer["selected"] is None


def test_repeat_mismatch_disqualifies_an_original_full_witness(monkeypatch):
    calls = []

    def response(command):
        calls.append(None)
        return entry(command, trace=int(len(calls) > 2))

    inputs = arguments(response)
    monkeypatch.setattr(
        planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("Initial full witness needs no QP")
    )
    answer = planning.search(**inputs)
    assert answer["initial_comparison"]["exact"] and not answer["repeatability"]["exact"]
    assert answer["witness_indices"] == [0, 1] and answer["qualified_witness_indices"] == []
    assert answer["selected"] is None and answer["termination_reason"] == "repeat_mismatch"


def test_near_original_caps_can_finish_below_the_fixed_reduction_deadzone(monkeypatch):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.00005 - 0.02 * command[0, 0]
        return entry(command, q=q)

    inputs = arguments(response, max_stages=1)
    inputs["Jq"][18, 0] = -0.02
    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**inputs)
    attempt = answer["stages"][0]["attempts"][0]
    assert 0 < attempt["predicted_reduction"] < 1e-4
    assert attempt["accepted"] and attempt["acceptance_reason"] == "original_witness"
    assert answer["qualified_witness_indices"] == [1] and answer["selected_response_index"] == 1
    assert answer["termination_reason"] == "qualified_witness" and not answer["control_qualified"]


def test_fractional_threshold_allows_partial_progress_near_completion(monkeypatch):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.00005 - 0.02 * command[0, 0]
        return entry(command, q=q)

    inputs = arguments(response, max_stages=1)
    inputs["Jq"][18, 0] = -0.02
    fake_qp(monkeypatch, [0.0005])
    answer = planning.search(**inputs)
    attempt = answer["stages"][0]["attempts"][0]
    assert 0 < attempt["predicted_reduction"] < 1e-4
    assert 0 < attempt["actual_reduction"] < 1e-4
    assert attempt["accepted"] and attempt["acceptance_reason"] == "accepted_reduction"
    assert answer["selected"] is None and answer["evaluated_candidates"][1]["prefix_length"] == 9


@pytest.mark.parametrize("defect", ["pose_ceiling", "safety"])
def test_objective_progress_cannot_override_actual_safety_or_global_pose_ceiling(monkeypatch, defect):
    def response(command):
        q = np.full((12, 2), 0.25)
        q[9, 0] = 1.5 - 10 * command[0, 0]
        if defect == "pose_ceiling" and np.any(command):
            q[0, 0] = 1.02
        reserve = -0.1 if defect == "safety" and np.any(command) else 2.0
        return entry(command, q=q, reserve=reserve)

    fake_qp(monkeypatch, [0.005])
    answer = planning.search(**arguments(response, max_stages=1))
    assert answer["final_anchor_index"] == 0 and answer["selected"] is None
    assert all(attempt["actual_reduction"] > 0 for attempt in answer["stages"][0]["attempts"])
    assert all(attempt["acceptance_reason"] == "actual_admission_failed" for attempt in answer["stages"][0]["attempts"])


def test_secant_uses_each_rows_causal_step_prefix_and_bounds_the_correction_norm():
    config = planning.DEFAULT_CONFIG
    J = np.zeros((3, 72))
    J[:, 0] = [1, 1, 0]
    step = np.zeros(72)
    step[0], step[6] = 0.001, 0.002
    change = np.array([0.002, 0.004, 10.0])
    new, metadata = planning.causal_secant_update(J, step, change, np.array([0, 1, 0]), config)
    assert metadata["denominator"][0] == pytest.approx(1e-6)
    assert metadata["denominator"][1] == pytest.approx(5e-6)
    assert new[0, 0] == 2 and not np.any(new[0, 6:])
    assert new[1] @ step == pytest.approx(change[1])
    assert metadata["theta"][2] < 1
    assert np.linalg.norm(new[2] - J[2]) == pytest.approx(2e-6)
    np.testing.assert_array_equal(J[:, 0], [1, 1, 0])


def test_tiny_causal_steps_freeze_rows_even_when_future_steps_are_large():
    step = np.zeros(72)
    step[6] = 0.01
    J = np.zeros((2, 72))
    new, metadata = planning.causal_secant_update(J, step, np.ones(2), np.array([0, 1]), planning.DEFAULT_CONFIG)
    assert metadata["denominator"][0] == metadata["theta"][0] == 0
    assert not metadata["updated"][0] and not np.any(new[0])
    assert metadata["updated"][1] and new[1, 6] != 0


def test_unqualified_qps_shrink_without_responses_or_restarts(monkeypatch):
    calls = fake_qp(monkeypatch, [0, 0, 0, 0, 0, 0], qualified=False)
    answer = planning.search(**arguments())
    assert len(calls) == answer["total_stages"] == 6
    assert answer["termination_reason"] == "radius_below_minimum"
    assert answer["response_evaluations"] == 2 and answer["unique_command_count"] == 1
    assert all(stage["termination"] == "qp_unqualified" and not stage["attempts"] for stage in answer["stages"])


def test_callback_failure_keeps_attempt_and_previously_completed_response(monkeypatch):
    inputs = arguments()
    normal = inputs["evaluate"]

    def interrupted(command):
        if np.any(command):
            raise RuntimeError("Synthetic interrupted response")
        return normal(command)

    inputs["evaluate"] = interrupted
    fake_qp(monkeypatch, [0.005])
    with pytest.raises(planning.SearchEvaluationError) as caught:
        planning.search(**inputs)
    answer = caught.value.answer
    assert answer["prediction_calls"] == 2 and answer["response_evaluations"] == 1
    assert not answer["request_ledger"][-1]["completed"] and answer["request_ledger"][-1]["charged"]
    assert isinstance(caught.value.__cause__, RuntimeError)
