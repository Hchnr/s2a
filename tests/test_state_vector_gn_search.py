"""Signed-state GN algebra and finite search gates using synthetic responses."""

import numpy as np
import pytest
from test_interior_trajectory_search import arguments
from test_rejected_response_search import fake_qp

import s2a.state_vector_gn_search as planning
from s2a.directional_derivative_probe import _canonical
from s2a.interior_trajectory_search import build_stage_problem as original_problem

SCALE = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])


def entry(commands, E, *, reserve=2.0, joint=True, trace=0):
    errors = np.asarray(E).reshape(12, 6) * SCALE
    q = np.column_stack((np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2))
    safety = np.full(4369, 0.001)
    safety[0] = 1e-8 + reserve * 1e-5
    failures = np.flatnonzero(np.any(q > 1.0, axis=1))
    prefix = int(failures[0]) if len(failures) else 12
    return dict(
        commands=commands.copy(),
        rollout=dict(trace=trace),
        audit=dict(feasible=joint),
        tcp_errors=errors,
        normalized_squared_errors=q,
        safety_slack=safety,
        prefix_length=prefix,
        maximum_normalized_squared_error=float(np.max(q)),
        maximum_pose_cap_violation=float(max(0, np.sqrt(np.max(q)) - 1)),
        pose_loss=float(np.sum(q)),
        feasible=bool(joint and prefix == 12),
    )


def inputs(*, actual_derivative=-10.0, **config):
    E = np.zeros(72)
    E[54] = 1.5
    derivative = np.zeros(72)
    derivative[54] = actual_derivative

    def response(command):
        return entry(command, E + derivative * command[0, 0])

    result = arguments(response, max_stages=1, **config)
    result["Je"] = np.zeros((72, 72))
    result["Je"][54, 0] = -10
    result["Jq"][18, 0] = -30
    return result


def stencil():
    x = np.zeros((12, 6))
    x[0, 0] = 0.123456789
    E = np.linspace(-0.5, 0.5, 72)
    periods = np.repeat(np.arange(12), 6)
    J = np.diag(np.linspace(0.1, 2.0, 72))
    J[6:, 0] = 0.3
    assert not np.any(J[np.repeat(np.arange(12), 6)[None, :] > periods[:, None]])
    base = entry(x, E)
    nominal = np.where(np.arange(72) % 2, -1e-4, 1e-4)
    probes = []
    for axis, delta in enumerate(nominal):
        command = x.copy()
        command.flat[axis] += delta
        probes.append(dict(commands=command, tcp_errors=((E + J[:, axis] * delta).reshape(12, 6) * SCALE)))
    return dict(base=base, probes=probes, nominal_deltas=nominal), J


def test_error_vector_jacobian_accepts_original_compact_schema_and_uses_nominal_signed_steps():
    bundle, target = stencil()
    observed = planning.initial_error_vector_jacobian(bundle)
    # Normalize before subtracting, as required by the registered state vector.
    normalized_base = (bundle["base"]["tcp_errors"] / SCALE).ravel()
    reference = np.column_stack(
        [
            ((probe["tcp_errors"] / SCALE).ravel() - normalized_base) / delta
            for probe, delta in zip(bundle["probes"], bundle["nominal_deltas"], strict=True)
        ]
    )
    np.testing.assert_array_equal(observed, reference)
    np.testing.assert_allclose(observed, target, rtol=0, atol=1e-12)
    actual = bundle["probes"][0]["commands"].flat[0] - bundle["base"]["commands"].flat[0]
    assert actual != bundle["nominal_deltas"][0]
    assert observed[0, 0] != ((bundle["probes"][0]["tcp_errors"] / SCALE).ravel()[0] - normalized_base[0]) / actual
    assert all("normalized_squared_errors" not in probe and "pose_loss" not in probe for probe in bundle["probes"])


@pytest.mark.parametrize("defect", ["base_q", "probe_q", "future", "axis", "nominal", "missing"])
def test_invalid_base_algebra_stencil_and_future_derivatives_are_rejected(defect):
    bundle, _ = stencil()
    if defect == "base_q":
        bundle["base"]["normalized_squared_errors"][0, 0] += 1e-12
    elif defect == "probe_q":
        bundle["probes"][0]["normalized_squared_errors"] = np.zeros((12, 2))
    elif defect == "future":
        bundle["probes"][-1]["tcp_errors"][0, 0] += 1e-9
    elif defect == "axis":
        bundle["probes"][0]["commands"].flat[1] += 1e-10
    elif defect == "nominal":
        bundle["nominal_deltas"][0] = 2e-4
    else:
        bundle["probes"].pop()
    with pytest.raises(ValueError):
        planning.initial_error_vector_jacobian(bundle)


def test_pose_qp_is_spd_gauss_newton_with_no_soft_pose_epigraph_rows():
    value = inputs()
    anchor, config = value["historical_initial"], value["config"]
    Je = value["Je"].copy()
    Je[6, :2] = [3, 4]
    ceiling = planning.initial_global_envelope(anchor["normalized_squared_errors"].ravel(), config)
    problem = planning.build_stage_problem(
        anchor,
        value["Js"],
        value["Jq"],
        Je,
        value["previous"],
        -np.ones((12, 6)),
        np.ones((12, 6)),
        ceiling,
        0.005,
        "pose",
        config,
    )
    assert problem["A"].shape == (4682, 73) and "mode_epigraph" not in problem["row_blocks"]
    assert not np.any(problem["A"][:-1, -1]) and problem["A"][-1, -1] == 1
    assert problem["initial_t"] == 0
    np.testing.assert_array_equal(problem["B"], np.eye(72) + Je.T @ Je)
    E = (anchor["tcp_errors"] / SCALE).ravel()
    np.testing.assert_array_equal(problem["g"], Je.T @ E)
    np.linalg.cholesky(problem["B"])
    assert np.linalg.eigvalsh(problem["B"]).min() >= 1 - 1e-12
    d = np.linspace(-0.001, 0.001, 72)
    linear_quadratic = problem["g"] @ d + 0.5 * d @ problem["B"] @ d
    state_loss_change = 0.5 * (E + Je @ d) @ (E + Je @ d) - 0.5 * E @ E + 0.5 * d @ d
    assert linear_quadratic == pytest.approx(state_loss_change, abs=1e-15)


def test_restore_problem_is_exactly_the_frozen_problem_including_its_epigraph():
    value = inputs()
    anchor, config = value["historical_initial"], value["config"]
    arguments = (
        anchor,
        value["Js"],
        value["Jq"],
        value["previous"],
        -np.ones((12, 6)),
        np.ones((12, 6)),
        np.full(24, 2.3),
        0.005,
        "restore",
        config,
    )
    expected = original_problem(*arguments)
    observed = planning.build_stage_problem(*arguments[:3], value["Je"], *arguments[3:])
    assert _canonical(expected) == _canonical(observed)


def test_state_loss_may_improve_when_worst_error_increases_without_execution_qualification(monkeypatch):
    E = np.zeros(72)
    E[0], E[3] = 2, 2
    direction = np.zeros(72)
    direction[0], direction[3] = 0.5, -10

    def response(command):
        return entry(command, E + direction * command[0, 0])

    value = arguments(response, max_stages=1)
    value["Je"] = np.zeros((72, 72))
    value["Je"][:, 0] = direction
    value["Jq"][:] = 0
    value["Jq"][0, 0], value["Jq"][1, 0] = 2, -40
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**value)
    stage = answer["stages"][0]
    assert stage["accepted_response_index"] == 1
    assert stage["anchor_metrics_after"]["pose_excess"] > stage["anchor_metrics_before"]["pose_excess"]
    assert stage["anchor_metrics_after"]["state_loss"] < stage["anchor_metrics_before"]["state_loss"]
    assert stage["attempts"][0]["ratio"] == pytest.approx(1.0)
    assert answer["selected"] is None and answer["qualified_witness_indices"] == [] and not answer["control_qualified"]


def test_worst_error_improvement_is_rejected_when_actual_state_loss_does_not_fall(monkeypatch):
    E = np.zeros(72)
    E[0], E[3] = 2, 1

    def response(command):
        actual = E.copy()
        actual[0] -= 0.5 * command[0, 0]
        actual[3] += 10 * command[0, 0]
        return entry(command, actual)

    value = arguments(response, max_stages=1)
    value["Je"] = np.zeros((72, 72))
    value["Je"][[0, 3], 0] = -1
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**value)
    stage = answer["stages"][0]
    for attempt in stage["attempts"]:
        assert attempt["actual_metrics"]["pose_excess"] < stage["anchor_metrics_before"]["pose_excess"]
        assert attempt["actual_reduction"] < 0 and attempt["acceptance_reason"] == "actual_reduction_insufficient"
    assert answer["final_anchor_index"] == 0 and answer["rejected_update_count"] == 0


@pytest.mark.parametrize("derivative,accepted", [(-2.0, True), (-0.5, False)])
def test_ratio_uses_the_actual_original_q_loss_and_predicted_signed_vector_loss(monkeypatch, derivative, accepted):
    value = inputs(actual_derivative=derivative)
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**value)
    attempt = answer["stages"][0]["attempts"][0]
    initial = answer["evaluated_candidates"][0]
    actual = answer["evaluated_candidates"][1]
    E = (initial["tcp_errors"] / SCALE).ravel()
    predicted_E = E + value["Je"] @ attempt["actual_step"].ravel()
    expected_actual = 0.5 * (np.sum(initial["normalized_squared_errors"]) - np.sum(actual["normalized_squared_errors"]))
    expected_prediction = 0.5 * np.sum(initial["normalized_squared_errors"]) - 0.5 * predicted_E @ predicted_E
    assert attempt["actual_reduction"] == pytest.approx(expected_actual, abs=1e-15)
    assert attempt["predicted_reduction"] == pytest.approx(expected_prediction, abs=1e-15)
    assert attempt["ratio"] == pytest.approx(expected_actual / expected_prediction)
    assert attempt["accepted"] is accepted and answer["stages"][0]["radius_after"] == 0.0025


def test_accepted_error_secant_is_used_by_the_next_qp_and_retains_causality(monkeypatch):
    value = inputs(actual_derivative=-2)
    value["config"]["max_stages"] = 2
    calls = fake_qp(monkeypatch, [0.005, 0.001], module=planning)
    answer = planning.search(**value)
    update = answer["stages"][0]["secant_update"]["error_vector"]
    assert update["updated"].shape == (72,) and update["updated"][54]
    assert calls[0]["B"][0, 0] == 101
    assert calls[1]["B"][0, 0] == pytest.approx(5)
    assert calls[1]["g"][0] == pytest.approx(-2 * 1.49)
    assert answer["stages"][1]["anchor_before_index"] == 1
    source = answer["evaluated_candidates"][1]
    observed_change = planning.normalized_error_vector(source) - planning.normalized_error_vector(
        answer["evaluated_candidates"][0]
    )
    new, _ = planning.causal_secant_update(
        value["Je"],
        answer["stages"][0]["secant_update"]["step"].ravel(),
        observed_change,
        np.repeat(np.arange(12), 6),
        value["config"],
    )
    mask = np.repeat(np.arange(12), 6)[None, :] <= np.repeat(np.arange(12), 6)[:, None]
    assert not np.any(new[~mask])


def test_rejected_feedback_updates_Je_on_the_same_smallest_safe_source_without_reanchoring(monkeypatch):
    E = np.zeros(72)
    E[0], E[54] = np.sqrt(0.999), 1.5

    def response(command):
        actual = E.copy()
        if command[0, 0] != 0:
            actual[0] = np.sqrt(2.3)
        actual[54] -= 10 * command[0, 0]
        return entry(command, actual)

    value = arguments(response, max_stages=2)
    value["Je"] = np.zeros((72, 72))
    value["Je"][0, 0], value["Je"][54, 0] = 0.2, -10
    value["Jq"][18, 0] = -30
    calls = fake_qp(monkeypatch, [0.005, -0.002], module=planning)
    answer = planning.search(**value)
    update = answer["stages"][0]["rejected_update"]
    assert update["source_response_index"] == 5 and update["alpha"] == 0.0625
    assert update["error_vector"]["updated"][0] and 0 < update["error_vector"]["theta"][0] < 1
    assert calls[0]["B"][0, 0] == pytest.approx(101.04)
    assert calls[1]["B"][0, 0] == pytest.approx(101.36)
    assert answer["final_anchor_index"] == 0 and answer["selected"] is None
    assert answer["response_evaluations"] == 7 and answer["rejected_update_count"] == 1


def test_pose_minimum_improvement_still_scales_with_F_not_state_loss_near_caps(monkeypatch):
    E = np.zeros(72)
    E[54] = np.sqrt(1.00005)

    def response(command):
        actual = E.copy()
        actual[54] -= 0.02 * command[0, 0]
        return entry(command, actual)

    value = arguments(response, max_stages=1)
    value["Je"] = np.zeros((72, 72))
    value["Je"][54, 0] = -0.02
    fake_qp(monkeypatch, [0.0005], module=planning)
    answer = planning.search(**value)
    attempt = answer["stages"][0]["attempts"][0]
    assert 0 < attempt["actual_reduction"] < 1e-4 and attempt["accepted"]
    assert attempt["acceptance_reason"] == "accepted_reduction" and answer["selected"] is None


def test_sixty_stage_permission_preserves_the_response_limit_and_paid_repeat(monkeypatch):
    value = inputs(max_responses=3)
    value["config"]["max_stages"] = 60
    calls = fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**value)
    assert len(calls) == 1 and answer["termination_reason"] == "response_budget_exhausted"
    assert answer["prediction_calls"] == answer["response_evaluations"] == 3 and answer["unique_command_count"] == 2
    assert [row["role"] for row in answer["request_ledger"]] == ["initial", "trial", "initial_repeat"]
    value["config"]["max_responses"] = 123
    with pytest.raises(ValueError):
        planning.search(**value)


def test_future_Je_is_rejected_before_any_callback_or_qp(monkeypatch):
    value = inputs()
    value["Je"][0, 6] = 1e-300
    calls = []
    value["evaluate"] = lambda command: calls.append(command)
    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_k: pytest.fail("Future derivative is invalid"))
    with pytest.raises(ValueError, match="Future command"):
        planning.search(**value)
    assert not calls


def test_unsafe_loss_progress_never_becomes_an_anchor_or_feedback_source(monkeypatch):
    value = inputs()
    original = value["evaluate"]

    def response(command):
        result = original(command)
        if np.any(command):
            result["safety_slack"][0] = -1e-9
        return result

    value["evaluate"] = response
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**value)
    assert answer["final_anchor_index"] == 0 and answer["selected"] is None and answer["rejected_update_count"] == 0
    assert all(attempt["acceptance_reason"] == "actual_admission_failed" for attempt in answer["stages"][0]["attempts"])


def test_full_original_witness_and_exact_final_repeat_are_still_required(monkeypatch):
    value = inputs()
    E = np.zeros(72)
    E[54] = 1.00001

    def response(command):
        actual = E.copy()
        actual[54] -= 10 * command[0, 0]
        return entry(command, actual)

    value["historical_initial"] = response(np.zeros((12, 6)))
    value["evaluate"] = response
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**value)
    assert answer["qualified_witness_indices"] == [1] and answer["selected_response_index"] == 1
    assert answer["initial_comparison"]["exact"] and answer["repeatability"]["exact"]
    assert np.all(answer["selected"]["normalized_squared_errors"] <= 1) and not answer["control_qualified"]


def test_a_complete_candidate_cannot_be_selected_after_the_final_repeat_changes(monkeypatch):
    value = inputs()
    E = np.zeros(72)
    E[54] = 1.00001
    calls = []

    def response(command):
        calls.append(None)
        actual = E.copy()
        actual[54] -= 10 * command[0, 0]
        return entry(command, actual, trace=int(len(calls) > 3))

    value["historical_initial"] = response(np.zeros((12, 6)))
    value["evaluate"] = response
    fake_qp(monkeypatch, [0.005], module=planning)
    answer = planning.search(**value)
    assert answer["witness_indices"] == [1] and answer["qualified_witness_indices"] == []
    assert answer["initial_comparison"]["exact"] and not answer["repeatability"]["exact"]
    assert answer["selected"] is None and answer["termination_reason"] == "repeat_mismatch"
    assert answer["prediction_calls"] == answer["response_evaluations"] == 3


def test_shared_callback_failure_keeps_actual_charged_prefix(monkeypatch):
    value = inputs()
    original = value["evaluate"]

    def response(command):
        if np.any(command):
            raise RuntimeError("Synthetic interrupted GN response")
        return original(command)

    value["evaluate"] = response
    fake_qp(monkeypatch, [0.005], module=planning)
    with pytest.raises(planning.SearchEvaluationError) as caught:
        planning.search(**value)
    answer = caught.value.answer
    assert answer["prediction_calls"] == 2 and answer["response_evaluations"] == 1
    assert answer["request_ledger"][-1]["charged"] and not answer["request_ledger"][-1]["completed"]
    assert isinstance(caught.value.__cause__, RuntimeError)
