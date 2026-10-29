"""Hand-certified GN and restoration ledgers; no optimizer or native model."""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_interior_trajectory_search_audit import certificate_from_point, failed_certificate, fixture_inputs
from test_rejected_response_search_audit import capped_restoration_certificate, nonlinear_inputs
from test_tail_translation_reserve import synthetic_case

import s2a.interior_trajectory_search as original_config
import s2a.state_vector_gn_search as core
from s2a.constraint_residual_repair import evaluate_repair_entry


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("state_vector_gn_search_audit")


def vector_inputs(*, stages=2, responses=122, curvature=0.0, worst=3.0, tail=2.0, slope=-20.0, actual_slope=None):
    """A constant worst endpoint exposes state-loss progress without F progress."""
    case = synthetic_case(base=2e-4, terminal=lambda command: 2e-4, full_witness=True)
    config = deepcopy(original_config.DEFAULT_CONFIG)
    config.update(max_stages=stages, max_responses=responses)
    Js, Jq, Je = np.zeros((4369, 72)), np.zeros((24, 72)), np.zeros((72, 72))
    Jq[22, 66], Je[66, 66] = 2.0 * tail * slope, slope
    calls = []
    actual_slope = slope if actual_slope is None else actual_slope

    def evaluate(command):
        calls.append(command.tobytes())
        rollout = case["predict"](command)
        delta = command[11, 0]
        error = tail + actual_slope * delta + curvature * delta**2
        rollout["pose"][10, 0, 3], rollout["pose"][11, 0, 3] = -0.01 * worst, -0.01 * error
        for index in (10, 11):
            rollout["periods"][index]["endpoint_pose"] = rollout["pose"][index].copy()
        return evaluate_repair_entry(
            rollout,
            command,
            case["previous"],
            case["interface"],
            case["lower"],
            case["upper"],
            case["desired"],
            case["site_from_tcp"],
            case["past_inner_error_rad"],
            case["terminal_normals"],
        )

    initial = evaluate(case["anchor"])
    calls.clear()
    arguments = (
        Js,
        Jq,
        Je,
        initial,
        case["previous"],
        case["interface"].input_min,
        case["interface"].input_max,
        config,
    )
    return evaluate, arguments, calls


def hand_certificate(B, g, A, b, initial_t, **kwargs):
    """The one-coordinate GN optimum is -g/B clamped to its hard interval."""
    if len(A) > 6000:
        return capped_restoration_certificate(B, g, A, b, initial_t, **kwargs)
    assert initial_t == 0.0 and len(A) == 4682
    point, dual = np.zeros(73), np.zeros(len(A))
    axis = 66
    assert np.all(g[np.arange(72) != axis] == 0)
    off_diagonal = B.copy()
    np.fill_diagonal(off_diagonal, 0)
    assert not np.any(off_diagonal)
    low, high, low_row, high_row = -np.inf, np.inf, None, None
    for index, coefficient in enumerate(A[:, axis]):
        if not coefficient or A[index, -1]:
            continue
        bound = -b[index] / coefficient
        if coefficient > 0 and bound > low:
            low, low_row = bound, index
        elif coefficient < 0 and bound < high:
            high, high_row = bound, index
    point[axis] = np.clip(-g[axis] / B[axis, axis], low, high)
    gradient = B[axis, axis] * point[axis] + g[axis]
    if point[axis] == low and gradient > 0:
        dual[low_row] = gradient / A[low_row, axis]
    elif point[axis] == high and gradient < 0:
        dual[high_row] = gradient / A[high_row, axis]
    dual[-1] = 1.0
    return certificate_from_point(B, g, A, b, point, dual)


def run(monkeypatch, evaluate, arguments, solver=hand_certificate):
    Js, Jq, Je, initial, previous, lower, upper, config = arguments
    monkeypatch.setattr(core, "solve_epigraph_qp", solver)
    return core.search(evaluate, initial, Js, Jq, Je, previous, lower, upper, config)


@pytest.fixture
def archive(monkeypatch):
    evaluate, arguments, calls = vector_inputs(curvature=20.0)
    return run(monkeypatch, evaluate, arguments), arguments, calls


def test_gn_accepts_state_loss_progress_even_when_the_original_maximum_is_constant(audit, archive):
    answer, arguments, calls = archive
    checked = audit.audit_search(answer, *arguments)
    assert checked["accepted_response_indices"] == [1, 2]
    assert checked["response_evaluations"] == len(calls) == 4
    stage = answer["stages"][0]
    assert stage["anchor_metrics_before"]["pose_excess"] == stage["anchor_metrics_after"]["pose_excess"] == 8.0
    assert stage["anchor_metrics_after"]["state_loss"] < stage["anchor_metrics_before"]["state_loss"]
    assert stage["attempts"][0]["actual_reduction"] > 0
    assert answer["pose_objective_policy"] == "normalized_state_vector_gauss_newton"
    assert checked["pose_damping"] == 1.0 and not checked["control_qualified"]
    assert checked["witness_indices"] == [] and checked["selected_response_index"] is None


def test_independent_problem_has_full_gn_hessian_gradient_and_only_the_t_nonnegative_row(audit, archive):
    answer, arguments, _ = archive
    Js, Jq, Je, initial, previous, _, _, config = arguments
    Je = Je.copy()
    Je[66, 67] = -10.0
    error = audit._error(initial)
    B, g, A, b = audit._problem(
        initial["commands"],
        initial["safety_slack"],
        initial["normalized_squared_errors"].ravel(),
        error,
        Js,
        Jq,
        Je,
        previous,
        answer["global_lower"],
        answer["global_upper"],
        answer["pose_ceiling"],
        config["radius_initial"],
        "pose",
        config,
    )
    expected = np.eye(72)
    expected[66, 66], expected[67, 67], expected[66, 67], expected[67, 66] = 401, 101, 200, 200
    np.testing.assert_array_equal(B, expected)
    np.testing.assert_array_equal(g[66:68], [-40.0, -20.0])
    assert np.count_nonzero(g) == 2
    assert A.shape == (4682, 73) and b.shape == (4682,)
    np.testing.assert_array_equal(np.flatnonzero(A[:, -1]), [4681])
    np.testing.assert_array_equal(A[-1], np.r_[np.zeros(72), 1.0])
    assert b[-1] == 0


def restoration_inputs(*, feedback=False):
    if feedback:
        evaluate, old = nonlinear_inputs()
    else:
        _, evaluate, old = fixture_inputs()
    Js, Jq, initial, previous, lower, upper, config = old
    Je = np.zeros((72, 72))
    if feedback:
        Je[66, 66] = -7.5
    return evaluate, (Js, Jq, Je, initial, previous, lower, upper, config)


def test_restoration_keeps_the_frozen_problem_exact_and_needs_no_gn_descent(audit, monkeypatch):
    evaluate, arguments = restoration_inputs()
    answer = run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    assert [stage["mode"] for stage in checked["stage_checks"]] == ["restore", "pose"]
    assert checked["accepted_response_indices"] == [1]
    assert answer["stages"][0]["anchor_metrics_before"]["state_loss"] == 6.0
    assert answer["stages"][0]["attempts"][0]["actual_reduction"] > 0
    assert answer["stages"][0]["anchor_metrics_after"]["state_loss"] == 6.0
    frozen = importlib.import_module("interior_trajectory_search_audit")
    Js, Jq, Je, initial, previous, _, _, config = arguments
    common = (initial["commands"], initial["safety_slack"], initial["normalized_squared_errors"].ravel())
    tail = (
        previous,
        answer["global_lower"],
        answer["global_upper"],
        answer["pose_ceiling"],
        config["radius_initial"],
        "restore",
        config,
    )
    expected = frozen._problem(*common, Js, Jq, *tail)
    observed = audit._problem(*common, audit._error(initial), Js, Jq, Je, *tail)
    for actual, wanted in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_near_cap_progress_floor_uses_f_not_the_larger_state_loss(audit, monkeypatch):
    evaluate, arguments, _ = vector_inputs(stages=1, worst=np.sqrt(1.0001), tail=0.1, slope=-0.1)
    answer = run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    attempt = answer["stages"][0]["attempts"][0]
    assert 1e-5 < attempt["actual_reduction"] < 1e-4
    assert attempt["accepted"] and checked["accepted_response_indices"] == [1]
    assert checked["qualified_witness_indices"] == []


@pytest.mark.parametrize("actual_slope,reason", [(1.0, "actual_reduction_insufficient"), (-0.5, "ratio_insufficient")])
def test_actual_state_loss_and_ratio_are_both_required(audit, monkeypatch, actual_slope, reason):
    evaluate, arguments, _ = vector_inputs(stages=1, actual_slope=actual_slope)
    answer = run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    assert checked["accepted_response_indices"] == []
    assert all(attempt["acceptance_reason"] == reason for attempt in answer["stages"][0]["attempts"])
    assert checked["rejected_update_count"] == 0 and checked["response_evaluations"] == 7


@pytest.mark.parametrize("defect", ["hessian", "gradient", "old_soft_rows", "state_loss", "ratio", "old_F_reduction"])
def test_old_objective_and_forged_metrics_cannot_pass_as_gn(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    stage, attempt = answer["stages"][0], answer["stages"][0]["attempts"][0]
    if defect == "hessian":
        stage["qp"]["B"] = np.eye(72)
    elif defect == "gradient":
        stage["qp"]["objective"] -= 1.0
    elif defect == "old_soft_rows":
        stage["qp"]["row_scale"] = np.r_[stage["qp"]["row_scale"], np.ones(24)]
    elif defect == "state_loss":
        attempt["predicted_metrics"]["state_loss"] = attempt["predicted_metrics"]["pose_excess"]
    elif defect == "ratio":
        attempt["ratio"] += 0.01
    else:
        attempt["predicted_reduction"] = attempt["actual_reduction"] = 0.0
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_old_max_epigraph_certificate_is_rejected_even_with_valid_legacy_kkt(audit, archive):
    answer, arguments, _ = deepcopy(archive)
    frozen = importlib.import_module("interior_trajectory_search_audit")
    Js, Jq, _, initial, previous, _, _, config = arguments
    B, g, A, b = frozen._problem(
        initial["commands"],
        initial["safety_slack"],
        initial["normalized_squared_errors"].ravel(),
        Js,
        Jq,
        previous,
        answer["global_lower"],
        answer["global_upper"],
        answer["pose_ceiling"],
        config["radius_initial"],
        "pose",
        config,
    )
    point, dual = np.zeros(73), np.zeros(len(A))
    row = 4369 + 24 + 20
    point[-1], dual[row] = 8.0, 1.0
    certificate = certificate_from_point(B, g, A, b, point, dual)
    assert frozen.audit_qp_certificate(certificate, B, g, A, b)["qualified"]
    answer["stages"][0]["qp"] = certificate
    with pytest.raises(ValueError, match="QP input B binding"):
        audit.audit_search(answer, *arguments)


def test_omitted_gn_linear_term_cannot_hide_behind_the_correct_hessian(audit, archive):
    answer, arguments, _ = deepcopy(archive)
    Js, Jq, Je, initial, previous, _, _, config = arguments
    B, g, A, b = audit._problem(
        initial["commands"],
        initial["safety_slack"],
        initial["normalized_squared_errors"].ravel(),
        audit._error(initial),
        Js,
        Jq,
        Je,
        previous,
        answer["global_lower"],
        answer["global_upper"],
        answer["pose_ceiling"],
        config["radius_initial"],
        "pose",
        config,
    )
    point, dual = np.zeros(73), np.zeros(len(A))
    dual[-1] = 1.0
    wrong_g = np.zeros(72)
    certificate = certificate_from_point(B, wrong_g, A, b, point, dual)
    assert np.any(g) and audit.audit_qp_certificate(certificate, B, wrong_g, A, b)["qualified"]
    answer["stages"][0]["qp"] = certificate
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_accepted_je_is_bound_to_the_next_qp_and_all_model_metadata(audit, archive):
    answer, arguments, _ = deepcopy(archive)
    checked = audit.audit_search(answer, *arguments)
    assert checked["accepted_response_indices"] == [1, 2]
    assert answer["stages"][0]["secant_update"]["error_vector"]["updated"][66]
    first_B = answer["stages"][0]["qp"]["B"]
    assert not np.array_equal(first_B, answer["stages"][1]["qp"]["B"])
    answer["stages"][1]["qp"]["B"] = first_B.copy()
    with pytest.raises(ValueError, match="QP input B binding"):
        audit.audit_search(answer, *arguments)


@pytest.fixture
def feedback_archive(monkeypatch):
    evaluate, arguments = restoration_inputs(feedback=True)
    return run(monkeypatch, evaluate, arguments), arguments


def test_safe_minimum_alpha_feedback_updates_je_without_moving_anchor(audit, feedback_archive):
    answer, arguments = feedback_archive
    checked = audit.audit_search(answer, *arguments)
    assert checked["rejected_response_indices"] == [5] and checked["accepted_response_indices"] == [10]
    stage = answer["stages"][0]
    update = stage["rejected_update"]
    assert update["alpha"] == 0.0625 and update["error_vector"]["updated"][66]
    assert stage["anchor_before_index"] == stage["anchor_after_index"] == 0
    assert stage["radius_after"] == stage["radius_before"] / 2


@pytest.mark.parametrize("defect", ["metadata", "future_denominator", "wrong_source", "anchor", "stale_prediction"])
def test_feedback_je_causality_source_and_next_prediction_are_bound(audit, feedback_archive, defect):
    answer, arguments = deepcopy(feedback_archive)
    update = answer["stages"][0]["rejected_update"]
    if defect == "metadata":
        update["error_vector"]["residual"][66] += 1e-3
    elif defect == "future_denominator":
        update["error_vector"]["denominator"][0] = 1e-30
    elif defect == "wrong_source":
        update["source_response_index"] = 1
    elif defect == "anchor":
        answer["stages"][0]["anchor_after_index"] = 5
    else:
        attempt = next(v for v in answer["stages"][1]["attempts"] if v["response_index"] is not None)
        initial_error = audit._error(arguments[3])
        stale = initial_error + arguments[2] @ attempt["actual_step"].ravel()
        attempt["predicted_metrics"]["state_loss"] = float(0.5 * (stale @ stale))
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_only_original_full_caps_and_exact_repeat_allow_selection(audit, monkeypatch):
    evaluate, arguments, calls = vector_inputs(stages=3, worst=0.0, tail=1.05)
    answer = run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    assert checked["qualified_witness_indices"] == [1] and checked["selected_response_index"] == 1
    assert checked["termination_reason"] == "qualified_witness" and len(calls) == 3
    assert checked["repeat_exact"] and not checked["control_qualified"]


@pytest.mark.parametrize("responses", [2, 7])
def test_response_budget_includes_initial_and_mandatory_repeat(audit, monkeypatch, responses):
    evaluate, arguments, calls = vector_inputs(stages=60, responses=responses)
    answer = run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    assert checked["response_evaluations"] == len(calls) == responses
    assert checked["termination_reason"] == "response_budget_exhausted"
    assert answer["request_ledger"][-1]["role"] == "initial_repeat" and calls[0] == calls[-1]


@pytest.mark.parametrize("kind", ["stage", "response"])
def test_a_complete_charged_ledger_cannot_exceed_its_registered_budget(audit, archive, kind):
    answer, arguments, _ = deepcopy(archive)
    config = arguments[-1]
    if kind == "stage":
        config["max_stages"] = len(answer["stages"]) - 1
    else:
        config["max_responses"] = len(answer["evaluated_candidates"]) - 1
    answer["config"] = deepcopy(config)
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize(
    "defect", ["Js_future", "Jq_future", "Je_future", "safety", "repeat", "selected", "counter", "policy", "damping"]
)
def test_original_safety_history_budget_and_objective_contract_remain_strict(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    if defect.endswith("_future"):
        arguments[{"Js_future": 0, "Jq_future": 1, "Je_future": 2}[defect]][0, 6] = 1e-30
    elif defect == "safety":
        answer["evaluated_candidates"][1]["safety_slack"][0] = np.nextafter(1e-8, -np.inf)
    elif defect == "repeat":
        answer["repeatability"]["commands_bytes_equal"] = False
    elif defect == "selected":
        answer["selected_response_index"] = 1
    elif defect == "counter":
        answer["prediction_calls"] += 1
    elif defect == "policy":
        answer["pose_objective_policy"] = "maximum_epigraph"
    else:
        answer["pose_damping"] = 0.5
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize(
    "key,value", [("max_stages", 61), ("max_responses", 123), ("seed", 43), ("safety_buffer", 1e-7)]
)
def test_complete_frozen_config_is_validated_before_reading_a_ledger(audit, key, value):
    config = deepcopy(original_config.DEFAULT_CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        audit.audit_search(None, None, None, None, None, None, None, None, config)


def test_unqualified_qp_cannot_create_trial_or_je_feedback(audit, monkeypatch):
    evaluate, arguments, calls = vector_inputs(stages=1)
    answer = run(monkeypatch, evaluate, arguments, failed_certificate)
    checked = audit.audit_search(answer, *arguments)
    assert checked["response_evaluations"] == len(calls) == 2
    assert checked["rejected_update_count"] == 0 and checked["accepted_response_indices"] == []
    assert answer["stages"][0]["secant_update"] is answer["stages"][0]["rejected_update"] is None


def test_audit_imports_no_search_core_solver_or_dynamics(audit):
    imports = []
    for node in ast.walk(ast.parse(Path(audit.__file__).read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("s2a." in name or "scipy" in name or "mujoco" in name for name in imports)
    assert "accepted_point_sqp_math_audit" in imports


def test_compact_signed_stencil_is_reconstructed_without_optional_redundant_labels(audit):
    base = synthetic_case(full_witness=True)["historical_anchor"]
    nominal = np.where(np.arange(72) % 2 == 0, 1e-4, -1e-4)
    scales = np.tile([0.01, 0.01, 0.01, 0.05, 0.05, 0.05], 12)
    probes, expected = [], np.zeros((72, 72))
    for axis, delta in enumerate(nominal):
        command, error = base["commands"].copy(), np.zeros(72)
        command.ravel()[axis] += delta
        expected[axis, axis] = axis + 1.0
        error[axis] = (axis + 1.0) * delta * scales[axis]
        if axis + 6 < 72:
            expected[axis + 6, axis] = 0.5
            error[axis + 6] = 0.5 * delta * scales[axis + 6]
        probe = dict(commands=command, tcp_errors=error.reshape(12, 6))
        if axis == 0:
            tcp = probe["tcp_errors"]
            probe["normalized_squared_errors"] = np.column_stack(
                (np.sum(tcp[:, :3] ** 2, axis=1) / 0.01**2, np.sum(tcp[:, 3:] ** 2, axis=1) / 0.05**2)
            )
        probes.append(probe)
    bundle = dict(base=base, probes=probes, nominal_deltas=nominal)
    np.testing.assert_allclose(audit.reconstruct_error_vector_jacobian(bundle), expected, rtol=0, atol=1e-12)
    for defect in ("nominal", "signed_axis", "q_label", "future"):
        malformed = deepcopy(bundle)
        if defect == "nominal":
            malformed["nominal_deltas"][0] = 1e-5
        elif defect == "signed_axis":
            malformed["probes"][0]["commands"][0, 0] *= -1
        elif defect == "q_label":
            malformed["probes"][0]["normalized_squared_errors"][0, 0] = 1.0
        else:
            malformed["probes"][6]["tcp_errors"][0, 0] = 1e-30
        with pytest.raises(ValueError):
            audit.reconstruct_error_vector_jacobian(malformed)
