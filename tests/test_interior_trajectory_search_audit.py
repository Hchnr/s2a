"""Analytic QP archives and synthetic full responses exercise the independent audit.

No QP solver, simulator, model service, or GPU is invoked by these tests.
"""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_tail_translation_reserve import synthetic_case

import s2a.interior_trajectory_search as search_core
from s2a.constraint_residual_repair import evaluate_repair_entry


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("interior_trajectory_search_audit")


def analytic_certificate(B, g, A, b, initial_t, **kwargs):
    """Certify the fixture's known terminal-restoration or constant-pose optimum."""
    del initial_t, kwargs
    d, dual = np.zeros(72), np.zeros(len(A))
    if len(A) > 6000:
        row = 4369 + 24 + 4368
        axis = 66
        d[axis] = -b[row] / A[row, axis]
        t = 0.0
        dual[row] = d[axis] / A[row, axis]
        dual[-1] = 1.0 - dual[row]
    else:
        row = 4369 + 24 + 18
        t = -b[row]
        dual[row] = 1.0
    point = np.r_[d, t]
    return certificate_from_point(B, g, A, b, point, dual)


def certificate_from_point(B, g, A, b, point, dual):
    """Recompute every diagnostic of a hand-proved optimum, without a solver."""
    d, t = point[:72], point[-1]
    scale = np.maximum.reduce((np.ones(len(A)), np.max(np.abs(A), axis=1), np.abs(b)))
    mu = dual * scale
    slack = A / scale[:, None] @ point + b / scale
    raw_slack = A @ point + b
    active = np.flatnonzero(slack <= 1e-5)
    objective = float(t + g @ d + 0.5 * d @ B @ d)
    gradient = np.r_[B @ d + g, 1.0]
    force = (A / scale[:, None]).T @ mu
    stationarity = gradient - force
    complementarity = mu * slack
    return dict(
        B=B.copy(),
        point=point,
        solver_success=True,
        solver_status=0,
        solver_message="Analytic fixture; no optimizer was run",
        solver_nit=1,
        solver_exception=None,
        qualified=True,
        objective=objective,
        multipliers=dual,
        scaled_multipliers=mu,
        row_scale=scale,
        active_indices=active.tolist(),
        primal_min=float(np.min(slack)),
        unscaled_primal_min=float(np.min(raw_slack)),
        stationarity_relative=float(np.max(np.abs(stationarity)))
        / max(1.0, float(np.max(np.abs(gradient))), float(np.max(np.abs(force)))),
        stationarity_residual_inf=float(np.max(np.abs(stationarity))),
        complementarity_relative=float(np.max(np.abs(complementarity))) / max(1.0, abs(objective)),
        complementarity_residual_inf=float(np.max(np.abs(complementarity))),
        complementarity_sum_gap=float(np.sum(complementarity)),
        certificate_tolerance=1e-6,
        nnls_success=True,
        nnls_error=None,
        nnls_residual_norm=float(np.linalg.norm(stationarity)),
        nnls_max_iterations=3 * max(len(active), 1),
        qualification_failures=[],
    )


def failed_certificate(B, g, A, b, initial_t, **kwargs):
    del g, initial_t, kwargs
    return dict(
        B=B.copy(),
        point=None,
        solver_success=False,
        solver_status=None,
        solver_nit=0,
        solver_exception="Synthetic unavailable algebra; no optimizer was run",
        qualified=False,
        certificate_tolerance=1e-6,
        nnls_success=False,
        qualification_failures=["solver_exception"],
        row_scale=np.maximum.reduce((np.ones(len(A)), np.max(np.abs(A), axis=1), np.abs(b))),
        objective=None,
        primal_min=None,
        unscaled_primal_min=None,
        active_indices=[],
        nnls_max_iterations=None,
        multipliers=None,
        scaled_multipliers=None,
        nnls_residual_norm=None,
        stationarity_relative=None,
        stationarity_residual_inf=None,
        complementarity_relative=None,
        complementarity_residual_inf=None,
        complementarity_sum_gap=None,
    )


def fixture_inputs():
    case = synthetic_case()
    config = deepcopy(search_core.DEFAULT_CONFIG)
    config["max_stages"] = 2
    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))
    Js[-1, 66] = 0.1

    def evaluate(command):
        return evaluate_repair_entry(
            case["predict"](command),
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

    arguments = (
        Js,
        Jq,
        case["historical_anchor"],
        case["previous"],
        case["interface"].input_min,
        case["interface"].input_max,
        config,
    )
    return case, evaluate, arguments


@pytest.fixture
def archive(monkeypatch):
    _, evaluate, arguments = fixture_inputs()
    Js, Jq, historical, previous, lower, upper, config = arguments
    monkeypatch.setattr(search_core, "solve_epigraph_qp", analytic_certificate)
    answer = search_core.search(evaluate, historical, Js, Jq, previous, lower, upper, config)
    return answer, arguments


def test_independent_full_ledger_binds_known_qps_modes_and_actual_updates(audit, archive):
    answer, arguments = archive
    checked = audit.audit_search(answer, *arguments)
    assert checked["passed"] and checked["response_evaluations"] == 3
    assert [stage["mode"] for stage in checked["stage_checks"]] == ["restore", "pose"]
    assert checked["accepted_response_indices"] == [1]
    assert checked["termination_reason"] == "stage_budget_exhausted"
    assert checked["selected_response_index"] is None
    assert checked["final_anchor_index"] == 1
    assert not checked["full_responses_independently_replayed_here"]
    assert not checked["control_qualified"]


@pytest.mark.parametrize(
    "defect",
    [
        "ratio",
        "radius",
        "predicted",
        "ceiling",
        "denominator",
        "theta",
        "updated",
        "step",
        "matrix",
        "omit_response",
        "orphan_response",
        "selected_anchor",
        "counter",
        "future",
        "repeat",
    ],
)
def test_independent_audit_rejects_control_flow_matrix_and_secant_forgery(audit, archive, defect):
    answer, arguments = deepcopy(archive)
    stage = answer["stages"][0]
    if defect == "ratio":
        stage["attempts"][0]["ratio"] += 0.1
    elif defect == "radius":
        stage["radius_after"] *= 2
    elif defect == "predicted":
        stage["attempts"][0]["predicted_metrics"]["reserve"] += 0.1
    elif defect == "ceiling":
        answer["pose_ceiling"][18] *= 1.01
    elif defect == "denominator":
        stage["secant_update"]["safety"]["denominator"][-1] *= 2
    elif defect == "theta":
        stage["secant_update"]["safety"]["theta"][-1] *= 0.5
    elif defect == "updated":
        stage["secant_update"]["safety"]["updated"][-1] = not stage["secant_update"]["safety"]["updated"][-1]
    elif defect == "step":
        stage["secant_update"]["step"][0, 0] = 1e-8
    elif defect == "matrix":
        stage["qp"]["B"][0, 0] = 2.0
    elif defect == "omit_response":
        answer["evaluated_candidates"].pop(1)
    elif defect == "orphan_response":
        answer["stages"] = []
    elif defect == "selected_anchor":
        answer["selected_response_index"] = answer["final_anchor_index"]
        answer["selected"] = deepcopy(answer["evaluated_candidates"][answer["final_anchor_index"]])
    elif defect == "counter":
        answer["prediction_calls"] -= 1
    elif defect == "future":
        arguments[0][0, 6] = 1e-30
    else:
        answer["repeatability"]["commands_bytes_equal"] = False
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_unqualified_qps_are_valid_scientific_failures_and_never_fabricate_calls(audit, monkeypatch):
    _, evaluate, arguments = fixture_inputs()
    Js, Jq, historical, previous, lower, upper, config = arguments
    monkeypatch.setattr(search_core, "solve_epigraph_qp", failed_certificate)
    answer = search_core.search(evaluate, historical, Js, Jq, previous, lower, upper, config)
    checked = audit.audit_search(answer, *arguments)
    assert checked["response_evaluations"] == 2 and checked["total_qp_iterations"] == 0
    assert all(not stage["qp"]["qualified"] for stage in checked["stage_checks"])
    assert checked["final_anchor_index"] == 0 and checked["selected_response_index"] is None


def test_repeat_budget_is_reserved_even_when_no_search_response_fits(audit, monkeypatch):
    _, evaluate, arguments = fixture_inputs()
    Js, Jq, historical, previous, lower, upper, config = arguments
    config["max_responses"] = 2

    def forbidden(*args, **kwargs):
        raise AssertionError("There is no budget for a local search QP")

    monkeypatch.setattr(search_core, "solve_epigraph_qp", forbidden)
    answer = search_core.search(evaluate, historical, Js, Jq, previous, lower, upper, config)
    checked = audit.audit_search(answer, *arguments)
    assert checked["termination_reason"] == "response_budget_exhausted"
    assert checked["total_stages"] == 0 and checked["response_evaluations"] == 2


def test_initial_history_mismatch_stops_before_any_qp_and_keeps_exact_repeat_report(audit, monkeypatch):
    _, evaluate, arguments = fixture_inputs()
    Js, Jq, historical, previous, lower, upper, config = arguments

    def different(command):
        entry = evaluate(command)
        entry["rollout"]["periods"][0]["requested_torque"][0, 0] = 1e-12
        return entry

    def forbidden(*args, **kwargs):
        raise AssertionError("History mismatch must prevent algebraic search")

    monkeypatch.setattr(search_core, "solve_epigraph_qp", forbidden)
    answer = search_core.search(different, historical, Js, Jq, previous, lower, upper, config)
    checked = audit.audit_search(answer, *arguments)
    assert not checked["initial_exact"] and checked["repeat_exact"]
    assert checked["termination_reason"] == "historical_initial_mismatch"


def test_nonexact_repeat_cannot_keep_a_provisional_witness_or_other_termination(audit, archive):
    answer, arguments = archive
    repeated = answer["evaluated_candidates"][-1]
    repeated["rollout"]["periods"][0]["clipped"][0, 0] = True
    answer["repeatability"] = search_core.canonical_initial_comparison(answer["evaluated_candidates"][0], repeated)
    answer["termination_reason"] = "repeat_mismatch"
    checked = audit.audit_search(answer, *arguments)
    assert not checked["repeat_exact"] and checked["qualified_witness_indices"] == []


def test_near_complete_pose_can_cross_original_caps_below_absolute_progress_floor(audit, monkeypatch):
    case = synthetic_case(base=2e-4, full_witness=True)
    excess = 5e-5

    def evaluate(command):
        rollout = case["predict"](command)
        rollout["pose"][11, 0, 3] = 0.01 * np.sqrt(1.0 + excess - command[11, 0])
        rollout["periods"][11]["endpoint_pose"] = rollout["pose"][11].copy()
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
    config = deepcopy(search_core.DEFAULT_CONFIG)
    config["max_stages"] = 1
    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))
    Jq[22, 66] = -1.0

    def exact_pose_optimum(B, g, A, b, initial_t, **kwargs):
        del initial_t, kwargs
        point, dual = np.zeros(73), np.zeros(len(A))
        row = 4369 + 24 + 22
        point[66] = -b[row] / A[row, 66]
        dual[row] = point[66] / A[row, 66]
        dual[-1] = 1.0 - dual[row]
        return certificate_from_point(B, g, A, b, point, dual)

    monkeypatch.setattr(search_core, "solve_epigraph_qp", exact_pose_optimum)
    answer = search_core.search(
        evaluate, initial, Js, Jq, case["previous"], case["interface"].input_min, case["interface"].input_max, config
    )
    checked = audit.audit_search(
        answer, Js, Jq, initial, case["previous"], case["interface"].input_min, case["interface"].input_max, config
    )
    assert checked["termination_reason"] == "qualified_witness"
    assert checked["selected_response_index"] == 1
    assert answer["stages"][0]["attempts"][0]["predicted_reduction"] < config["minimum_predicted_reduction"]
    assert answer["stages"][0]["attempts"][0]["acceptance_reason"] == "original_witness"


def test_initial_full_witness_uses_only_initial_and_forced_repeat(audit, monkeypatch):
    case = synthetic_case(full_witness=True)
    config = deepcopy(search_core.DEFAULT_CONFIG)
    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))

    def evaluate(command):
        return evaluate_repair_entry(
            case["predict"](command),
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

    def forbidden(*args, **kwargs):
        raise AssertionError("No local QP after an original complete witness")

    monkeypatch.setattr(search_core, "solve_epigraph_qp", forbidden)
    answer = search_core.search(
        evaluate,
        case["historical_anchor"],
        Js,
        Jq,
        case["previous"],
        case["interface"].input_min,
        case["interface"].input_max,
        config,
    )
    checked = audit.audit_search(
        answer,
        Js,
        Jq,
        case["historical_anchor"],
        case["previous"],
        case["interface"].input_min,
        case["interface"].input_max,
        config,
    )
    assert checked["termination_reason"] == "initial_witness"
    assert checked["selected_response_index"] == 0
    assert checked["response_evaluations"] == 2 and checked["total_stages"] == 0


def test_rowwise_causal_secants_skip_tiny_prefixes_and_bound_corrections(audit):
    config = search_core.DEFAULT_CONFIG
    periods = np.array([0, 1, 11])
    J, step, observed = np.zeros((3, 72)), np.zeros(72), np.array([1.0, 2.0, 0.0])
    step[6] = 1e-3
    result, metadata = audit._causal_update(J, step, observed, periods, config)
    assert metadata["denominator"][0] == 0 and not metadata["updated"][0]
    assert metadata["theta"][0] == 0
    assert metadata["theta"][1] < 1 and metadata["updated"][1]
    assert metadata["theta"][2] == 1 and not metadata["updated"][2]
    assert np.linalg.norm(result[1]) <= 2e-6
    assert np.all(result[0] == 0) and np.all(result[1, 12:] == 0)


def test_pure_audit_imports_neither_search_nor_model_or_optimizer(audit):
    tree = ast.parse(Path(audit.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "s2a.interior_trajectory_search" not in imports
    assert not any("scipy" in name or "mujoco" in name for name in imports)
