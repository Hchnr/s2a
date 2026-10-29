"""Actual synthetic response ledgers with hand-certified QPs, never a solver."""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_interior_trajectory_search_audit import certificate_from_point
from test_tail_translation_reserve import synthetic_case

import s2a.global_envelope_search as original_core
import s2a.interior_trajectory_search as original_config
import s2a.stage_budget_search as core
from s2a.constraint_residual_repair import evaluate_repair_entry
from s2a.directional_derivative_probe import _canonical


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("stage_budget_search_audit")


def linear_inputs(stages=25, responses=122):
    """Every stage makes actual admitted progress without reaching a radius/box stop."""
    case = synthetic_case(base=2e-4, full_witness=True)
    config = deepcopy(original_config.DEFAULT_CONFIG)
    config.update(max_stages=stages, max_responses=responses)
    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))
    Jq[22] = -0.0013
    calls = []

    def evaluate(command):
        calls.append(command.tobytes())
        rollout = case["predict"](command)
        squared = 4.0 - 0.0013 * float(np.sum(command))
        rollout["pose"][11, 0, 3] = 0.01 * np.sqrt(squared)
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

    historical = evaluate(case["anchor"])
    calls.clear()
    arguments = (Js, Jq, historical, case["previous"], case["interface"].input_min, case["interface"].input_max, config)
    return evaluate, arguments, calls


def analytic_certificate(B, g, A, b, initial_t, **kwargs):
    """The sole active pose row has dual1 and the exact interior optimum d=-Jq22."""
    del initial_t, kwargs
    row = 4369 + 24 + 22
    point, dual = np.zeros(73), np.zeros(len(A))
    point[:72] = A[row, :72]
    point[-1] = -b[row] - A[row, :72] @ point[:72]
    dual[row] = 1.0
    return certificate_from_point(B, g, A, b, point, dual)


def run(monkeypatch, *, stages=25, responses=122, module=core):
    evaluate, arguments, calls = linear_inputs(stages, responses)
    monkeypatch.setattr(module, "solve_epigraph_qp", analytic_certificate)
    Js, Jq, historical, previous, lower, upper, config = arguments
    answer = module.search(evaluate, historical, Js, Jq, previous, lower, upper, config)
    return answer, arguments, calls


@pytest.fixture
def archive(monkeypatch):
    return run(monkeypatch)


def test_old24_stage_answer_receives_the_exact_frozen_audit_result(audit, monkeypatch):
    answer, arguments, calls = run(monkeypatch, stages=24, module=original_core)
    expected = audit.audit_global_search(answer, *arguments)
    observed = audit.audit_search(answer, *arguments)
    assert _canonical(observed) == _canonical(expected)
    assert len(calls) == observed["response_evaluations"] == 26
    assert observed["total_stages"] == 24 and observed["repeat_exact"]


@pytest.mark.parametrize("stages", [25, 60])
def test_extended_budget_has_real_charged_synthetic_responses_at_every_stage(audit, monkeypatch, stages):
    answer, arguments, calls = run(monkeypatch, stages=stages)
    checked = audit.audit_search(answer, *arguments)
    assert checked["total_stages"] == len(answer["stages"]) == stages
    assert checked["total_qp_iterations"] == stages
    assert checked["response_evaluations"] == len(calls) == stages + 2
    assert checked["accepted_response_indices"] == list(range(1, stages + 1))
    assert all(stage["accepted_response_index"] is not None for stage in answer["stages"])
    assert all(stage["radius_before"] == stage["radius_after"] == 0.005 for stage in answer["stages"])
    assert answer["request_ledger"][0]["role"] == "initial"
    assert answer["request_ledger"][-1]["role"] == "initial_repeat"
    assert calls[0] == calls[-1] and len(set(calls)) == stages + 1
    assert checked["termination_reason"] == "stage_budget_exhausted"
    assert checked["selected_response_index"] is None and not checked["control_qualified"]


def test_122_response_cap_is_not_reinterpreted_as_a_new_per_stage_allowance(audit, monkeypatch):
    answer, arguments, calls = run(monkeypatch, stages=60, responses=7)
    checked = audit.audit_search(answer, *arguments)
    assert len(calls) == checked["response_evaluations"] == 7
    assert checked["total_stages"] == 5 and checked["accepted_response_indices"] == [1, 2, 3, 4, 5]
    assert checked["termination_reason"] == "response_budget_exhausted"
    assert answer["request_ledger"][-1]["role"] == "initial_repeat"


def test_no_trial_budget_still_reserves_both_initial_and_repeat(audit, monkeypatch):
    evaluate, arguments, calls = linear_inputs(stages=60, responses=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("No QP is permitted when only initial/repeat responses fit")

    monkeypatch.setattr(core, "solve_epigraph_qp", forbidden)
    Js, Jq, historical, previous, lower, upper, config = arguments
    answer = core.search(evaluate, historical, Js, Jq, previous, lower, upper, config)
    checked = audit.audit_search(answer, *arguments)
    assert checked["total_stages"] == 0 and checked["response_evaluations"] == len(calls) == 2
    assert checked["termination_reason"] == "response_budget_exhausted" and checked["repeat_exact"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_stages", 0),
        ("max_stages", 61),
        ("max_stages", True),
        ("max_stages", 24.0),
        ("max_responses", 1),
        ("max_responses", 123),
        ("max_responses", False),
        ("max_responses", 122.0),
        ("seed", 43),
        ("seed", 42.0),
        ("safety_buffer", 1e-7),
        ("pose_ceiling_fraction", 0.02),
        ("alphas", [1.0, 0.5]),
        ("secant_relative_limit", 2),
    ],
)
def test_independent_complete_configuration_rejects_changes_before_reading_a_ledger(audit, key, value):
    config = deepcopy(original_config.DEFAULT_CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        audit.audit_search(None, None, None, None, None, None, None, config)


@pytest.mark.parametrize("defect", ["missing", "unknown"])
def test_complete_configuration_cannot_omit_or_add_contract_fields(audit, defect):
    config = deepcopy(original_config.DEFAULT_CONFIG)
    if defect == "missing":
        del config["reserve_entry_fraction"]
    else:
        config["extra_budget"] = 1000
    with pytest.raises(ValueError):
        audit.audit_search(None, None, None, None, None, None, None, config)


@pytest.mark.parametrize("kind", ["stage", "response"])
def test_full_real_ledger_cannot_exceed_its_own_registered_budget(audit, archive, kind):
    answer, arguments, _ = deepcopy(archive)
    config = arguments[-1]
    if kind == "stage":
        config["max_stages"] = len(answer["stages"]) - 1
    else:
        config["max_responses"] = len(answer["evaluated_candidates"]) - 1
    answer["config"] = deepcopy(config)
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("defect", ["certificate", "safety", "repeat", "selected", "future", "counter"])
def test_extended_stage_allowance_preserves_original_full_audit_invariants(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    if defect == "certificate":
        answer["stages"][0]["qp"]["certificate_tolerance"] = 1e-5
    elif defect == "safety":
        answer["evaluated_candidates"][1]["safety_slack"][0] = np.nextafter(1e-8, -np.inf)
    elif defect == "repeat":
        answer["repeatability"]["commands_bytes_equal"] = False
    elif defect == "selected":
        answer["selected_response_index"] = 1
        answer["selected"] = deepcopy(answer["evaluated_candidates"][1])
    elif defect == "future":
        arguments[0][0, 6] = 1e-30
    else:
        answer["prediction_calls"] -= 1
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_frozen_contract_and_no_solver_import_are_explicit(audit):
    assert audit._FIXED_CONFIG == original_config.DEFAULT_CONFIG
    tree = ast.parse(Path(audit.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "global_envelope_search_audit" in imports
    assert not any("s2a." in name or "scipy" in name or "mujoco" in name for name in imports)
