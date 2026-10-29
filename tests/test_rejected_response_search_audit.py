"""Analytic rejected-response feedback ledgers; no optimizer or native model."""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_interior_trajectory_search_audit import (
    analytic_certificate,
    certificate_from_point,
    failed_certificate,
    fixture_inputs,
)
from test_tail_translation_reserve import synthetic_case

import s2a.interior_trajectory_search as baseline
import s2a.rejected_response_search as variant
from s2a.constraint_residual_repair import evaluate_repair_entry


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("rejected_response_search_audit")


def capped_restoration_certificate(B, g, A, b, initial_t, **kwargs):
    """Known one-axis optimum, including a later active pose-ceiling row."""
    if len(A) < 6000:
        return analytic_certificate(B, g, A, b, initial_t, **kwargs)
    soft_row, ceiling_row, axis = 4369 + 24 + 4368, 4369 + 22, 66
    point, dual = np.zeros(73), np.zeros(len(A))
    target = -b[soft_row] / A[soft_row, axis]
    upper = np.inf if A[ceiling_row, axis] >= 0 else -b[ceiling_row] / A[ceiling_row, axis]
    point[axis] = min(target, upper)
    point[-1] = max(0.0, -b[soft_row] - A[soft_row, axis] * point[axis])
    if upper < target:
        dual[soft_row] = 1.0
        dual[ceiling_row] = (A[soft_row, axis] - point[axis]) / (-A[ceiling_row, axis])
    else:
        dual[soft_row] = point[axis] / A[soft_row, axis]
        dual[-1] = 1.0 - dual[soft_row]
    return certificate_from_point(B, g, A, b, point, dual)


def nonlinear_inputs(*, unsafe=False, tiny=False):
    slope = 100.0 if tiny else 0.1
    case = synthetic_case(terminal=lambda commands: 2e-8 + (-slope if unsafe else slope) * commands[11, 0])
    config = deepcopy(baseline.DEFAULT_CONFIG)
    config["max_stages"] = 2
    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))
    Js[-1, 66], Jq[22, 66] = slope, 30.0
    curvature = 2e7 * (slope / 0.1) ** 2

    def evaluate(command):
        rollout = case["predict"](command)
        delta = command[11, 0]
        rollout["pose"][11, 0, 3] = 0.01 * np.sqrt(4.0 + 30.0 * delta + curvature * delta**2)
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
    arguments = (Js, Jq, historical, case["previous"], case["interface"].input_min, case["interface"].input_max, config)
    return evaluate, arguments


def run(monkeypatch, evaluate, arguments, *, core=variant, solver=capped_restoration_certificate):
    Js, Jq, historical, previous, lower, upper, config = arguments
    monkeypatch.setattr(core, "solve_epigraph_qp", solver)
    return core.search(evaluate, historical, Js, Jq, previous, lower, upper, config)


@pytest.fixture
def archive(monkeypatch):
    evaluate, arguments = nonlinear_inputs()
    return run(monkeypatch, evaluate, arguments), arguments, evaluate


def test_minimum_alpha_safe_rejection_updates_models_without_moving_anchor(audit, archive):
    answer, arguments, _ = archive
    checked = audit.audit_search(answer, *arguments)
    first, second = answer["stages"]
    assert checked["passed"] and checked["rejected_update_count"] == 1
    assert checked["rejected_response_indices"] == [5]
    assert first["rejected_update"]["source_response_index"] == 5
    assert first["rejected_update"]["alpha"] == 0.0625
    assert first["anchor_before_index"] == first["anchor_after_index"] == 0
    assert first["accepted_response_index"] is None and first["secant_update"] is None
    assert first["radius_after"] == first["radius_before"] / 2
    assert second["anchor_before_index"] == 0
    assert second["accepted_response_index"] == 10 and second["rejected_update"] is None
    assert checked["accepted_response_indices"] == [10]
    assert checked["selected_response_index"] is None
    assert checked["response_evaluations"] == 12 and answer["prediction_calls"] == 12
    assert not checked["control_qualified"]


@pytest.mark.parametrize(
    "defect",
    [
        "larger_alpha",
        "source_alpha",
        "source_step",
        "denominator",
        "theta",
        "updated",
        "count",
        "missing",
        "moved_anchor",
        "selected",
        "policy",
        "feedback_after_acceptance",
        "future",
    ],
)
def test_feedback_selection_and_every_update_cannot_be_forged(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    first = answer["stages"][0]
    update = first["rejected_update"]
    if defect == "larger_alpha":
        update["source_response_index"] = 1
    elif defect == "source_alpha":
        update["alpha"] = 1.0
    elif defect == "source_step":
        update["step"][11, 0] *= 2
    elif defect == "denominator":
        update["pose"]["denominator"][-1] *= 2
    elif defect == "theta":
        update["pose"]["theta"][22] *= 2
    elif defect == "updated":
        update["pose"]["updated"][22] = False
    elif defect == "count":
        answer["rejected_update_count"] += 1
    elif defect == "missing":
        first["rejected_update"] = None
    elif defect == "moved_anchor":
        first["anchor_after_index"] = 5
    elif defect == "selected":
        answer["selected_response_index"] = 5
        answer["selected"] = deepcopy(answer["evaluated_candidates"][5])
    elif defect == "policy":
        answer["feedback_policy"] = "largest_residual"
    elif defect == "feedback_after_acceptance":
        answer["stages"][1]["rejected_update"] = deepcopy(update)
    else:
        arguments[1][0, 6] = 1e-30
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_next_qp_is_certified_against_feedback_derivatives_not_the_old_model(audit, archive, monkeypatch):
    answer, arguments, evaluate = archive
    old = run(monkeypatch, evaluate, arguments, core=baseline)
    assert old["stages"][1]["qp"]["point"][66] > answer["stages"][1]["qp"]["point"][66]
    answer["stages"][1]["qp"] = old["stages"][1]["qp"]
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("kind", ["unsafe", "tiny", "no_call", "unqualified"])
def test_ineligible_rejections_or_missing_calls_never_produce_feedback(audit, archive, monkeypatch, kind):
    if kind in ("unsafe", "tiny"):
        evaluate, arguments = nonlinear_inputs(**{kind: True})
        answer = run(monkeypatch, evaluate, arguments)
    elif kind == "no_call":
        case = synthetic_case(base=2e-4)
        _, evaluate, arguments = fixture_inputs()

        # Use a pose-only constant response with a valid already-restored terminal.
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
            np.zeros((4369, 72)),
            np.zeros((24, 72)),
            case["historical_anchor"],
            case["previous"],
            case["interface"].input_min,
            case["interface"].input_max,
            arguments[-1],
        )
        answer = run(monkeypatch, evaluate, arguments)
    else:
        evaluate, arguments = nonlinear_inputs()
        answer = run(monkeypatch, evaluate, arguments, solver=failed_certificate)
    checked = audit.audit_search(answer, *arguments)
    assert checked["passed"] and checked["rejected_update_count"] == 0
    assert all(stage["rejected_update"] is None for stage in answer["stages"])
    if kind in ("no_call", "unqualified"):
        assert checked["response_evaluations"] == 2
    # A copied real feedback record cannot bypass the freshly rebuilt eligible set.
    answer["stages"][0]["rejected_update"] = deepcopy(archive[0]["stages"][0]["rejected_update"])
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_accepted_baseline_path_has_identical_commands_and_no_feedback(audit, monkeypatch):
    _, evaluate, arguments = fixture_inputs()
    old = run(monkeypatch, evaluate, arguments, core=baseline, solver=analytic_certificate)
    answer = run(monkeypatch, evaluate, arguments, solver=analytic_certificate)
    checked = audit.audit_search(answer, *arguments)
    assert checked["rejected_update_count"] == 0
    assert [row["commands"].tobytes() for row in old["evaluated_candidates"]] == [
        row["commands"].tobytes() for row in answer["evaluated_candidates"]
    ]
    assert answer["final_anchor_index"] == old["final_anchor_index"]
    assert answer["selected_response_index"] is old["selected_response_index"] is None


def test_audit_imports_only_frozen_mathematics_and_never_variant_core(audit):
    tree = ast.parse(Path(audit.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "s2a.rejected_response_search" not in imports
    assert "s2a.interior_trajectory_search" not in imports
    assert "interior_trajectory_search_audit" in imports
    assert not any("scipy" in name or "mujoco" in name for name in imports)
