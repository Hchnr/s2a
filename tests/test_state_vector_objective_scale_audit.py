"""Synthetic whole-objective metadata and original-unit hand certificates."""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_interior_trajectory_search_audit import certificate_from_point, failed_certificate
from test_state_vector_gn_search_audit import hand_certificate, restoration_inputs, vector_inputs

import s2a.state_vector_gn_search as frozen_core

POLICY = "pose_whole_objective_divisor"
PROVENANCE = "original_nnls_certificate_divided_by_objective_divisor"


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("state_vector_objective_scale_audit")


def decorate(qp, divisor):
    qp["solver_objective_divisor"] = float(divisor)
    qp["solver_objective_value"] = None if qp["objective"] is None else qp["objective"] / divisor
    mu = qp["scaled_multipliers"]
    qp["objective_scaled_multipliers"] = None if mu is None else mu / divisor
    qp["objective_scaled_multipliers_provenance"] = PROVENANCE
    return qp


def archive_run(monkeypatch, evaluate, arguments, *, solver=hand_certificate, nonzero_t=False):
    """Use frozen GN control flow and analytic certificates, never a solver."""

    def fake(B, g, A, b, initial_t, **kwargs):
        qp = solver(B, g, A, b, initial_t, **kwargs)
        if nonzero_t and qp["point"] is not None and len(A) == 4682:
            point = qp["point"].copy()
            point[-1] = 2e-7  # Original complementarity remains below1e-6.
            qp = certificate_from_point(B, g, A, b, point, qp["multipliers"])
        # In these fixtures ||g|| dominates pose radius*||B||inf at both anchors.
        divisor = 1.0 if len(A) > 6000 else max(1.0, float(np.abs(g).max()))
        return decorate(qp, divisor)

    monkeypatch.setattr(frozen_core, "solve_epigraph_qp", fake)
    Js, Jq, Je, initial, previous, lower, upper, config = arguments
    answer = frozen_core.search(evaluate, initial, Js, Jq, Je, previous, lower, upper, config)
    answer["qp_objective_scaling_policy"] = POLICY
    return answer


@pytest.fixture
def archive(monkeypatch):
    evaluate, arguments, calls = vector_inputs(curvature=20.0)
    return archive_run(monkeypatch, evaluate, arguments), arguments, calls


def test_wrapper_retains_every_frozen_original_gate_and_binds_each_scale(audit, archive):
    answer, arguments, calls = archive
    expected = audit.audit_gn_search(answer, *arguments)
    checked = audit.audit_search(answer, *arguments)
    assert {key: checked[key] for key in expected} == expected
    assert checked["accepted_response_indices"] == [1, 2] and len(calls) == 4
    scales = checked["objective_scaling_checks"]
    assert scales[0]["divisor"] == 40.0 and scales[1]["divisor"] != 40.0
    assert all(row["mode"] == "pose" and row["original_certificate_qualified"] for row in scales)
    assert checked["qp_objective_scaling_policy"] == POLICY
    assert not checked["solver_callback_execution_verified_here"] and not checked["control_qualified"]


def test_independent_divisor_comparison_preserves_original_roundoff_tolerance(audit, archive):
    answer, arguments, _ = deepcopy(archive)
    qp = answer["stages"][0]["qp"]
    decorate(qp, np.nextafter(qp["solver_objective_divisor"], np.inf))
    checked = audit.audit_search(answer, *arguments)
    row = checked["objective_scaling_checks"][0]
    assert row["divisor"] > row["independent_divisor_estimate"] == 40.0
    np.testing.assert_array_equal(qp["objective_scaled_multipliers"], qp["scaled_multipliers"] / row["divisor"])


def test_curvature_divisor_uses_absolute_matrix_row_sums_including_off_diagonal(audit, monkeypatch):
    evaluate, arguments, _ = vector_inputs(stages=1, tail=0.0, slope=0.0)
    arguments[2][66, 66:68] = 100.0

    def zero_certificate(B, g, A, b, initial_t, **kwargs):
        del initial_t, kwargs
        np.testing.assert_array_equal(g, np.zeros(72))
        assert B[66, 66] == 10001.0 and B[66, 67] == 10000.0
        point, dual = np.zeros(73), np.zeros(len(A))
        dual[-1] = 1.0
        return decorate(certificate_from_point(B, g, A, b, point, dual), 0.005 * 20001.0)

    monkeypatch.setattr(frozen_core, "solve_epigraph_qp", zero_certificate)
    Js, Jq, Je, initial, previous, lower, upper, config = arguments
    answer = frozen_core.search(evaluate, initial, Js, Jq, Je, previous, lower, upper, config)
    answer["qp_objective_scaling_policy"] = POLICY
    checked = audit.audit_search(answer, *arguments)
    assert checked["objective_scaling_checks"][0]["divisor"] == 100.005
    qp = answer["stages"][0]["qp"]
    qp["solver_objective_divisor"] = 0.005 * np.abs(qp["B"]).max()
    with pytest.raises(ValueError, match="divisor"):
        audit.audit_search(answer, *arguments)


def test_restore_is_unscaled_and_original_certificate_fields_are_unchanged(audit, monkeypatch):
    evaluate, arguments = restoration_inputs()
    answer = archive_run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    assert checked["objective_scaling_checks"][0]["mode"] == "restore"
    assert checked["objective_scaling_checks"][0]["divisor"] == 1.0
    qp = answer["stages"][0]["qp"]
    assert qp["objective"] == qp["solver_objective_value"]
    np.testing.assert_array_equal(qp["scaled_multipliers"], qp["objective_scaled_multipliers"])
    decorate(qp, 2.0)
    with pytest.raises(ValueError, match="divisor"):
        audit.audit_search(answer, *arguments)


def test_nonzero_t_binds_the_entire_scaled_value_not_just_B_and_g(audit, monkeypatch):
    evaluate, arguments, _ = vector_inputs(stages=1)
    answer = archive_run(monkeypatch, evaluate, arguments, nonzero_t=True)
    checked = audit.audit_search(answer, *arguments)
    assert checked["accepted_response_indices"] == [1]
    qp = answer["stages"][0]["qp"]
    assert qp["point"][-1] == 2e-7
    qp["solver_objective_value"] += qp["point"][-1] * (1.0 - 1.0 / qp["solver_objective_divisor"])
    with pytest.raises(ValueError, match="including t"):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize(
    "defect", ["divisor", "nonpositive", "wrong_type", "value", "dual", "provenance", "stale_C", "policy", "missing"]
)
def test_scaling_metadata_cannot_be_fabricated_or_bound_to_the_wrong_stage(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    qp = answer["stages"][0]["qp"]
    if defect == "divisor":
        decorate(qp, 2.0)
    elif defect == "nonpositive":
        qp["solver_objective_divisor"] = 0.0
    elif defect == "wrong_type":
        qp["solver_objective_divisor"] = 40
    elif defect == "value":
        qp["solver_objective_value"] += 1e-4
    elif defect == "dual":
        qp["objective_scaled_multipliers"][-1] += 1e-9
    elif defect == "provenance":
        qp["objective_scaled_multipliers_provenance"] = "returned_by_SLSQP"
    elif defect == "stale_C":
        decorate(answer["stages"][1]["qp"], qp["solver_objective_divisor"])
    elif defect == "policy":
        answer["qp_objective_scaling_policy"] = "scale_B_and_g_only"
    else:
        del qp["objective_scaled_multipliers"]
    with pytest.raises((ValueError, KeyError)):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("field", ["objective", "multipliers", "scaled_multipliers", "B"])
def test_scaled_units_cannot_replace_original_KKT_or_hessian_fields(audit, archive, field):
    answer, arguments, _ = deepcopy(archive)
    qp = answer["stages"][0]["qp"]
    qp[field] /= qp["solver_objective_divisor"]
    if field == "scaled_multipliers":
        qp["objective_scaled_multipliers"] = qp[field] / qp["solver_objective_divisor"]
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_solver_failure_remains_unqualified_even_when_saved_algebraic_KKT_passes(audit, monkeypatch):
    evaluate, arguments, calls = vector_inputs(stages=1)

    def unsuccessful(B, g, A, b, initial_t, **kwargs):
        qp = hand_certificate(B, g, A, b, initial_t, **kwargs)
        assert qp["stationarity_relative"] < 1e-6 and qp["complementarity_relative"] < 1e-6
        qp.update(
            solver_success=False, solver_status=8, qualified=False, qualification_failures=["solver_unsuccessful"]
        )
        return qp

    answer = archive_run(monkeypatch, evaluate, arguments, solver=unsuccessful)
    checked = audit.audit_search(answer, *arguments)
    assert len(calls) == checked["response_evaluations"] == 2
    assert checked["accepted_response_indices"] == [] and not answer["stages"][0]["attempts"]
    assert not checked["objective_scaling_checks"][0]["original_certificate_qualified"]
    answer["stages"][0]["qp"]["qualified"] = True
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("defect", [None, "value", "dual"])
def test_incomplete_certificate_requires_absent_value_and_dual(audit, monkeypatch, defect):
    evaluate, arguments, calls = vector_inputs(stages=1)
    answer = archive_run(monkeypatch, evaluate, arguments, solver=failed_certificate)
    checked = audit.audit_search(answer, *arguments)
    assert checked["response_evaluations"] == len(calls) == 2
    qp = answer["stages"][0]["qp"]
    assert qp["solver_objective_divisor"] == 40.0
    assert qp["solver_objective_value"] is qp["objective_scaled_multipliers"] is None
    if defect is None:
        return
    qp["solver_objective_value" if defect == "value" else "objective_scaled_multipliers"] = 0.0
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_rejected_response_updates_and_factual_anchor_are_still_audited(audit, monkeypatch):
    evaluate, arguments = restoration_inputs(feedback=True)
    answer = archive_run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    assert checked["rejected_response_indices"] == [5] and checked["accepted_response_indices"] == [10]
    assert all(row["divisor"] == 1.0 for row in checked["objective_scaling_checks"])
    answer["stages"][0]["rejected_update"]["error_vector"]["denominator"][0] = 1e-30
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("defect", ["repeat", "unsafe", "counter", "fullcaps", "stage_budget"])
def test_metadata_does_not_weaken_original_safety_repeat_selection_or_budgets(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    if defect == "repeat":
        answer["repeatability"]["commands_bytes_equal"] = False
    elif defect == "unsafe":
        answer["evaluated_candidates"][1]["safety_slack"][0] = 0.0
    elif defect == "counter":
        answer["prediction_calls"] += 1
    elif defect == "fullcaps":
        answer["selected_response_index"] = 1
    else:
        arguments[-1]["max_stages"] = 1
        answer["config"] = deepcopy(arguments[-1])
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_independent_initializer_is_reexported_and_no_new_production_solver_is_imported(audit):
    original = importlib.import_module("state_vector_gn_search_audit")
    assert audit.reconstruct_error_vector_jacobian is original.reconstruct_error_vector_jacobian
    imports = []
    for node in ast.walk(ast.parse(Path(audit.__file__).read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("s2a." in name or "scipy" in name or "mujoco" in name for name in imports)
    assert "state_vector_gn_search_audit" in imports
