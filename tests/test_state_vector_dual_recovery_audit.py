"""Manual dual certificates and fake ledgers; no optimizer or NNLS is called."""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_interior_trajectory_search_audit import certificate_from_point, failed_certificate
from test_state_vector_gn_search_audit import hand_certificate, restoration_inputs, vector_inputs
from test_state_vector_objective_scale_audit import decorate

import s2a.state_vector_gn_search as frozen_core

POLICY = "pose_slack_restricted_nnls_restore_full_active"
POSE_PROVENANCE = "unscaled_gradient_subset_nnls_certificate_divided_by_objective_divisor"
RESTORE_PROVENANCE = "original_nnls_certificate_divided_by_objective_divisor"


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("state_vector_dual_recovery_audit")


def annotate(qp, A, b, mode, divisor):
    """Declare algebraic fixture metadata; deliberately never repair its dual."""
    decorate(qp, divisor)
    qp["objective_scaled_multipliers_provenance"] = POSE_PROVENANCE if mode == "pose" else RESTORE_PROVENANCE
    qp["dual_recovery_mode"] = mode
    qp["full_active_threshold"] = 10.0 * qp["certificate_tolerance"]
    qp["recovery_threshold"] = (0.01 if mode == "pose" else 10.0) * qp["certificate_tolerance"]
    complete = qp["point"] is not None and qp["objective"] is not None
    if complete:
        scale = np.maximum.reduce((np.ones(len(A)), np.max(np.abs(A), axis=1), np.abs(b)))
        slack = (A / scale[:, None]) @ qp["point"] + b / scale
        full = np.flatnonzero(slack <= qp["full_active_threshold"]).tolist()
        recovery = [index for index in full if slack[index] <= qp["recovery_threshold"]]
        qp["active_indices"] = full
        qp["nnls_max_iterations"] = 3 * max(len(full), 1)
    else:
        recovery = []
    qp["recovery_indices"] = recovery
    qp["nnls_fit_attempts"] = int(complete and bool(recovery))
    return qp


def archive_run(monkeypatch, evaluate, arguments, *, solver=hand_certificate):
    """Frozen control flow with explicit hand certificates, not a new solver."""

    def fake(B, g, A, b, initial_t, **kwargs):
        qp = solver(B, g, A, b, initial_t, **kwargs)
        mode = "restore" if len(A) > 6000 else "pose"
        # Fixture pose gradients dominate radius*||B||inf at each actual anchor.
        divisor = 1.0 if mode == "restore" else max(1.0, float(np.abs(g).max()))
        return annotate(qp, A, b, mode, divisor)

    monkeypatch.setattr(frozen_core, "solve_epigraph_qp", fake)
    Js, Jq, Je, initial, previous, lower, upper, config = arguments
    answer = frozen_core.search(evaluate, initial, Js, Jq, Je, previous, lower, upper, config)
    answer["qp_objective_scaling_policy"] = "pose_whole_objective_divisor"
    answer["dual_recovery_policy"] = POLICY
    return answer


@pytest.fixture
def archive(monkeypatch):
    evaluate, arguments, calls = vector_inputs(curvature=20.0)
    return archive_run(monkeypatch, evaluate, arguments), arguments, calls


def classify(qp):
    """Apply original thresholds to completed arithmetic, including failures."""
    failures = [] if qp["solver_success"] else ["solver_unsuccessful"]
    tolerance = qp["certificate_tolerance"]
    if qp["primal_min"] < -tolerance:
        failures.append("primal_infeasible")
    if np.any(qp["scaled_multipliers"] < 0) or np.any(qp["multipliers"] < 0):
        failures.append("negative_multiplier")
    if qp["stationarity_relative"] > tolerance:
        failures.append("stationarity_failed")
    if qp["complementarity_relative"] > tolerance:
        failures.append("complementarity_failed")
    qp.update(qualification_failures=failures, qualified=not failures)
    return qp


def test_full_gn_ledger_is_retained_and_each_subset_is_bound_to_actual_returned_point(audit, archive):
    answer, arguments, calls = archive
    expected = audit.audit_gn_search(answer, *arguments)
    checked = audit.audit_search(answer, *arguments)
    assert {key: checked[key] for key in expected} == expected
    assert checked["accepted_response_indices"] == [1, 2] and len(calls) == 4
    assert checked["total_nnls_fit_attempts"] == checked["total_successful_nnls_fits"] == 2
    assert checked["total_failed_nnls_fits"] == 0
    assert all(row["mode"] == "pose" for row in checked["dual_recovery_checks"])
    # The optimum lies on a command upper bound, unlike the zero initial point.
    assert any(index != 4681 for index in checked["dual_recovery_checks"][0]["recovery_indices"])
    assert checked["objective_scaling_checks"][1]["divisor"] != 40.0
    assert answer["stages"][0]["anchor_metrics_before"]["pose_excess"] == 8.0
    assert answer["stages"][0]["anchor_metrics_after"]["pose_excess"] == 8.0
    assert not checked["solver_callback_execution_verified_here"]
    assert not checked["nnls_callback_execution_verified_here"] and not checked["control_qualified"]


def test_restore_keeps_full_set_budget_and_original_certificate_exact(audit, monkeypatch):
    evaluate, arguments = restoration_inputs()
    originals = []

    def capture(B, g, A, b, initial_t, **kwargs):
        qp = hand_certificate(B, g, A, b, initial_t, **kwargs)
        originals.append(deepcopy(qp))
        return qp

    answer = archive_run(monkeypatch, evaluate, arguments, solver=capture)
    checked = audit.audit_search(answer, *arguments)
    row = checked["dual_recovery_checks"][0]
    assert row["mode"] == "restore" and row["recovery_indices"] == row["full_active_indices"]
    assert checked["objective_scaling_checks"][0]["divisor"] == 1.0
    qp = answer["stages"][0]["qp"]
    for key, original in originals[0].items():
        np.testing.assert_equal(qp[key], original)
    assert qp["objective_scaled_multipliers_provenance"] == RESTORE_PROVENANCE
    qp["recovery_indices"] = qp["recovery_indices"][:-1]
    with pytest.raises(ValueError, match="indices"):
        audit.audit_search(answer, *arguments)


def test_rejected_feedback_and_following_three_models_remain_chronological(audit, monkeypatch):
    evaluate, arguments = restoration_inputs(feedback=True)
    answer = archive_run(monkeypatch, evaluate, arguments)
    checked = audit.audit_search(answer, *arguments)
    assert checked["rejected_response_indices"] == [5] and checked["accepted_response_indices"] == [10]
    assert answer["stages"][0]["anchor_before_index"] == answer["stages"][0]["anchor_after_index"] == 0
    assert all(row["mode"] == "restore" for row in checked["dual_recovery_checks"])
    answer["stages"][0]["rejected_update"]["error_vector"]["denominator"][0] = 1e-30
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize(
    "defect",
    [
        "mode",
        "full_threshold",
        "recovery_threshold",
        "full_indices",
        "indices",
        "bool_indices",
        "fit_bool",
        "fit_zero",
        "fit_two",
        "provenance",
        "policy",
        "missing",
    ],
)
def test_declared_subsets_modes_and_one_fit_accounting_cannot_be_fabricated(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    qp = answer["stages"][0]["qp"]
    if defect == "mode":
        qp["dual_recovery_mode"] = "restore"
    elif defect == "full_threshold":
        qp["full_active_threshold"] = 1e-5  # Differs from the original expression by one ULP.
    elif defect == "recovery_threshold":
        qp["recovery_threshold"] *= 10
    elif defect == "full_indices":
        qp["active_indices"] = []
    elif defect == "indices":
        qp["recovery_indices"] = []
    elif defect == "bool_indices":
        qp["recovery_indices"][0] = True
    elif defect.startswith("fit_"):
        qp["nnls_fit_attempts"] = dict(fit_bool=True, fit_zero=0, fit_two=2)[defect]
    elif defect == "provenance":
        qp["objective_scaled_multipliers_provenance"] = RESTORE_PROVENANCE
    elif defect == "policy":
        answer["dual_recovery_policy"] = "retry_until_certified"
    else:
        del qp["recovery_indices"]
    with pytest.raises((ValueError, KeyError)):
        audit.audit_search(answer, *arguments)


def test_original_unit_fields_and_scale_metadata_remain_separate(audit, archive):
    answer, arguments, _ = deepcopy(archive)
    qp = answer["stages"][0]["qp"]
    divisor = np.nextafter(qp["solver_objective_divisor"], np.inf)
    qp["solver_objective_divisor"] = float(divisor)
    qp["solver_objective_value"] = qp["objective"] / divisor
    qp["objective_scaled_multipliers"] = qp["scaled_multipliers"] / divisor
    checked = audit.audit_search(answer, *arguments)
    assert checked["objective_scaling_checks"][0]["divisor"] != 40.0
    qp["scaled_multipliers"] = qp["objective_scaled_multipliers"].copy()
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_solver_failure_cannot_be_overruled_by_a_successful_tight_fit(audit, monkeypatch):
    evaluate, arguments, calls = vector_inputs(stages=1)

    def unsuccessful(B, g, A, b, initial_t, **kwargs):
        qp = hand_certificate(B, g, A, b, initial_t, **kwargs)
        qp.update(solver_success=False, solver_status=8)
        return classify(qp)

    answer = archive_run(monkeypatch, evaluate, arguments, solver=unsuccessful)
    checked = audit.audit_search(answer, *arguments)
    assert len(calls) == checked["response_evaluations"] == 2
    assert checked["total_successful_nnls_fits"] == 1 and checked["accepted_response_indices"] == []
    assert not checked["dual_recovery_checks"][0]["original_certificate_qualified"]


@pytest.mark.parametrize("kind", ["solver_exception", "invalid_point", "nnls_exception"])
def test_incomplete_certificates_preserve_zero_or_single_attempt_and_never_create_response(audit, monkeypatch, kind):
    evaluate, arguments, calls = vector_inputs(stages=1)

    def partial(B, g, A, b, initial_t, **kwargs):
        if kind != "nnls_exception":
            qp = failed_certificate(B, g, A, b, initial_t, **kwargs)
            if kind == "invalid_point":
                qp.update(
                    solver_success=True,
                    solver_status=0,
                    solver_exception=None,
                    solver_nit=1,
                    qualification_failures=["invalid_solver_point"],
                )
            return qp
        qp = hand_certificate(B, g, A, b, initial_t, **kwargs)
        for key in (
            "multipliers",
            "scaled_multipliers",
            "nnls_residual_norm",
            "stationarity_relative",
            "stationarity_residual_inf",
            "complementarity_relative",
            "complementarity_residual_inf",
            "complementarity_sum_gap",
        ):
            qp[key] = None
        qp.update(
            nnls_success=False,
            nnls_error="Synthetic NNLS exception; no call",
            qualified=False,
            qualification_failures=["nnls_failed"],
        )
        return qp

    answer = archive_run(monkeypatch, evaluate, arguments, solver=partial)
    checked = audit.audit_search(answer, *arguments)
    expected = int(kind == "nnls_exception")
    assert checked["total_nnls_fit_attempts"] == checked["total_failed_nnls_fits"] == expected
    assert checked["total_successful_nnls_fits"] == 0
    assert checked["response_evaluations"] == len(calls) == 2
    qp = answer["stages"][0]["qp"]
    assert qp["objective_scaled_multipliers"] is None
    qp["nnls_fit_attempts"] = 1 - expected
    with pytest.raises(ValueError, match="NNLS call"):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("kind", ["empty", "insufficient", "primal"])
def test_empty_or_insufficient_tight_subset_and_primal_failure_still_block_responses(audit, monkeypatch, kind):
    evaluate, arguments, calls = vector_inputs(stages=1)

    def deficient(B, g, A, b, initial_t, **kwargs):
        del initial_t, kwargs
        point, dual = np.zeros(73), np.zeros(len(A))
        if kind == "empty":
            point[-1] = 2e-6
        else:
            dual[-1] = 1.0
        if kind == "primal":
            point[66] = 0.0051
        return classify(certificate_from_point(B, g, A, b, point, dual))

    answer = archive_run(monkeypatch, evaluate, arguments, solver=deficient)
    checked = audit.audit_search(answer, *arguments)
    assert checked["response_evaluations"] == len(calls) == 2
    assert checked["accepted_response_indices"] == []
    row = checked["dual_recovery_checks"][0]
    if kind == "empty":
        assert row["full_active_count"] > row["recovery_count"] == 0
        assert row["nnls_fit_attempts"] == row["successful_nnls_fits"] == 0
        assert answer["stages"][0]["qp"]["nnls_success"] is True
        answer["stages"][0]["qp"]["nnls_max_iterations"] = 0
        with pytest.raises(ValueError, match="iteration"):
            audit.audit_search(answer, *arguments)
    else:
        assert row["successful_nnls_fits"] == 1 and not row["original_certificate_qualified"]
        if kind == "primal":
            assert "primal_infeasible" in answer["stages"][0]["qp"]["qualification_failures"]


def test_positive_slack_columns_can_preserve_stationarity_while_breaking_original_complementarity(audit):
    # Hand calculation: two identical x normals have different offsets.
    # mu=(1,0,1) is stationary but not complementary; (0,1,1) is both.
    B, g, A, b = np.eye(72), np.zeros(72), np.zeros((3, 73)), np.array([2e-6, 0.0, 0.0])
    g[0], A[0, 0], A[1, 0], A[2, -1] = 1.0, 1.0, 1.0, 1.0
    original = importlib.import_module("accepted_point_sqp_math_audit")
    bad = annotate(
        classify(certificate_from_point(B, g, A, b, np.zeros(73), np.array([1.0, 0.0, 1.0]))), A, b, "pose", 1.0
    )
    assert bad["stationarity_relative"] == 0 and bad["complementarity_relative"] == 2e-6
    assert not original.audit_qp_certificate(bad, B, g, A, b)["qualified"]
    with pytest.raises(ValueError, match="outside"):
        audit._recovery_record(bad, B, g, A, b, "pose", audit.numerical_compare)
    good = annotate(
        classify(certificate_from_point(B, g, A, b, np.zeros(73), np.array([0.0, 1.0, 1.0]))), A, b, "pose", 1.0
    )
    assert original.audit_qp_certificate(good, B, g, A, b)["qualified"]
    checked = audit._recovery_record(good, B, g, A, b, "pose", audit.numerical_compare)
    assert checked["recovery_indices"] == [1, 2] and checked["nnls_max_iterations"] == 9
    good["nnls_max_iterations"] = 6
    with pytest.raises(ValueError, match="full-active"):
        audit._recovery_record(good, B, g, A, b, "pose", audit.numerical_compare)


def test_even_tiny_nonzero_mu_outside_subset_is_rejected(audit):
    B, g, A, b = np.eye(72), np.zeros(72), np.zeros((2, 73)), np.array([2e-6, 0.0])
    A[:, -1] = 1.0
    qp = annotate(certificate_from_point(B, g, A, b, np.zeros(73), np.array([1e-30, 1.0])), A, b, "pose", 1.0)
    with pytest.raises(ValueError, match="exact zero"):
        audit._recovery_record(qp, B, g, A, b, "pose", audit.numerical_compare)


@pytest.mark.parametrize("defect", ["repeat", "safety", "count", "caps", "stage_budget"])
def test_new_recovery_does_not_weaken_original_full_contract(audit, archive, defect):
    answer, arguments, _ = deepcopy(archive)
    if defect == "repeat":
        answer["repeatability"]["commands_bytes_equal"] = False
    elif defect == "safety":
        answer["evaluated_candidates"][1]["safety_slack"][0] = 0.0
    elif defect == "count":
        answer["prediction_calls"] += 1
    elif defect == "caps":
        answer["selected_response_index"] = 1
    else:
        arguments[-1]["max_stages"] = 1
        answer["config"] = deepcopy(arguments[-1])
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_only_frozen_independent_helpers_are_imported_and_initializer_is_reused(audit):
    gn = importlib.import_module("state_vector_gn_search_audit")
    assert audit.reconstruct_error_vector_jacobian is gn.reconstruct_error_vector_jacobian
    imports = []
    for node in ast.walk(ast.parse(Path(audit.__file__).read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
            if node.module == "state_vector_objective_scale_audit":
                assert [alias.name for alias in node.names] == ["_divisor"]
    assert not any("s2a." in name or "scipy" in name or "mujoco" in name for name in imports)
    assert "state_vector_gn_search_audit" in imports
