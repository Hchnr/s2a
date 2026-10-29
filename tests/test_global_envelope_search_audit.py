"""Hand-certified full-response ledgers for the fixed global pose envelope.

Every QP certificate is analytic. No optimizer, native simulator, model service,
or GPU is run; the audit never imports the new search core.
"""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_interior_trajectory_search_audit import certificate_from_point, failed_certificate
from test_rejected_response_search_audit import capped_restoration_certificate, nonlinear_inputs
from test_tail_translation_reserve import synthetic_case

import s2a.global_envelope_search as core
import s2a.interior_trajectory_search as baseline
from s2a.constraint_residual_repair import evaluate_repair_entry


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("global_envelope_search_audit")


def heterogeneous_inputs():
    """The worst row can descend while another row crosses its old private cap."""
    case = synthetic_case(base=2e-4, full_witness=True)
    config = deepcopy(baseline.DEFAULT_CONFIG)
    config["max_stages"] = 1
    Js, Jq = np.zeros((4369, 72)), np.zeros((24, 72))
    Jq[22, 66], Jq[23, 66] = -100.0, 100.0

    def evaluate(command):
        rollout = case["predict"](command)
        displacement = command[11, 0]
        rollout["pose"][11, 0, 3] = 0.01 * np.sqrt(4.0 - 100.0 * displacement)
        rollout["pose"][11, :3, :3] = Rotation.from_rotvec(
            [0.0, 0.0, 0.05 * np.sqrt(1.0 + 100.0 * displacement)]
        ).as_matrix()
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


def heterogeneous_certificate(B, g, A, b, initial_t, **kwargs):
    """Known optimum: d66=.005 at its trust bound, t=3-100*d66."""
    del initial_t, kwargs
    point, dual = np.zeros(73), np.zeros(len(A))
    pose_row, command_upper = 4369 + 24 + 22, len(A) - 1 - 72 + 66
    point[66] = b[command_upper]
    point[-1] = -b[pose_row] - A[pose_row, 66] * point[66]
    dual[pose_row] = 1.0
    dual[command_upper] = A[pose_row, 66] - point[66]
    return certificate_from_point(B, g, A, b, point, dual)


def run(monkeypatch, evaluate, arguments, solver):
    Js, Jq, historical, previous, lower, upper, config = arguments
    monkeypatch.setattr(core, "solve_epigraph_qp", solver)
    return core.search(evaluate, historical, Js, Jq, previous, lower, upper, config)


@pytest.fixture
def heterogeneous_archive(monkeypatch):
    evaluate, arguments = heterogeneous_inputs()
    return run(monkeypatch, evaluate, arguments, heterogeneous_certificate), arguments


@pytest.fixture
def feedback_archive(monkeypatch):
    evaluate, arguments = nonlinear_inputs()
    return run(monkeypatch, evaluate, arguments, capped_restoration_certificate), arguments


def test_heterogeneous_global_ceiling_reaches_every_qp_row_and_actual_admission(audit, heterogeneous_archive):
    answer, arguments = heterogeneous_archive
    checked = audit.audit_search(answer, *arguments)
    np.testing.assert_array_equal(answer["pose_ceiling"], np.full(24, 4.04))
    assert checked["pose_ceiling_policy"] == "fixed_initial_global_maximum"
    stage, actual = answer["stages"][0], answer["evaluated_candidates"][1]
    assert stage["accepted_response_index"] == 1 and stage["attempts"][0]["accepted"]
    assert actual["normalized_squared_errors"][11, 0] == pytest.approx(3.5)
    assert actual["normalized_squared_errors"][11, 1] == pytest.approx(1.5)
    assert actual["normalized_squared_errors"][11, 1] > 1.01
    assert checked["accepted_response_indices"] == [1] and checked["response_evaluations"] == 3
    assert checked["rejected_update_count"] == 0
    assert checked["witness_indices"] == [] and checked["selected_response_index"] is None
    assert not checked["control_qualified"]


@pytest.mark.parametrize("defect", ["per_item", "current_anchor", "accumulated", "policy", "missing_policy"])
def test_envelope_is_uniform_and_bound_to_the_original_initial_state(audit, heterogeneous_archive, defect):
    answer, arguments = deepcopy(heterogeneous_archive)
    initial = arguments[2]["normalized_squared_errors"].ravel()
    if defect == "per_item":
        answer["pose_ceiling"] = np.maximum(initial, 1.0) * 1.01
    elif defect == "current_anchor":
        current = answer["evaluated_candidates"][answer["final_anchor_index"]]
        answer["pose_ceiling"] = np.full(24, current["maximum_normalized_squared_error"] * 1.01)
    elif defect == "accumulated":
        answer["pose_ceiling"] *= 1.01
    elif defect == "policy":
        answer["pose_ceiling_policy"] = "per_component"
    else:
        del answer["pose_ceiling_policy"]
    with pytest.raises((ValueError, KeyError)):
        audit.audit_search(answer, *arguments)


def test_legacy_per_item_qp_certificate_cannot_hide_behind_global_actual_admission(audit, monkeypatch):
    evaluate, arguments = heterogeneous_inputs()
    historical = arguments[2]

    def legacy_certificate(B, g, A, b, initial_t, **kwargs):
        del initial_t, kwargs
        legacy_b = b.copy()
        q0 = historical["normalized_squared_errors"].ravel()
        legacy_b[4369 : 4369 + 24] = np.maximum(q0, 1.0) * 1.01 - q0
        ceiling_row, mode_row = 4369 + 23, 4369 + 24 + 22
        point, dual = np.zeros(73), np.zeros(len(A))
        point[66] = -legacy_b[ceiling_row] / A[ceiling_row, 66]
        point[-1] = -legacy_b[mode_row] - A[mode_row, 66] * point[66]
        dual[mode_row] = 1.0
        dual[ceiling_row] = (A[mode_row, 66] - point[66]) / -A[ceiling_row, 66]
        return certificate_from_point(B, g, A, legacy_b, point, dual)

    answer = run(monkeypatch, evaluate, arguments, legacy_certificate)
    assert answer["stages"][0]["accepted_response_index"] == 1
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("defect", ["feasible", "selected", "witness_indices"])
def test_numeric_envelope_can_never_replace_original_execution_caps(audit, heterogeneous_archive, defect):
    answer, arguments = deepcopy(heterogeneous_archive)
    observed = answer["evaluated_candidates"][1]
    assert np.all(observed["normalized_squared_errors"].ravel() < answer["pose_ceiling"])
    assert np.any(observed["normalized_squared_errors"] > 1)
    if defect == "feasible":
        observed["feasible"] = True
    elif defect == "selected":
        answer["selected_response_index"], answer["selected"] = 1, deepcopy(observed)
    else:
        answer["witness_indices"] = answer["qualified_witness_indices"] = [1]
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_global_feedback_source_and_next_qp_still_use_the_same_fixed_envelope(audit, feedback_archive):
    answer, arguments = feedback_archive
    checked = audit.audit_search(answer, *arguments)
    assert checked["rejected_response_indices"] == [5]
    assert checked["accepted_response_indices"] == [10]
    first, second = answer["stages"]
    source = answer["evaluated_candidates"][5]
    assert first["rejected_update"]["alpha"] == 0.0625
    assert first["anchor_before_index"] == first["anchor_after_index"] == second["anchor_before_index"] == 0
    assert np.any(source["normalized_squared_errors"].ravel() > answer["pose_ceiling"])
    assert source["audit"]["feasible"] and min(source["safety_slack"]) >= arguments[-1]["safety_buffer"]
    assert checked["selected_response_index"] is None and not checked["control_qualified"]


def test_next_qp_cannot_reuse_the_stale_model_after_global_ceiling_feedback(audit, feedback_archive):
    answer, arguments = deepcopy(feedback_archive)
    Js, Jq, historical, previous, _, _, config = arguments
    second = answer["stages"][1]
    x = np.asarray(historical["commands"])
    B, g, A, b = audit._problem(
        x,
        np.asarray(historical["safety_slack"]),
        historical["normalized_squared_errors"].ravel(),
        Js,
        Jq,
        previous,
        answer["global_lower"],
        answer["global_upper"],
        answer["pose_ceiling"],
        second["radius_before"],
        second["mode"],
        config,
    )
    second["qp"] = capped_restoration_certificate(B, g, A, b, 0.0)
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


@pytest.mark.parametrize("defect", ["unsafe_source", "wrong_alpha", "moved_anchor", "future", "repeat", "counter"])
def test_original_safety_causality_repeat_budget_and_feedback_invariants_remain(audit, feedback_archive, defect):
    answer, arguments = deepcopy(feedback_archive)
    first = answer["stages"][0]
    if defect == "unsafe_source":
        answer["evaluated_candidates"][5]["safety_slack"][0] = np.nextafter(1e-8, -np.inf)
    elif defect == "wrong_alpha":
        first["rejected_update"]["source_response_index"] = 1
    elif defect == "moved_anchor":
        first["anchor_after_index"] = 5
    elif defect == "future":
        arguments[0][0, 6] = 1e-30
    elif defect == "repeat":
        answer["repeatability"]["commands_bytes_equal"] = False
    else:
        answer["prediction_calls"] += 1
    with pytest.raises(ValueError):
        audit.audit_search(answer, *arguments)


def test_unqualified_qp_keeps_the_initial_repeat_and_no_fake_feedback(audit, monkeypatch):
    evaluate, arguments = heterogeneous_inputs()
    answer = run(monkeypatch, evaluate, arguments, failed_certificate)
    checked = audit.audit_search(answer, *arguments)
    assert checked["passed"] and checked["response_evaluations"] == 2
    assert checked["rejected_update_count"] == 0 and checked["accepted_response_indices"] == []
    assert checked["repeat_exact"] and checked["selected_response_index"] is None


def test_independent_audit_imports_only_frozen_algebra_not_any_search_or_solver(audit):
    tree = ast.parse(Path(audit.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("s2a." in name or "scipy" in name or "mujoco" in name for name in imports)
    assert "interior_trajectory_search_audit" in imports
