"""Synthetic geometry/admission/ledger checks; no native dynamics or saved experiment inputs."""

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_prefix_preserving_trajectory_plan import inputs, optimizer_result, response

import s2a.prefix_preserving_trajectory_plan as planning
from s2a.paired_reference_phase import prepare_paired_reference_phases, score_phase_response


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_tracking_v5_paired_reference_phase")


def geometry():
    targets = np.tile(np.eye(4), (40, 1, 1))
    targets[:, 0, 3] = np.arange(40) * 0.03
    targets[:, :3, :3] = Rotation.from_euler(
        "xyz", np.c_[0.01 * np.arange(40), 0.005 * np.arange(40), -0.007 * np.arange(40)]
    ).as_matrix()
    opening = np.linspace(0.0, 1.0, 40)
    warm = np.tile(np.arange(12)[:, None] * 0.01, (1, 6))
    return targets, opening, warm


@pytest.mark.parametrize("name,insertion", [("early", 6), ("late", 9)])
def test_independent_contract_matches_producer_and_keeps_all_original_nodes(checker, name, insertion):
    targets, openings, warm = geometry()
    before = [value.copy() for value in (targets, openings, warm)]
    case = checker.reconstruct_phase_case(name, targets, openings, warm)
    archived = prepare_paired_reference_phases(
        targets, openings, warm, previous=np.zeros(6), interface=inputs()["interface"], case_name=name
    )[name]
    checker.compare_contract(case, archived)
    checker.check_command_admission(case, np.zeros(6), -np.ones(6), np.ones(6))
    indices = np.delete(np.arange(13), insertion)
    np.testing.assert_array_equal(case["original_node_endpoint_indices"], indices)
    np.testing.assert_array_equal(case["desired_tcp"][indices], targets[:12])
    np.testing.assert_array_equal(case["warm"][indices], warm)
    np.testing.assert_array_equal(case["original_timed_tcp"][-1], targets[12])
    assert case["scheduled_final_original_cursor"] == 11
    assert case["scheduled_original_node_count"] == 12 and case["actual_consumed_reference_count"] == 0
    assert case["physical_time_s"][-1] == pytest.approx(0.65)
    assert case["reference_time_s"][-1] == pytest.approx(0.60)
    for original, actual in zip(before, (targets, openings, warm), strict=True):
        np.testing.assert_array_equal(original, actual)


@pytest.mark.parametrize("name,insertion", [("early", 6), ("late", 9)])
def test_timed13_and_inserted_point_cannot_be_hidden_by_twelve_good_nodes(checker, name, insertion):
    case = checker.reconstruct_phase_case(name, *geometry())
    tool = np.eye(4)
    tool[:3, 3] = [0.03, -0.02, 0.01]
    tool[:3, :3] = Rotation.from_rotvec([0.12, -0.08, 0.06]).as_matrix()
    predicted_tcp = case["desired_tcp"].copy()
    site = predicted_tcp @ np.linalg.inv(tool)
    score = checker.score_phase_contracts(site, case, tool)
    checker.compare_contract(score, score_phase_response(site, case, tool))
    assert score["retimed_path"]["all_pose_caps_satisfied"]
    assert score["original_nodes"]["prefix_length"] == 12
    assert not score["original_timed"]["pose_cap_pass"][-1]
    assert score["original_timed"]["tcp_errors"][-1, 0] == pytest.approx(0.03)
    predicted_tcp[insertion, 1, 3] += 0.011
    bad = checker.score_phase_contracts(predicted_tcp @ np.linalg.inv(tool), case, tool)
    assert bad["original_nodes"]["all_pose_caps_satisfied"]
    assert bad["predicted_qualified_original_node_prefix"] == 12
    assert bad["retimed_path"]["prefix_length"] == insertion
    assert not bad["retimed_path"]["all_pose_caps_satisfied"]
    assert bad["actual_consumed_reference_count"] == 0


def test_strict_contract_labels_do_not_inherit_numeric_comparison_tolerance(checker):
    targets = np.tile(np.eye(4), (40, 1, 1))
    case = checker.reconstruct_phase_case("early", targets, np.ones(40), np.zeros((12, 6)))
    measured = case["desired_tcp"].copy()
    measured[-1, 1, 3] = 0.01
    inside = checker.score_phase_contracts(measured, case, np.eye(4))
    measured[-1, 1, 3] = np.nextafter(0.01, np.inf)
    outside = checker.score_phase_contracts(measured, case, np.eye(4))
    assert inside["retimed_path"]["all_pose_caps_satisfied"]
    assert not outside["retimed_path"]["all_pose_caps_satisfied"]
    assert abs(inside["retimed_path"]["pose_loss"] - outside["retimed_path"]["pose_loss"]) < 1e-10
    with pytest.raises(ValueError, match="Exact"):
        checker.compare_contract(inside, outside)


def rejected_command_record():
    return dict(
        status="command_admission_failed",
        case_contract=None,
        answer=None,
        warm_preflight=None,
        warm_admission=None,
        command_admission=dict(admitted=False, reason="preflight rejected"),
        response_evaluations=0,
        total_iterations=0,
        prediction_calls=0,
        cache_hits=0,
        query_count=0,
        numerical_inner_steps=0,
        trace=[],
        phases=[],
        dual_metrics={},
        witness_found=False,
    )


def test_roundoff_clipping_is_rejected_before_queries_and_needs_independent_evidence(checker):
    targets, opening, warm = geometry()
    warm[:, 0] = 1.0 + 5e-13  # Within the old box tolerance, but solver clipping changes bytes.
    previous = np.zeros(6)
    previous[0] = 0.8
    case = checker.reconstruct_phase_case("late", targets, opening, warm)
    record = rejected_command_record()
    assert "warm byte" in checker.audit_command_rejection(record, case, previous, -np.ones(6), np.ones(6))
    leaked = deepcopy(record)
    leaked["query_count"] = 1
    with pytest.raises(ValueError, match="zero queries"):
        checker.audit_command_rejection(leaked, case, previous, -np.ones(6), np.ones(6))
    valid = checker.reconstruct_phase_case("late", *geometry())
    with pytest.raises(ValueError, match="valid registered warm"):
        checker.audit_command_rejection(record, valid, np.zeros(6), -np.ones(6), np.ones(6))


@pytest.mark.parametrize(
    "safe,slack,reason",
    [(False, -0.001, "joint_or_command_audit_failed"), (True, 5e-9, "optimization_safety_buffer_not_met")],
)
def test_dynamic_rejection_retains_one_complete_response_and_no_search(checker, safe, slack, reason):
    # Prevalidated classification fixture: ledger tests do not stand in for the
    # independent dynamics reconstruction that must precede this helper.
    normalized = np.tile([4.0, 0.0], (13, 1))
    normalized[0] = 0.0
    classification = dict(
        joint_feasible=safe, feasible=False, safety_min=slack, prefix_length=1, normalized_squared_errors=normalized
    )
    producer = dict(
        trace=[{}],
        answer=None,
        response_evaluations=1,
        prediction_calls=1,
        cache_hits=0,
        numerical_inner_steps=325,
        query_count=339,
        total_iterations=0,
        phases=[],
        witness_found=False,
        status="warm_admission_failed",
        warm_admission=checker.warm_admission_status(classification),
    )
    result = checker.audit_case_accounting(producer, [classification])
    assert producer["warm_admission"]["rejection_reason"] == reason
    assert result["termination_reason"] == "warm_admission_failed"
    assert result["selected_trace_index"] is None and result["witness_indices"] == []
    for field, forged in [("prediction_calls", 2), ("query_count", 338), ("answer", {})]:
        modified = deepcopy(producer)
        modified[field] = forged
        with pytest.raises(ValueError):
            checker.audit_case_accounting(modified, [classification])


@pytest.mark.parametrize("complete", [True, False])
def test_real_h13_solver_ledger_reuses_preflight_and_does_not_assume_prefix_six(monkeypatch, checker, complete):
    def predict(commands):
        result = response(commands)
        if complete:
            result["joint_velocity"][-1, 6] = 5e-11
        return result

    arguments = inputs(predict, horizon=13)
    arguments["terminal_normals"] = np.eye(7)[6:]
    if not complete:
        arguments["desired"][1, 0, 3] = 0.02

    def unchanged(_objective, start, **_kwargs):
        assert not complete, "A complete witness inside the optimization buffer needs no optimizer"
        return optimizer_result(start, nit=0)

    # For an incomplete warm the terminal buffer must be positive.
    if not complete:

        def predict(commands):
            result = response(commands)
            result["joint_velocity"][-1, 6] = -0.01
            return result

        arguments["predict"] = predict
    monkeypatch.setattr(planning, "minimize", unchanged)
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments)
    normalized = np.zeros((13, 2))
    if not complete:
        normalized[1, 0] = 4.0
    classification = dict(
        joint_feasible=True,
        feasible=complete,
        safety_min=-5e-11 if complete else 0.01,
        prefix_length=13 if complete else 1,
        normalized_squared_errors=normalized,
    )
    producer = dict(
        answer=answer,
        trace=deepcopy(answer["evaluated_candidates"]),
        response_evaluations=1,
        prediction_calls=2,
        cache_hits=1,
        numerical_inner_steps=325,
        query_count=339,
        total_iterations=answer["total_iterations"],
        phases=deepcopy(answer["phases"]),
        witness_found=answer["witness_found"],
        status=answer["status"],
        warm_admission=checker.warm_admission_status(classification),
    )
    checked = checker.audit_case_accounting(producer, [classification])
    assert checked["initial_prefix_length"] == (13 if complete else 1)
    assert checked["termination_reason"] == ("witness" if complete else "no_prefix_progress")
    assert checked["total_iterations"] == 0
    for field, forged in [("query_count", 313), ("cache_hits", 0), ("numerical_inner_steps", 300)]:
        modified = deepcopy(producer)
        modified[field] = forged
        with pytest.raises(ValueError):
            checker.audit_case_accounting(modified, [classification])
