"""Synthetic ledger binding checks, not independent robot dynamics evidence.

Classifications here come from the saved synthetic response records. Formal
use must first obtain them from the separate complete response replay.
"""

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_prefix_optimization_reserve_plan import inputs, response

from s2a.accepted_point_sqp_core import damped_bfgs
from s2a.accepted_point_sqp_plan import solve_accepted_point_sqp_response


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("accepted_point_sqp_ledger_audit").audit_accepted_point_sqp_ledger


def saved_classifications(answer):
    records = []
    for row in answer["evaluated_candidates"]:
        errors = np.asarray(row["tcp_errors"])
        classified = {
            name: deepcopy(row[name])
            for name in ("joint_feasible", "feasible", "prefix_length", "tcp_errors", "safety_min", "safety_slack")
        }
        classified["normalized_squared_errors"] = np.stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2),
            axis=1,
        )
        records.append(classified)
    return records


@pytest.fixture(scope="module")
def archives():
    records = {}

    def save(name, arguments, **budgets):
        answer = solve_accepted_point_sqp_response(**arguments, **budgets)
        records[name] = (answer, saved_classifications(answer))

    save("warm", inputs())
    arguments = inputs()
    arguments["desired"][:, 0, 3] = 0.015
    save("qp_witness", arguments, max_stages=5)

    def nonlinear(commands):
        value = response(commands)
        value["pose"][:] = np.eye(4)
        u = commands[0, 0]
        value["pose"][0, 0, 3] = u - 1000 * u**2
        return value

    arguments = inputs(nonlinear, horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    save("pending", arguments, max_stages=1)
    save("curvature", arguments, max_stages=2)
    arguments = inputs(horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    save("partial_fd", arguments, max_evaluations=9)

    near_cap = 0.01 * np.sqrt(1 - 1e-4 / 2)

    def near_cap_response(commands):
        value = response(commands)
        value["pose"][0, 0, 3] = near_cap
        return value

    arguments = inputs(near_cap_response)
    arguments["desired"][1, 0, 3] = near_cap + 1e-4
    save("fd_witness", arguments, max_evaluations=13)
    return records


@pytest.mark.parametrize("kind", ["warm", "qp_witness", "pending", "curvature", "partial_fd", "fd_witness"])
def test_actual_synthetic_archives_validate_without_promoting_evidence(audit, archives, kind):
    answer, classified = archives[kind]
    checked = audit(answer, classified)
    assert checked["request_count"] == len(answer["request_ledger"])
    assert checked["total_iterations"] == answer["total_iterations"]
    assert checked["total_qp_iterations"] == answer["total_qp_iterations"]
    assert checked["termination_reason"] == answer["termination_reason"]
    assert checked["accepted_anchor_indices"] == [anchor["response_index"] for anchor in answer["accepted_anchors"]]
    assert not checked["full_ray_and_anchor_audits_checked_here"]
    assert bool(checked["witness_indices"]) == answer["witness_found"]
    if kind == "curvature":
        assert answer["phases"][1]["curvature_update"]["updated"]
        assert checked["curvature_checks"]
    if kind == "partial_fd":
        assert not answer["phases"][1]["linearization"]["complete"]
        assert checked["selected_trace_index"] != checked["accepted_anchor_indices"][-1]


def test_probe_response_cannot_be_rebound_to_another_fd_axis(audit, archives):
    answer, classified = deepcopy(archives["qp_witness"])
    probes = answer["phases"][0]["linearization"]["probe_response_indices"]
    probes[0], probes[1] = probes[1], probes[0]
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_algebraically_valid_bfgs_for_full_direction_is_not_the_accepted_pair(audit, archives):
    answer, classified = deepcopy(archives["curvature"])
    first, second = answer["phases"]
    update = second["curvature_update"]
    wrong_s = first["qp_direction"]["displacement"].ravel()
    assert not np.array_equal(wrong_s, update["displacement"])
    wrong_update = damped_bfgs(second["hessian_before"], wrong_s, update["gradient_change"])
    assert wrong_update["updated"]
    update.update(wrong_update, displacement=wrong_s.copy())
    second["hessian_after_update"] = wrong_update["matrix"].copy()
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_objective_only_secant_cannot_replace_same_dual_lagrangian_change(audit, archives):
    answer, classified = deepcopy(archives["curvature"])
    first, second = answer["phases"]
    update = second["curvature_update"]
    old_g = first["linearization"]["objective_gradient"]
    new_g = second["linearization"]["objective_gradient"]
    wrong_y = new_g - old_g
    assert not np.allclose(wrong_y, update["gradient_change"])
    wrong_update = damped_bfgs(second["hessian_before"], update["displacement"], wrong_y)
    assert wrong_update["updated"]
    update.update(
        wrong_update,
        gradient_change=wrong_y,
        previous_lagrangian_gradient=old_g.copy(),
        current_lagrangian_gradient=new_g.copy(),
    )
    second["hessian_after_update"] = wrong_update["matrix"].copy()
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_incomplete_fd_group_cannot_claim_a_curvature_update(audit, archives):
    answer, classified = deepcopy(archives["partial_fd"])
    answer["phases"][1]["curvature_update"]["updated"] = True
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_saved_qp_certificate_is_checked_by_the_ledger(audit, archives):
    answer, classified = deepcopy(archives["qp_witness"])
    answer["phases"][0]["qp"]["stationarity_relative"] = 0.1
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_cached_anchor_cannot_be_billed_as_a_new_response(audit, archives):
    answer, classified = deepcopy(archives["pending"])
    anchor_request = next(item for item in answer["request_ledger"] if item["role"] == "stage_anchor")
    assert anchor_request["outcome"] == "cache_hit"
    anchor_request["outcome"] = "new_response"
    anchor_request["new_response_index"] = anchor_request["response_index"]
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_rejected_ray_cannot_replace_the_recorded_accepted_anchor(audit, archives):
    answer, classified = deepcopy(archives["pending"])
    rejected = answer["phases"][0]["ray_attempts"][0]
    assert not rejected["eligible"]
    answer["accepted_anchors"][1]["response_index"] = rejected["response_index"]
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_global_selection_must_include_completed_fd_responses(audit, archives):
    answer, classified = deepcopy(archives["partial_fd"])
    last_anchor = next(attempt["entry"] for attempt in answer["phases"][0]["ray_attempts"] if attempt["selected"])
    assert not answer["final_selected_is_last_anchor"]
    answer["selected"] = deepcopy(last_anchor)
    answer["final_selected_is_last_anchor"] = True
    with pytest.raises(ValueError):
        audit(answer, classified)


def test_normalized_errors_cannot_silently_relax_original_pose_caps(audit, archives):
    answer, classified = deepcopy(archives["pending"])
    # Keep measured synthetic errors intact while claiming a wider position cap.
    for row in classified:
        row["normalized_squared_errors"][:, 0] *= 0.5
    with pytest.raises(ValueError):
        audit(answer, classified)
