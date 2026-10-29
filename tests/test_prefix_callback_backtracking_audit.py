"""Independent ledger checks on synthetic responses, never native dynamics.

The new solver is only an archive generator in these tests. The audit module
never imports it; the toy pose and all nine constraint groups are reclassified
by the frozen independent test arithmetic before ledger admission.
"""

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_prefix_preserving_trajectory_plan import inputs, optimizer_result, response
from test_prefix_trajectory_audit_ledger import classify as independent_classify

import s2a.prefix_callback_backtracking_plan as planning


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prefix_callback_backtracking_audit").audit_callback_backtracking_ledger


def classifications(answer, arguments):
    metrics = independent_classify(answer, arguments)
    for checked, row in zip(metrics, answer["evaluated_candidates"], strict=True):
        checked["tcp_errors"] = np.asarray(row["tcp_errors"])
    return metrics


def arguments():
    value = inputs()
    value["desired"][1, 0, 3] = 0.1
    return value


def capture_trial(_objective, start, *, callback, **kwargs):
    trial = start.copy()
    trial[[0, 1, 6]] += 0.01
    callback(trial)
    pytest.fail("The first nonzero callback must interrupt generation")


@pytest.fixture
def half_ray_archive(monkeypatch):
    monkeypatch.setattr(planning, "minimize", capture_trial)
    values = arguments()
    answer = planning.solve_prefix_callback_backtracking_response(**values, max_stages=1)
    return answer, classifications(answer, values)


def test_first_safe_half_ray_is_accepted_and_alpha1_reuses_the_callback(audit, half_ray_archive):
    answer, metrics = half_ray_archive
    checked = audit(answer, metrics)
    phase = answer["phases"][0]
    assert phase["status"] == "direction_captured" and not phase["optimizer_returned"]
    assert phase["callback_count"] == phase["iterations"] == 1
    assert [attempt["alpha"] for attempt in phase["ray_attempts"]] == [1.0, 0.5]
    assert phase["ray_attempts"][0]["outcome"] == "cache_hit"
    assert phase["ray_attempts"][0]["rejection_reasons"] == ["prefix_lost"]
    assert phase["ray_attempts"][1]["eligible"] and phase["ray_attempts"][1]["selected"]
    assert checked["accepted_anchor_indices"] == [0, 2]
    assert checked["ray_response_indices"] == [1, 2]
    assert checked["total_backtracking_attempts"] == 2 and checked["total_backtracking_new_responses"] == 1
    assert not checked["full_ray_and_anchor_audits_checked_here"]


def test_zero_callback_counts_but_ray_finishes_at_the_shared_iteration_limit(audit, monkeypatch):
    def optimizer(objective, start, *, callback, **kwargs):
        callback(start)
        capture_trial(objective, start, callback=callback, **kwargs)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = arguments()
    answer = planning.solve_prefix_callback_backtracking_response(**values, max_iterations=2)
    checked = audit(answer, classifications(answer, values))
    assert answer["phases"][0]["zero_callback_count"] == 1
    assert checked["total_iterations"] == 2 and checked["termination_reason"] == "iteration_budget"
    assert checked["accepted_anchor_indices"] == [0, 2]
    assert answer["response_evaluations"] == 3


def test_safe_fd_descent_cannot_become_an_ordinary_anchor(audit, monkeypatch):
    def predict(commands):
        value = response(commands)
        if commands[0, 1] != 0:
            value["inner_joint_position"][0, 6] = 1.99
        return value

    def optimizer(_objective, start, *, jac, callback, **kwargs):
        jac(start)
        trial = start.copy()
        trial[1] += 0.001
        callback(trial)
        pytest.fail("Captured once")

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = arguments()
    values["predict"] = predict
    answer = planning.solve_prefix_callback_backtracking_response(**values)
    checked = audit(answer, classifications(answer, values))
    assert checked["termination_reason"] == "trust_region_stalled"
    assert checked["accepted_anchor_indices"] == [0]
    assert checked["selected_trace_index"] != 0 and not answer["final_selected_is_last_anchor"]
    assert [phase["radius"] for phase in answer["phases"]] == [0.01, 0.005, 0.0025]
    assert all(phase["ray_termination"] == "alphas_exhausted" for phase in answer["phases"])
    assert checked["total_backtracking_attempts"] == 27
    assert checked["total_backtracking_new_responses"] == 8


def test_new_stage_bounds_change_fd_sign_without_charging_cached_probes_twice(audit, monkeypatch):
    def optimizer(_objective, start, *, jac, constraints, **kwargs):
        point = start.copy()
        point[0] = 0.005
        jac(point)
        constraints[1]["jac"](point)
        return optimizer_result(start)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = arguments()
    values["predict"] = lambda commands: response(np.zeros_like(commands))
    answer = planning.solve_prefix_callback_backtracking_response(**values, max_stages=2)
    metrics = classifications(answer, values)
    checked = audit(answer, metrics)
    probes = [r for r in answer["request_ledger"] if r["role"] == "gradient_probe" and r["axis"] == 0]
    assert [r["nominal_delta"] for r in probes] == [1e-4, -1e-4]
    assert checked["total_backtracking_attempts"] == 0 and answer["response_evaluations"] == 15
    forged = deepcopy(answer)
    probe = next(r for r in forged["request_ledger"] if r["role"] == "gradient_probe" and r["stage_index"] == 1)
    probe["nominal_delta"] = -probe["nominal_delta"]
    with pytest.raises(ValueError, match="nominal step"):
        audit(forged, metrics)


def test_denied_ray_records_no_entry_and_keeps_all_previous_billing(audit, monkeypatch):
    def predict(commands):
        value = response(commands)
        value["pose"][0, :3, 3] *= 3
        return value

    def optimizer(objective, start, *, callback, **kwargs):
        for index in range(1, 11):
            point = start.copy()
            point[7] = index * 1e-6
            objective(point)
        capture_trial(objective, start, callback=callback, **kwargs)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = arguments()
    values["predict"] = predict
    answer = planning.solve_prefix_callback_backtracking_response(**values, max_evaluations=13)
    checked = audit(answer, classifications(answer, values))
    phase = answer["phases"][0]
    assert phase["status"] == "direction_captured" and phase["ray_termination"] == "response_budget"
    assert [attempt["alpha"] for attempt in phase["ray_attempts"]] == [1, 0.5, 0.25]
    denied = phase["ray_attempts"][-1]
    assert denied["entry"] is denied["response_index"] is denied["new_response_index"] is None
    assert denied["outcome"] == "budget_rejected"
    assert checked["accepted_anchor_indices"] == [0] and checked["termination_reason"] == "response_budget"
    assert answer["response_evaluations"] == 13 and checked["total_backtracking_attempts"] == 3


@pytest.mark.parametrize("interrupt", ["probe", "callback", "final"])
def test_last_completed_global_witness_survives_generation_budget_rejection(audit, monkeypatch, interrupt):
    def optimizer(objective, start, *, callback, jac, **kwargs):
        for index in range(1, 13):
            point = start.copy()
            point[6] = index * 1e-4
            objective(point)
        missing = start.copy()
        missing[7] = 1e-4
        if interrupt == "probe":
            jac(start)
        if interrupt == "callback":
            callback(missing)
        return optimizer_result(missing, nit=2)

    monkeypatch.setattr(planning, "minimize", optimizer)
    values = inputs()
    values["desired"][1, 0, 3] = 0.01115
    answer = planning.solve_prefix_callback_backtracking_response(**values, max_evaluations=13)
    checked = audit(answer, classifications(answer, values))
    assert checked["witness_indices"] == [12] and checked["selected_trace_index"] == 12
    assert checked["termination_reason"] == "witness" and checked["accepted_anchor_indices"] == [0, 12]
    assert answer["phases"][0]["status"] == "response_budget"
    assert answer["phases"][0]["ray_attempts"] == []
    assert answer["phases"][0]["zero_callback_count"] == 0


def test_ray_full_witness_uses_original_audit_tolerance_without_extra_buffer(audit, monkeypatch):
    def predict(commands):
        value = response(commands)
        value["joint_velocity"][-1, 6] = 5e-11 if np.any(commands) else -0.01
        return value

    monkeypatch.setattr(planning, "minimize", capture_trial)
    values = inputs()
    values["predict"] = predict
    values["desired"][1, 0, 3] = 0.018
    values["terminal_normals"] = np.eye(7)[6:]
    answer = planning.solve_prefix_callback_backtracking_response(**values)
    checked = audit(answer, classifications(answer, values))
    assert checked["termination_reason"] == "witness"
    assert answer["phases"][0]["ray_attempts"][-1]["alpha"] == 0.5
    assert answer["selected"]["audit"]["feasible"] and answer["selected"]["safety_slack"].min() < 1e-8


def test_complete_initial_warm_has_no_optimizer_or_ray(audit, monkeypatch):
    monkeypatch.setattr(planning, "minimize", lambda *args, **kwargs: pytest.fail("No optimizer for complete warm"))
    values = inputs()
    answer = planning.solve_prefix_callback_backtracking_response(**values)
    checked = audit(answer, classifications(answer, values))
    assert checked["witness_indices"] == [0] and checked["request_count"] == 1
    assert checked["accepted_anchor_indices"] == [0] and checked["ray_response_indices"] == []


@pytest.mark.parametrize(
    "defect",
    [
        "alpha_list_ulp",
        "alpha_order",
        "alpha_command_ulp",
        "captured_request",
        "captured_displacement",
        "generation_end",
        "zero_callbacks",
        "fake_nit",
        "cache_charge",
        "response_count",
        "ray_count",
        "ray_eligibility",
        "ray_reason",
        "ray_selection",
        "anchor_from_callback",
        "final_rank",
        "last_anchor_flag",
        "termination",
        "radius",
    ],
)
def test_forged_ray_or_global_ledger_contract_is_rejected(audit, half_ray_archive, defect):
    original, metrics = half_ray_archive
    answer = deepcopy(original)
    phase = answer["phases"][0]
    attempt = phase["ray_attempts"][-1]
    if defect == "alpha_list_ulp":
        answer["backtracking_alphas"][1] = np.nextafter(0.5, np.inf)
    elif defect == "alpha_order":
        attempt["alpha"] = 0.25
    elif defect == "alpha_command_ulp":
        attempt["commands"][0, 0] = np.nextafter(attempt["commands"][0, 0], np.inf)
    elif defect == "captured_request":
        phase["captured_callback"]["request_index"] -= 1
    elif defect == "captured_displacement":
        phase["captured_callback"]["displacement"][0, 0] += 1e-12
    elif defect == "generation_end":
        phase["generation_requests_after"] += 1
    elif defect == "zero_callbacks":
        phase["zero_callback_count"] += 1
    elif defect == "fake_nit":
        phase["optimizer_returned"] = True
        phase["optimizer_iterations"] = 1
    elif defect == "cache_charge":
        answer["request_ledger"][phase["ray_attempts"][0]["request_index"]]["outcome"] = "new_response"
    elif defect == "response_count":
        answer["request_ledger"][-1]["response_evaluations_after"] += 1
    elif defect == "ray_count":
        answer["total_backtracking_attempts"] += 1
    elif defect == "ray_eligibility":
        attempt["eligible"] = False
    elif defect == "ray_reason":
        phase["ray_attempts"][0]["rejection_reasons"] = []
    elif defect == "ray_selection":
        phase["ray_attempts"][0]["selected"] = True
    elif defect == "anchor_from_callback":
        phase["anchor_after_index"] = phase["captured_callback"]["response_index"]
    elif defect == "final_rank":
        answer["selected"] = answer["initial"]
    elif defect == "last_anchor_flag":
        answer["final_selected_is_last_anchor"] = False
    elif defect == "termination":
        answer["termination_reason"] = "witness"
    elif defect == "radius":
        phase["radius_after"] = 0.005
    with pytest.raises(ValueError):
        audit(answer, metrics)
