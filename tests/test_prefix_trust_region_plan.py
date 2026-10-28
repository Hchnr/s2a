"""Synthetic acceptance, finite-difference, and shared-budget contracts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import s2a.prefix_trust_region_plan as planning
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.osc import OSCInterface
from s2a.prefix_preserving_trajectory_plan import solve_prefix_preserving_trajectory_response


def response(commands):
    increments = np.c_[commands, np.zeros(len(commands))]
    endpoints = np.cumsum(increments, axis=0)
    starts = np.vstack((np.zeros(7), endpoints[:-1]))
    inner = starts[:, None] + (np.arange(1, 26) / 25)[None, :, None] * increments[:, None]
    velocity = increments / 0.05
    poses = np.tile(np.eye(4), (len(commands), 1, 1))
    poses[:, :3, 3] = endpoints[:, :3]
    poses[:, :3, :3] = Rotation.from_rotvec(endpoints[:, 3:6]).as_matrix()
    return dict(
        inner_joint_position=inner.reshape(-1, 7),
        inner_joint_velocity=np.repeat(velocity, 25, axis=0),
        joint_position=endpoints,
        joint_velocity=velocity,
        pose=poses,
        twist=velocity[:, :6],
    )


def inputs(predict=response, horizon=2):
    return dict(
        predict=predict,
        previous=np.zeros(6),
        warm=np.zeros((horizon, 6)),
        interface=OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6)),
        lower=-2 * np.ones(7),
        upper=2 * np.ones(7),
        desired=np.tile(np.eye(4), (horizon, 1, 1)),
        site_from_tcp=np.eye(4),
        past_inner_error_rad=np.zeros((25, 7)),
        terminal_normals=np.empty((0, 7)),
    )


def optimizer_result(x, nit=0):
    return SimpleNamespace(success=True, status=0, message="synthetic return", nit=nit, x=x.copy())


def assert_ledger(result, calls):
    """Independently replay first-seen keys, not optimizer internals."""
    known = {}
    for index, request in enumerate(result["request_ledger"]):
        assert request["request_index"] == index
        assert request["response_evaluations_before"] == len(known)
        key = request["command_key"]
        if request["outcome"] == "new_response":
            assert key not in known
            assert request["response_index"] == request["new_response_index"] == len(known)
            known[key] = len(known)
        elif request["outcome"] == "cache_hit":
            assert known[key] == request["response_index"] and request["new_response_index"] is None
        else:
            assert request["outcome"] == "budget_rejected" and key not in known
            assert request["response_index"] is request["new_response_index"] is None
            assert len(known) == result["max_evaluations"]
        assert request["response_evaluations_after"] == len(known)
    assert len(calls) == len(known) == result["response_evaluations"]
    assert [value.tobytes().hex() for value in calls] == list(known)


def test_real_slsqp_reports_budget_without_witness_for_a_feasible_hard_case():
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.045
    before = deepcopy(arguments)
    # Feasibility does not imply that the fixed five-iteration stages find it.
    # This command has zero pose error, rate <= .045, and strict joint margins.
    known = np.zeros((2, 6))
    known[1, 0] = 0.045
    known_rollout = response(known)
    np.testing.assert_array_equal(known_rollout["pose"], arguments["desired"])
    known_audit = audit_joint_preview_constraints(
        known_rollout,
        known,
        arguments["previous"],
        arguments["interface"],
        arguments["lower"],
        arguments["upper"],
        past_inner_error_rad=arguments["past_inner_error_rad"],
        terminal_normals=arguments["terminal_normals"],
        braking_time_s=0.1,
    )
    assert known_audit["feasible"]
    assert all(group["minimum"] is None or group["minimum"] > 1e-8 for group in known_audit["groups"].values())
    result = planning.solve_prefix_trust_region_response(**arguments)
    assert not result["witness_found"] and result["selected_prefix_length"] == 1
    assert result["termination_reason"] == "iteration_budget" and result["total_iterations"] == 60
    assert result["selected"]["audit"]["feasible"]
    assert np.max(result["selected"]["normalized_squared_errors"][1]) > 1
    assert result["total_stages"] >= 2
    assert result["response_evaluations"] <= 4000
    assert any(
        "prefix_lost" in candidate["rejection_reasons"]
        for phase in result["phases"]
        for candidate in phase["candidate_decisions"]
    )
    for phase in result["phases"]:
        assert phase["iterations"] <= phase["max_iterations"] <= 5
        anchor = phase["initial_commands"]
        assert np.max(abs(phase["selected_commands"] - anchor)) <= phase["radius"] + 1e-12
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(arguments[name], value)


def test_real_slsqp_finds_a_witness_with_strict_pose_and_joint_margins():
    arguments = inputs()
    arguments["desired"][:, 0, 3] = 0.015
    # u0=.01,u1=0 yields .005 positional error at both endpoints, so this
    # feasible set has interior; the actual optimizer must still find a witness.
    result = planning.solve_prefix_trust_region_response(**arguments)
    assert result["witness_found"] and result["termination_reason"] == "witness"
    assert result["selected_prefix_length"] == 2 and result["total_iterations"] > 0
    assert result["selected"]["audit"]["feasible"]
    assert np.max(result["selected"]["normalized_squared_errors"]) < 0.5
    assert np.min(result["selected"]["safety_slack"]) > 1e-8
    assert result["total_iterations"] <= 60 and result["response_evaluations"] <= 4000


def test_same_prefix_real_descent_recenters_and_resets_radius(monkeypatch):
    calls, starts = [], []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    def advance(objective, start, *, callback, **kwargs):
        starts.append(start.copy())
        candidate = start.copy()
        candidate[6] += 0.004
        objective(candidate)
        callback(candidate)
        return optimizer_result(candidate, nit=1)

    monkeypatch.setattr(planning, "minimize", advance)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.2
    result = planning.solve_prefix_trust_region_response(**arguments, max_stages=3)
    assert result["termination_reason"] == "stage_budget"
    assert len(starts) == result["total_stages"] == 3
    assert [value[6] for value in starts] == pytest.approx([0, 0.004, 0.008])
    assert len(result["accepted_anchors"]) == 4
    assert all(anchor["prefix_length"] == 1 for anchor in result["accepted_anchors"])
    for anchor, phase in zip(result["accepted_anchors"][1:], result["phases"], strict=True):
        assert anchor["acceptance_reason"] == phase["acceptance_reason"] == "target_descent"
        assert anchor["target_improvement"] > 1e-12 and anchor["audit"]["feasible"]
        assert phase["radius"] == phase["radius_after"] == 0.01
    assert_ledger(result, calls)


def test_three_failed_radii_stop_even_if_optimizer_reports_success(monkeypatch):
    monkeypatch.setattr(planning, "minimize", lambda _objective, start, **kwargs: optimizer_result(start))
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.1
    result = planning.solve_prefix_trust_region_response(**arguments)
    assert [phase["radius"] for phase in result["phases"]] == [0.01, 0.005, 0.0025]
    assert [phase["radius_after"] for phase in result["phases"]] == [0.005, 0.0025, None]
    assert result["termination_reason"] == "trust_region_stalled"
    assert result["response_evaluations"] == 1 and result["total_iterations"] == 0
    assert len(result["accepted_anchors"]) == 1
    assert all(not phase["accepted"] and phase["success"] for phase in result["phases"])


@pytest.mark.parametrize("failure", ["prefix", "inner", "braking", "terminal", "rate", "buffer"])
def test_actual_audit_prefix_and_buffer_reject_a_nominal_improvement(monkeypatch, failure):
    def predict(commands):
        value = response(commands)
        active = commands[1, 0] != (0.25 if failure == "rate" else 0.0)
        if failure == "prefix" and active:
            value["pose"][0, 0, 3] = 0.011
        if failure == "inner" and active:
            value["inner_joint_position"][30, 6] = 1.99
        if failure == "braking" and active:
            value["joint_velocity"][1, 6] = 20
        if failure in ("terminal", "buffer"):
            value["joint_velocity"][-1, 6] = (0.01 if failure == "terminal" else -5e-9) if active else -0.01
        return value

    def improve(objective, start, *, bounds, **kwargs):
        candidate = start.copy()
        candidate[6] = min(start[6] + 0.002, bounds[6][1])
        objective(candidate)
        return optimizer_result(candidate, nit=1)

    monkeypatch.setattr(planning, "minimize", improve)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    if failure in ("terminal", "buffer"):
        arguments["terminal_normals"] = np.eye(7)[6:]
    if failure == "rate":
        arguments["warm"][1, 0] = 0.25
        arguments["desired"][1, 0, 3] = 0.3
    result = planning.solve_prefix_trust_region_response(**arguments)
    assert len(result["accepted_anchors"]) == 1 and result["termination_reason"] == "trust_region_stalled"
    assert all(not phase["accepted"] for phase in result["phases"])
    reasons = {
        reason
        for phase in result["phases"]
        for item in phase["rejected_candidates"]
        for reason in item["rejection_reasons"]
    }
    assert {"prefix": "prefix_lost", "buffer": "safety_buffer"}.get(failure, "joint_or_command") in reasons
    if failure == "buffer":
        # Final safe ranking is deliberately distinct from stricter anchor admission.
        assert result["selected"]["commands"][1, 0] > 0
        assert np.min(result["selected"]["safety_slack"]) < 1e-8


def test_different_stage_bounds_recompute_fd_sign_at_the_same_cached_point(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return response(np.zeros_like(commands))

    def probe(_objective, start, *, jac, constraints, **kwargs):
        point = start.copy()
        point[0] = 0.005
        jac(point)
        constraints[1]["jac"](point)  # Same-stage derivative cache must avoid extra requests.
        return optimizer_result(start)

    monkeypatch.setattr(planning, "minimize", probe)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.1
    result = planning.solve_prefix_trust_region_response(**arguments, max_stages=2)
    probes = [row for row in result["request_ledger"] if row["role"] == "gradient_probe" and row["axis"] == 0]
    assert [row["nominal_delta"] for row in probes] == [1e-4, -1e-4]
    assert result["response_evaluations"] == 15
    assert not any(row["role"] == "jacobian_probe" for row in result["request_ledger"])
    for row in result["request_ledger"]:
        if row["stage_index"] is None:
            continue
        commands = np.frombuffer(bytes.fromhex(row["command_key"]), dtype=float).reshape(2, 6)
        phase = result["phases"][row["stage_index"]]
        assert np.all(commands >= phase["stage_lower_command"] - 1e-12)
        assert np.all(commands <= phase["stage_upper_command"] + 1e-12)
    assert_ledger(result, calls)


def test_response_budget_spans_stages_and_keeps_the_last_completed_descent(monkeypatch):
    calls, stages = [], []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    def fill(objective, start, *, callback, **kwargs):
        stages.append(start.copy())
        if len(stages) == 1:
            candidate = start.copy()
            candidate[6] += 0.003
            objective(candidate)
            callback(candidate)
            return optimizer_result(candidate, nit=1)
        for step in range(1, 13):
            candidate = start.copy()
            candidate[6] += step * 0.0002
            objective(candidate)
        pytest.fail("Shared response budget should stop the second stage")

    monkeypatch.setattr(planning, "minimize", fill)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.2
    result = planning.solve_prefix_trust_region_response(**arguments, max_evaluations=13)
    assert result["termination_reason"] == "response_budget"
    assert result["total_stages"] == 2 and result["response_evaluations"] == 13
    assert result["accepted_anchors"][-1]["response_index"] == 12
    assert result["request_ledger"][-1]["outcome"] == "budget_rejected"
    assert result["phases"][-1]["accepted"]
    assert_ledger(result, calls)


def test_last_budget_response_can_be_a_full_witness_without_extra_buffer(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        value = response(np.zeros_like(commands))
        value["pose"][-1, 0, 3] = commands[-1, -1]
        value["joint_velocity"][-1, 6] = 5e-11 if commands[-1, -1] > 0 else -0.01
        return value

    def fill(objective, start, *, jac, **kwargs):
        jac(start)
        extra = start.copy()
        extra[-2] = 0.0002
        objective(extra)
        pytest.fail("The response after the final FD probe must be budget-rejected")

    monkeypatch.setattr(planning, "minimize", fill)
    arguments = inputs(predict)
    arguments["desired"][-1, 0, 3] = 0.01005
    arguments["terminal_normals"] = np.eye(7)[6:]
    result = planning.solve_prefix_trust_region_response(**arguments, max_evaluations=13)
    assert result["witness_found"] and result["termination_reason"] == "witness"
    anchor = result["accepted_anchors"][-1]
    assert anchor["response_index"] == 12 and anchor["acceptance_reason"] == "witness"
    assert anchor["safety_min"] == -5e-11
    assert result["phases"][0]["status"] == "response_budget"
    assert_ledger(result, calls)


def test_complete_witness_stops_at_callback_without_waiting_for_optimizer_success(monkeypatch):
    def complete(objective, start, *, callback, **kwargs):
        candidate = start.copy()
        candidate[6] = 0.005
        objective(candidate)
        callback(candidate)
        pytest.fail("Complete witness must stop at this callback")

    monkeypatch.setattr(planning, "minimize", complete)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.012
    result = planning.solve_prefix_trust_region_response(**arguments)
    assert result["witness_found"] and result["total_iterations"] == 1
    assert result["phases"][0]["status"] == "witness_found"
    assert not result["phases"][0]["optimizer_returned"]


def test_iteration_budget_is_global_even_when_every_stage_accepts(monkeypatch):
    limits = []

    def use_iterations(objective, start, *, callback, options, **kwargs):
        limits.append(options["maxiter"])
        candidate = start.copy()
        candidate[6] += 0.003
        objective(candidate)
        for _ in range(options["maxiter"]):
            callback(candidate)
        return optimizer_result(candidate, nit=options["maxiter"])

    monkeypatch.setattr(planning, "minimize", use_iterations)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.2
    result = planning.solve_prefix_trust_region_response(**arguments, max_iterations=7)
    assert limits == [5, 2] and result["total_iterations"] == 7
    assert result["termination_reason"] == "iteration_budget"
    assert [phase["callback_count"] for phase in result["phases"]] == [5, 2]


def test_complete_warm_preserves_old_answer_contract_and_uses_no_stage(monkeypatch):
    def predict(commands):
        value = response(commands)
        value["joint_velocity"][-1, 6] = 5e-11
        return value

    monkeypatch.setattr(planning, "minimize", lambda *_args, **kwargs: pytest.fail("Warm witness must not optimize"))
    arguments = inputs(predict)
    arguments["terminal_normals"] = np.eye(7)[6:]
    result = planning.solve_prefix_trust_region_response(**arguments)
    original = solve_prefix_preserving_trajectory_response(**arguments)
    assert set(original) <= set(result)
    assert result["witness_found"] and result["response_evaluations"] == 1
    assert result["total_stages"] == result["total_iterations"] == 0
    assert result["accepted_anchors"][0]["acceptance_reason"] == "initial_witness"
    assert result["accepted_anchors"][0]["stage_target_cursor"] is None
    assert result["request_ledger"][0]["stage_index"] is None
    assert result["request_ledger"][0]["role"] == "warm"
    for group, value in original["initial"]["audit"]["groups"].items():
        np.testing.assert_array_equal(result["initial"]["audit"]["groups"][group]["slack"], value["slack"])


@pytest.mark.parametrize("value,expected", [(0.01, True), (np.nextafter(0.01, np.inf), False)])
def test_original_caps_remain_strict_at_a_one_ulp_boundary(monkeypatch, value, expected):
    monkeypatch.setattr(planning, "minimize", lambda _objective, start, **kwargs: optimizer_result(start))
    arguments = inputs()
    arguments["desired"][1, 0, 3] = value
    result = planning.solve_prefix_trust_region_response(**arguments)
    assert result["witness_found"] is expected


@pytest.mark.parametrize("slack", [-0.01, 5e-9])
def test_incomplete_warm_cannot_silently_start_safety_restoration(monkeypatch, slack):
    def predict(commands):
        value = response(commands)
        value["inner_joint_position"][30, 6] = 1.95 - slack
        return value

    monkeypatch.setattr(planning, "minimize", lambda *_args, **kwargs: pytest.fail("Unsafe warm reached optimizer"))
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.1
    with pytest.raises(ValueError):
        planning.solve_prefix_trust_region_response(**arguments)
