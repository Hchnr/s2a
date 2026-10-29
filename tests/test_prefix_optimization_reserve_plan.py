"""Synthetic reserve tests: no actual trajectory input, simulator, or GPU."""

from copy import deepcopy
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import s2a.prefix_callback_backtracking_plan as frozen
import s2a.prefix_optimization_reserve_plan as planning
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.osc import OSCInterface

solve = planning.solve_prefix_optimization_reserve_response
MARGIN = planning.PREFIX_OPTIMIZATION_MARGIN


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


@pytest.mark.parametrize("prefix", [0, 1, 2])
def test_only_attained_prefix_constraint_changes_not_target_safety_objective_or_jacobian(monkeypatch, prefix):
    captured = []

    def inspect(objective, start, *, jac, constraints, bounds, options, **_kwargs):
        records = []
        point = start.copy()
        point[0] = bounds[0][1]  # Also exercise a backward finite difference.
        point[6] = 0.001
        point[12] = 0.002
        point[-1] += 0.25
        for value in (start, point):
            records.append(
                dict(
                    objective=objective(value),
                    gradient=jac(value),
                    constraint=constraints[1]["fun"](value),
                    jacobian=constraints[1]["jac"](value),
                )
            )
        captured.append(
            dict(
                records=records,
                rate_matrix=constraints[0].A,
                rate_lower=constraints[0].lb,
                rate_upper=constraints[0].ub,
                bounds=bounds,
                options=options,
            )
        )
        return optimizer_result(start)

    monkeypatch.setattr(frozen, "minimize", inspect)
    monkeypatch.setattr(planning, "minimize", inspect)
    arguments = inputs(horizon=3)
    arguments["desired"][prefix, 0, 3] = 0.03
    old = frozen.solve_prefix_callback_backtracking_response(**deepcopy(arguments), max_stages=1)
    new = solve(**deepcopy(arguments), max_stages=1)
    assert signature(solve) == signature(frozen.solve_prefix_callback_backtracking_response)
    assert MARGIN == new["prefix_optimization_margin"] == 1e-4
    assert old["initial_prefix_length"] == new["initial_prefix_length"] == prefix
    before, after = captured
    for key in ("rate_matrix", "rate_lower", "rate_upper"):
        np.testing.assert_array_equal(after[key], before[key])
    assert after["bounds"] == before["bounds"] and after["options"] == before["options"]
    for a, b in zip(before["records"], after["records"], strict=True):
        assert b["objective"] == a["objective"]
        np.testing.assert_array_equal(b["gradient"], a["gradient"])
        np.testing.assert_array_equal(b["jacobian"], a["jacobian"])
        safety_width = len(a["constraint"]) - 2 * prefix - 2
        np.testing.assert_array_equal(b["constraint"][:safety_width], a["constraint"][:safety_width])
        np.testing.assert_array_equal(b["constraint"][-2:], a["constraint"][-2:])
        np.testing.assert_allclose(
            b["constraint"][safety_width:-2],
            a["constraint"][safety_width:-2] - MARGIN,
            rtol=0,
            atol=3e-16,
        )
    # The margin is recorded once, without changing stage or ledger schemas.
    assert old["phases"][0].keys() == new["phases"][0].keys()
    assert old["phases"][0]["phase"] == new["phases"][0]["phase"] == "prefix_callback_backtracking"
    assert old["request_ledger"] == new["request_ledger"]


@pytest.mark.parametrize("component", ["position", "rotation"])
def test_complete_warm_witness_inside_original_caps_needs_no_optimization_reserve(monkeypatch, component):
    monkeypatch.setattr(planning, "minimize", lambda *_a, **_kw: pytest.fail("complete witness does not optimize"))
    arguments = inputs()
    norm = np.sqrt(1.0 - MARGIN / 2)
    if component == "position":
        arguments["desired"][0, 0, 3] = 0.01 * norm
    else:
        arguments["desired"][0, :3, :3] = Rotation.from_rotvec([0, 0, 0.05 * norm]).as_matrix()
    result = solve(**arguments)
    largest = np.max(result["selected"]["normalized_squared_errors"])
    assert 1 - MARGIN < largest <= 1
    assert result["witness_found"] and result["termination_reason"] == "witness"
    assert result["response_evaluations"] == 1 and result["total_iterations"] == result["total_stages"] == 0
    assert result["selected"]["audit"]["feasible"]


def test_callback_witness_is_kept_when_original_caps_pass_but_prefix_reserve_fails(monkeypatch):
    near_cap = 0.01 * np.sqrt(1 - MARGIN / 2)

    def predict(commands):
        value = response(commands)
        value["pose"][0, 0, 3] = near_cap
        return value

    def direction(_objective, start, *, callback, **_kwargs):
        candidate = start.copy()
        candidate[6] = 0.011 - near_cap
        callback(candidate)
        pytest.fail("full witness must stop at the callback boundary")

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.011
    result = solve(**arguments)
    assert result["witness_found"] and result["response_evaluations"] == 2
    assert np.all(result["selected"]["normalized_squared_errors"] <= 1)
    assert np.max(result["selected"]["normalized_squared_errors"][0]) > 1 - MARGIN
    phase = result["phases"][0]
    assert phase["status"] == "witness_found"
    assert phase["ray_termination"] == "witness"
    assert phase["captured_callback"] is None and phase["ray_attempts"] == []
    assert phase["acceptance_reason"] == "witness" and phase["iterations"] == 1


def test_fd_witness_without_reserve_survives_the_last_response_budget(monkeypatch):
    near_cap = 0.01 * np.sqrt(1 - MARGIN / 2)

    def predict(commands):
        value = response(commands)
        value["pose"][0, 0, 3] = near_cap
        return value

    def exhaust(objective, start, *, jac, **_kwargs):
        jac(start)
        trial = start.copy()
        trial[6] = 0.004
        objective(trial)
        pytest.fail("new response must be denied after the completed FD group")

    monkeypatch.setattr(planning, "minimize", exhaust)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = near_cap + 1e-4
    result = solve(**arguments, max_evaluations=13)
    assert result["witness_found"] and result["response_evaluations"] == 13
    assert result["request_ledger"][-1]["outcome"] == "budget_rejected"
    assert result["termination_reason"] == "witness"
    assert result["phases"][0]["status"] == "response_budget"
    assert result["phases"][0]["ray_termination"] == "witness"
    assert np.max(result["selected"]["normalized_squared_errors"][0]) > 1 - MARGIN


@pytest.mark.parametrize("prefix_position,accepted", [(0.01, True), (np.nextafter(0.01, np.inf), False)])
def test_ordinary_ray_admission_uses_original_strict_caps_not_optimization_margin(
    monkeypatch, prefix_position, accepted
):
    def predict(commands):
        value = response(commands)
        value["pose"][0, 0, 3] = prefix_position if commands[1, 0] != 0 else 0.009
        return value

    def direction(_objective, start, *, callback, **_kwargs):
        trial = start.copy()
        trial[6] = 0.004
        callback(trial)

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_stages=1)
    phase = result["phases"][0]
    assert phase["accepted"] is accepted and not result["witness_found"]
    if accepted:
        assert len(phase["ray_attempts"]) == 1
        assert phase["acceptance_reason"] == "target_descent"
        anchor = result["accepted_anchors"][-1]
        assert np.max(anchor["normalized_squared_errors"][0]) == 1 > 1 - MARGIN
        assert anchor["prefix_length"] == 1 and anchor["safety_min"] >= 1e-8
        assert anchor["target_improvement"] > 1e-12
    else:
        assert len(phase["ray_attempts"]) == 9 and len(result["accepted_anchors"]) == 1
        assert all("prefix_lost" in attempt["rejection_reasons"] for attempt in phase["ray_attempts"])


@pytest.mark.parametrize("example", ["hard_feasible", "strict_interior"])
def test_real_slsqp_reports_actual_synthetic_outcome_without_adjusting_parameters(example):
    arguments = inputs()
    known = np.zeros((2, 6))
    if example == "hard_feasible":
        arguments["desired"][1, 0, 3] = 0.045
        known[1, 0] = 0.045
    else:
        arguments["desired"][:, 0, 3] = 0.015
        known[0, 0] = 0.01
    before = deepcopy(arguments)
    known_rollout = response(known)
    audit = audit_joint_preview_constraints(
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
    assert audit["feasible"]
    assert np.all(np.linalg.norm(known_rollout["pose"][:, :3, 3] - arguments["desired"][:, :3, 3], axis=1) < 0.01)
    result = solve(**arguments)
    assert result["selected"]["audit"]["feasible"]
    assert result["witness_found"] == bool(np.all(result["selected"]["normalized_squared_errors"] <= 1))
    if result["witness_found"]:
        assert result["termination_reason"] == "witness" and result["selected_prefix_length"] == 2
    else:
        assert result["termination_reason"] in {
            "response_budget",
            "iteration_budget",
            "stage_budget",
            "trust_region_stalled",
        }
    assert 0 < result["total_iterations"] <= 60
    assert 1 < result["response_evaluations"] <= 4000
    assert result["total_stages"] <= 60 and result["prefix_optimization_margin"] == 1e-4
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(arguments[name], value)
    print(
        example,
        {
            key: result[key]
            for key in (
                "witness_found",
                "termination_reason",
                "selected_prefix_length",
                "response_evaluations",
                "total_iterations",
                "total_stages",
                "total_backtracking_attempts",
            )
        },
    )
