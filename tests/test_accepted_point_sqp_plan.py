"""Accepted-point SQP control-flow checks on synthetic CPU responses only."""

from copy import deepcopy
from inspect import signature

import numpy as np
import pytest
from test_prefix_optimization_reserve_plan import inputs, response

import s2a.accepted_point_sqp_plan as planning

solve = planning.solve_accepted_point_sqp_response


def test_real_qp_finds_full_caps_witness_without_hidden_response_queries(monkeypatch):
    actual_qp = planning.solve_epigraph_qp
    calls, algebra_counts = [], []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    def inspect_qp(*args, **kwargs):
        before = len(calls)
        result = actual_qp(*args, **kwargs)
        algebra_counts.append((before, len(calls)))
        return result

    monkeypatch.setattr(planning, "solve_epigraph_qp", inspect_qp)
    arguments = inputs(predict)
    arguments["desired"][:, 0, 3] = 0.015
    before = deepcopy(arguments)
    result = solve(**arguments, max_stages=5)
    assert result["witness_found"] and result["termination_reason"] == "witness"
    assert result["selected_prefix_length"] == 2
    assert result["selected"]["audit"]["feasible"]
    predicted = response(result["selected"]["commands"])
    assert np.all(np.linalg.norm(predicted["pose"][:, :3, 3] - arguments["desired"][:, :3, 3], axis=1) <= 0.01)
    assert np.all(result["selected"]["normalized_squared_errors"] <= 1)
    assert algebra_counts and all(start == end for start, end in algebra_counts)
    assert len(calls) == result["response_evaluations"]
    assert len(calls) == sum(item["outcome"] == "new_response" for item in result["request_ledger"])
    assert all(stage["qp"]["qualified"] for stage in result["phases"])
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(arguments[name], value)


def test_real_backtracking_uses_accepted_displacement_and_same_dual_at_both_ends(monkeypatch):
    actual_qp = planning.solve_epigraph_qp
    quadratic_problems = []

    def capture_qp(B, g, A, b, initial_t, **kwargs):
        result = actual_qp(B, g, A, b, initial_t, **kwargs)
        quadratic_problems.append(dict(g=g.copy(), A=A.copy(), result=result))
        return result

    def nonlinear(commands):
        value = response(commands)
        value["pose"][:] = np.eye(4)
        u = commands[0, 0]
        value["pose"][0, 0, 3] = u - 1000 * u**2
        return value

    monkeypatch.setattr(planning, "solve_epigraph_qp", capture_qp)
    arguments = inputs(nonlinear, horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    result = solve(**arguments, max_stages=2)
    first, second = result["phases"]
    selected = [attempt for attempt in first["ray_attempts"] if attempt["selected"]]
    assert len(selected) == 1 and selected[0]["alpha"] < 1
    assert any(not attempt["eligible"] for attempt in first["ray_attempts"])
    assert first["acceptance_reason"] == "target_descent"
    update = second["curvature_update"]
    assert update["updated"] and update["source_stage"] == 0
    accepted_displacement = (first["selected_commands"] - first["initial_commands"]).ravel()
    np.testing.assert_array_equal(update["displacement"], accepted_displacement)
    np.testing.assert_allclose(
        update["displacement"],
        selected[0]["alpha"] * first["qp_direction"]["displacement"].ravel(),
        rtol=1e-12,
        atol=1e-15,
    )
    assert not np.array_equal(update["displacement"], first["qp_direction"]["displacement"].ravel())
    old, new = quadratic_problems
    rows = first["linearization"]["nonlinear_row_count"]
    old_dual = old["result"]["multipliers"][:rows]
    old_gradient = old["g"] - old["A"][:rows, :-1].T @ old_dual
    new_gradient = new["g"] - new["A"][:rows, :-1].T @ old_dual
    np.testing.assert_allclose(update["previous_lagrangian_gradient"], old_gradient, atol=1e-12)
    np.testing.assert_allclose(update["current_lagrangian_gradient"], new_gradient, atol=1e-12)
    np.testing.assert_allclose(update["gradient_change"], new_gradient - old_gradient, atol=1e-12)
    assert not np.allclose(update["gradient_change"], new["g"] - old["g"])
    np.testing.assert_array_equal(second["hessian_after_update"], update["matrix"])


@pytest.mark.parametrize("mode", ["rejected", "zero", "uncertified"])
def test_unaccepted_qp_directions_do_not_create_curvature_pairs(monkeypatch, mode):
    def qp(B, g, A, b, initial_t, **kwargs):
        direction = np.zeros(len(g))
        direction[0] = -0.001 if mode == "rejected" else 0
        return dict(
            qualified=mode != "uncertified",
            solver_nit=1,
            point=np.r_[direction, initial_t],
            multipliers=np.ones(len(b)),
        )

    monkeypatch.setattr(planning, "solve_epigraph_qp", qp)
    monkeypatch.setattr(planning, "damped_bfgs", lambda *_a, **_kw: pytest.fail("no accepted pair exists"))
    arguments = inputs(horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    result = solve(**arguments)
    assert not result["witness_found"] and result["termination_reason"] == "trust_region_stalled"
    assert len(result["accepted_anchors"]) == 1 and result["pending_pair_source_stage"] is None
    assert [stage["radius"] for stage in result["phases"]] == list(planning.RADII)
    for stage in result["phases"]:
        assert not stage["accepted"]
        assert stage["curvature_update"]["reason"] == "no_pending_pair"
        np.testing.assert_array_equal(stage["hessian_after_stage"], np.eye(6))
        if mode == "rejected":
            assert len(stage["ray_attempts"]) == 9
            assert all("no_target_descent" in ray["rejection_reasons"] for ray in stage["ray_attempts"])
        else:
            assert stage["ray_attempts"] == []
            assert stage["status"] == ("qp_zero_direction" if mode == "zero" else "qp_certificate_failed")


def test_cursor_growth_discards_previous_target_curvature(monkeypatch):
    steps = iter([0.005, 0.005, 0.01, 0.0])

    def prescribed_step(B, g, A, b, initial_t, **kwargs):
        # Only stage progression is prescribed; response audit and BFGS remain real.
        direction = np.zeros(len(g))
        direction[0] = next(steps)
        return dict(qualified=True, solver_nit=1, point=np.r_[direction, initial_t], multipliers=np.ones(len(b)))

    monkeypatch.setattr(planning, "solve_epigraph_qp", prescribed_step)
    arguments = inputs()
    arguments["desired"][:, 0, 3] = [0.029, 0.1]
    result = solve(**arguments, max_stages=4)
    before_growth, growth, next_cursor = result["phases"][1:]
    assert before_growth["curvature_update"]["updated"]
    assert not np.allclose(growth["hessian_after_update"], np.eye(12))
    assert growth["prefix_before"] == 0 and growth["prefix_after"] == 1
    assert growth["acceptance_reason"] == "prefix_growth"
    assert growth["curvature_reset"] == "cursor_changed"
    assert growth["pending_pair_after_stage"] is None
    np.testing.assert_array_equal(growth["hessian_after_stage"], np.eye(12))
    assert next_cursor["target_cursor"] == 1
    assert next_cursor["curvature_update"]["reason"] == "no_pending_pair"
    assert next_cursor["pending_pair_source_stage"] is None
    np.testing.assert_array_equal(next_cursor["hessian_before"], np.eye(12))


def test_response_budget_mid_stencil_keeps_pending_pair_without_updating(monkeypatch):
    monkeypatch.setattr(planning, "damped_bfgs", lambda *_a, **_kw: pytest.fail("stencil is incomplete"))
    arguments = inputs(horizon=1)
    arguments["desired"][0, 0, 3] = 0.03
    result = solve(**arguments, max_evaluations=9)
    first, second = result["phases"]
    assert first["acceptance_reason"] == "target_descent" and first["qp"]["qualified"]
    assert second["pending_pair_source_stage"] == 0
    assert second["status"] == "response_budget"
    assert not second["linearization"]["complete"]
    assert second["curvature_update"] == dict(updated=False, reason="incomplete_stencil")
    assert second["qp"] is None and second["ray_attempts"] == []
    assert result["pending_pair_source_stage"] == 0
    assert result["response_evaluations"] == 9
    assert result["request_ledger"][-1]["outcome"] == "budget_rejected"
    np.testing.assert_array_equal(result["final_hessian"], np.eye(6))


def test_fd_full_witness_precedes_qp_and_does_not_require_prefix_reserve(monkeypatch):
    near_cap = 0.01 * np.sqrt(1 - planning.PREFIX_OPTIMIZATION_MARGIN / 2)

    def predict(commands):
        value = response(commands)
        value["pose"][0, 0, 3] = near_cap
        return value

    monkeypatch.setattr(planning, "solve_epigraph_qp", lambda *_a, **_kw: pytest.fail("FD found a full witness"))
    monkeypatch.setattr(planning, "damped_bfgs", lambda *_a, **_kw: pytest.fail("witness takes priority"))
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = near_cap + 1e-4
    result = solve(**arguments, max_evaluations=13)
    assert result["witness_found"] and result["termination_reason"] == "witness"
    assert result["response_evaluations"] == 13
    stage = result["phases"][0]
    assert stage["linearization"]["complete"] and stage["status"] == "witness_found"
    assert stage["qp"] is None and stage["ray_attempts"] == []
    assert np.max(result["selected"]["normalized_squared_errors"][0]) > 1 - planning.PREFIX_OPTIMIZATION_MARGIN
    assert np.all(result["selected"]["normalized_squared_errors"] <= 1)


def test_public_input_contract_matches_frozen_reserve_planner():
    from s2a.prefix_optimization_reserve_plan import solve_prefix_optimization_reserve_response

    assert signature(solve) == signature(solve_prefix_optimization_reserve_response)
