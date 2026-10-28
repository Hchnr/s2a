"""Synthetic control-flow tests; these do not run a robot or real case data."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import s2a.prefix_callback_backtracking_plan as planning
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.osc import OSCInterface

solve = planning.solve_prefix_callback_backtracking_response


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


def curved_response(commands):
    """The endpoint target improves along x1, but the prefix bends outward."""
    value = response(commands)
    value["pose"][0, 0, 3] = 0.009 + 500.0 * commands[1, 0] ** 2
    return value


def test_real_slsqp_hard_feasible_example_reports_its_actual_finite_outcome():
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.045
    before = deepcopy(arguments)
    known = np.zeros((2, 6))
    known[1, 0] = 0.045
    known_rollout = response(known)
    np.testing.assert_array_equal(known_rollout["pose"], arguments["desired"])
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
    result = solve(**arguments)
    # Observed once with the preregistered parameters in native SciPy 1.15.3:
    # 102 responses / 7 iterations / 7 stages / 17 ray attempts, full witness.
    # Preserve that successful behavior without imposing backend-specific
    # iteration counts. This is a synthetic regression, not real-task evidence.
    assert result["selected"]["audit"]["feasible"]
    assert result["witness_found"] and np.all(result["selected"]["normalized_squared_errors"] <= 1)
    assert result["termination_reason"] == "witness" and result["selected_prefix_length"] == 2
    assert 0 < result["total_iterations"] <= 60
    assert 1 < result["response_evaluations"] <= 4000
    assert result["total_stages"] <= 60
    for anchor in result["accepted_anchors"][1:]:
        assert anchor["feasible"] or (
            anchor["joint_feasible"]
            and anchor["safety_min"] >= 1e-8
            and anchor["prefix_length"] >= anchor["stage_target_cursor"]
            and (anchor["prefix_length"] > anchor["stage_target_cursor"] or anchor["target_improvement"] > 1e-12)
        )
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(arguments[name], value)
    print(
        "hard toy:",
        {
            name: result[name]
            for name in (
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


def test_real_slsqp_strict_interior_example_finds_a_witness():
    arguments = inputs()
    arguments["desired"][:, 0, 3] = 0.015
    result = solve(**arguments)
    assert result["witness_found"] and result["termination_reason"] == "witness"
    assert result["selected_prefix_length"] == 2
    assert np.max(result["selected"]["normalized_squared_errors"]) < 0.5
    assert np.min(result["selected"]["safety_slack"]) > 1e-8
    print(
        "interior toy:", {name: result[name] for name in ("response_evaluations", "total_iterations", "total_stages")}
    )


def test_first_eligible_ray_point_preserves_curved_prefix_and_descends(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return curved_response(commands)

    def generate(_objective, start, *, callback, **_kwargs):
        trial = start.copy()
        trial[6] = 0.004
        callback(trial)
        pytest.fail("nonzero callback must interrupt direction generation")

    monkeypatch.setattr(planning, "minimize", generate)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_stages=1)
    phase = result["phases"][0]
    attempts = phase["ray_attempts"]
    assert [item["alpha"] for item in attempts] == [1, 0.5, 0.25]
    assert [item["eligible"] for item in attempts] == [False, False, True]
    assert all("prefix_lost" in item["rejection_reasons"] for item in attempts[:2])
    assert attempts[0]["outcome"] == "cache_hit"
    assert attempts[2]["selected"] and phase["ray_termination"] == "accepted"
    assert phase["status"] == "direction_captured" and not phase["optimizer_returned"]
    assert "optimizer_iterations" not in phase and phase["iterations"] == 1
    assert phase["radius_after"] == 0.01 and phase["prefix_after"] == 1
    assert result["accepted_anchors"][-1]["commands"][1, 0] == 0.001
    assert result["total_backtracking_new_responses"] == 2
    for item in attempts:
        assert item["entry"]["audit"]["feasible"]
        np.testing.assert_array_equal(item["entry"]["commands"], item["commands"])
        assert item["entry"]["rollout"]["inner_joint_position"].shape == (50, 7)
    assert_ledger(result, calls)


def test_fd_descent_is_final_candidate_but_cannot_be_an_ordinary_anchor(monkeypatch):
    def only_fd(_objective, start, *, jac, **_kwargs):
        jac(start)
        return optimizer_result(start)

    monkeypatch.setattr(planning, "minimize", only_fd)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_stages=1)
    phase = result["phases"][0]
    assert not phase["accepted"] and phase["ray_termination"] == "no_direction"
    assert phase["ray_attempts"] == [] and len(result["accepted_anchors"]) == 1
    assert result["selected"]["commands"][1, 0] == 1e-4
    assert not result["final_selected_is_last_anchor"]


def test_ray_anchor_is_first_eligible_even_when_fd_has_a_better_global_rank(monkeypatch):
    def direction(_objective, start, *, jac, callback, **_kwargs):
        jac(start)
        trial = start.copy()
        trial[6] = 0.00005
        callback(trial)

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_stages=1)
    assert result["accepted_anchors"][-1]["commands"][1, 0] == 0.00005
    assert result["selected"]["commands"][1, 0] == 0.0001
    assert not result["final_selected_is_last_anchor"]
    assert len(result["phases"][0]["ray_attempts"]) == 1


def test_zero_callback_counts_then_nonzero_capture_finishes_ray_at_iteration_limit(monkeypatch):
    def direction(_objective, start, *, callback, options, **_kwargs):
        assert options["maxiter"] == 2
        callback(start)
        trial = start.copy()
        trial[6] = 0.004
        callback(trial)
        pytest.fail("captured direction must interrupt")

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(curved_response)
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_iterations=2)
    phase = result["phases"][0]
    assert result["total_iterations"] == 2 and result["total_stages"] == 1
    assert result["termination_reason"] == "iteration_budget"
    assert phase["zero_callback_count"] == 1 and phase["callback_count"] == 2
    assert phase["ray_attempts"][-1]["alpha"] == 0.25 and phase["accepted"]
    assert not phase["optimizer_returned"] and "optimizer_iterations" not in phase
    capture = phase["captured_callback"]
    assert not capture["zero_step"] and capture["commands"][1, 0] == 0.004
    assert result["request_ledger"][capture["request_index"]]["role"] == "callback"


def test_all_zero_callbacks_use_real_iterations_and_do_not_create_ray(monkeypatch):
    def stationary(_objective, start, *, callback, options, **_kwargs):
        for _ in range(options["maxiter"]):
            callback(start)
        return optimizer_result(start, nit=options["maxiter"])

    monkeypatch.setattr(planning, "minimize", stationary)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_iterations=2)
    phase = result["phases"][0]
    assert phase["zero_callback_count"] == phase["callback_count"] == phase["iterations"] == 2
    assert phase["captured_callback"] is None and phase["ray_attempts"] == []
    assert phase["optimizer_returned"] and phase["optimizer_iterations"] == 2
    assert result["response_evaluations"] == 1 and result["termination_reason"] == "iteration_budget"


def test_alpha_one_preserves_trial_signed_zero_bytes_and_hits_callback_cache(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    def direction(_objective, start, *, callback, **_kwargs):
        trial = start.copy()
        trial[5] = -0.0
        trial[6] = 0.004
        callback(trial)

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_stages=1)
    phase = result["phases"][0]
    attempt = phase["ray_attempts"][0]
    assert np.signbit(attempt["commands"][0, 5])
    assert attempt["commands"].tobytes() == phase["captured_callback"]["commands"].tobytes()
    assert attempt["outcome"] == "cache_hit" and result["response_evaluations"] == 2
    assert_ledger(result, calls)


@pytest.mark.parametrize("failure", ["prefix", "inner", "braking", "terminal", "rate", "buffer"])
def test_all_fixed_ray_points_are_audited_before_acceptance(monkeypatch, failure):
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

    def direction(_objective, start, *, callback, **_kwargs):
        trial = start.copy()
        trial[6] += 0.002
        callback(trial)

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    if failure in ("terminal", "buffer"):
        arguments["terminal_normals"] = np.eye(7)[6:]
    if failure == "rate":
        arguments["warm"][1, 0] = 0.25
        arguments["desired"][1, 0, 3] = 0.3
    result = solve(**arguments, max_stages=1)
    phase = result["phases"][0]
    assert not phase["accepted"] and phase["ray_termination"] == "alphas_exhausted"
    assert len(phase["ray_attempts"]) == 9 and len(result["accepted_anchors"]) == 1
    reason = {"prefix": "prefix_lost", "buffer": "safety_buffer"}.get(failure, "joint_or_command")
    assert all(reason in item["rejection_reasons"] for item in phase["ray_attempts"])
    assert all(item["entry"] is not None and not item["selected"] for item in phase["ray_attempts"])


def test_three_failed_radii_stop_without_result_driven_extra_trials(monkeypatch):
    monkeypatch.setattr(planning, "minimize", lambda _fun, start, **_kw: optimizer_result(start))
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments)
    assert result["termination_reason"] == "trust_region_stalled"
    assert [phase["radius"] for phase in result["phases"]] == [0.01, 0.005, 0.0025]
    assert [phase["radius_after"] for phase in result["phases"]] == [0.005, 0.0025, None]
    assert result["total_backtracking_attempts"] == 0 and result["response_evaluations"] == 1


def test_shared_budget_rejects_unfinished_ray_without_fabricating_an_audit(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return curved_response(commands)

    def direction(_fun, start, *, jac, callback, **_kw):
        jac(start)
        trial = start.copy()
        trial[6] = 0.004
        callback(trial)

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_evaluations=14)
    phase = result["phases"][0]
    assert result["termination_reason"] == phase["ray_termination"] == "response_budget"
    assert phase["status"] == "direction_captured" and phase["iterations"] == 1
    assert [item["outcome"] for item in phase["ray_attempts"]] == ["cache_hit", "budget_rejected"]
    denied = phase["ray_attempts"][-1]
    assert denied["entry"] is denied["response_index"] is denied["new_response_index"] is None
    assert denied["rejection_reasons"] == ["response_budget"] and not denied["eligible"]
    assert result["total_backtracking_attempts"] == 2 and result["total_backtracking_new_responses"] == 0
    assert_ledger(result, calls)


def test_last_available_response_on_ray_can_be_a_complete_witness(monkeypatch):
    def predict(commands):
        value = response(commands)
        value["pose"][0, 0, 3] = 0.009 + 200 * commands[1, 0] ** 2
        return value

    def direction(_fun, start, *, jac, callback, **_kw):
        jac(start)
        trial = start.copy()
        trial[6] = 0.004
        callback(trial)

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.0115
    result = solve(**arguments, max_evaluations=15)
    assert result["witness_found"] and result["response_evaluations"] == 15
    assert result["termination_reason"] == "witness"
    phase = result["phases"][0]
    assert phase["ray_termination"] == "witness" and phase["acceptance_reason"] == "witness"
    assert [attempt["alpha"] for attempt in phase["ray_attempts"]] == [1, 0.5]
    assert phase["ray_attempts"][-1]["entry"]["feasible"]


def test_last_fd_witness_survives_generation_budget_and_needs_no_extra_buffer(monkeypatch):
    def predict(commands):
        value = response(np.zeros_like(commands))
        value["pose"][-1, 0, 3] = commands[-1, -1]
        value["joint_velocity"][-1, -1] = 5e-11 if commands[-1, -1] > 0 else -0.01
        return value

    def exhaust(objective, start, *, jac, **_kw):
        jac(start)
        candidate = start.copy()
        candidate[0] = 0.005
        objective(candidate)
        pytest.fail("budget denial must interrupt generation")

    monkeypatch.setattr(planning, "minimize", exhaust)
    arguments = inputs(predict)
    arguments["desired"][-1, 0, 3] = 0.01005
    arguments["terminal_normals"] = np.eye(7)[6:]
    result = solve(**arguments, max_evaluations=13)
    assert result["witness_found"] and result["response_evaluations"] == 13
    assert np.min(result["selected"]["safety_slack"]) == -5e-11
    phase = result["phases"][0]
    assert phase["status"] == "response_budget" and phase["ray_termination"] == "witness"
    assert phase["captured_callback"] is None and phase["ray_attempts"] == []
    assert phase["anchor_after_index"] == 12 and phase["acceptance_reason"] == "witness"


def test_callback_budget_denial_is_counted_without_a_fake_capture(monkeypatch):
    def exhaust(_fun, start, *, jac, callback, **_kw):
        jac(start)
        trial = start.copy()
        trial[6] = 0.004
        callback(trial)

    monkeypatch.setattr(planning, "minimize", exhaust)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_evaluations=13)
    phase = result["phases"][0]
    assert phase["callback_count"] == phase["iterations"] == 1
    assert phase["zero_callback_count"] == 0 and phase["captured_callback"] is None
    assert phase["status"] == phase["ray_termination"] == "response_budget"
    assert result["request_ledger"][-1]["role"] == "callback"
    assert result["request_ledger"][-1]["outcome"] == "budget_rejected"


def test_derivative_cache_resets_between_stages_with_different_one_sided_bounds(monkeypatch):
    def predict(commands):
        return response(np.zeros_like(commands))

    def probe(_fun, start, *, jac, constraints, **_kw):
        point = start.copy()
        point[0] = 0.005
        jac(point)
        constraints[1]["jac"](point)
        return optimizer_result(start)

    monkeypatch.setattr(planning, "minimize", probe)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_stages=2)
    probes = [row for row in result["request_ledger"] if row["role"] == "gradient_probe" and row["axis"] == 0]
    assert [row["nominal_delta"] for row in probes] == [1e-4, -1e-4]
    assert result["response_evaluations"] == 15
    assert not any(row["role"] == "jacobian_probe" for row in result["request_ledger"])


@pytest.mark.parametrize("cap,complete", [(0.01, True), (np.nextafter(0.01, np.inf), False)])
def test_pose_caps_remain_strict_at_the_boundary(monkeypatch, cap, complete):
    monkeypatch.setattr(planning, "minimize", lambda _fun, start, **_kw: optimizer_result(start))
    arguments = inputs()
    arguments["desired"][1, 0, 3] = cap
    result = solve(**arguments, max_stages=1)
    assert result["witness_found"] is complete
    assert result["response_evaluations"] == 1
    if complete:
        assert result["total_iterations"] == result["total_stages"] == result["total_backtracking_attempts"] == 0


def test_unsafe_incomplete_warm_is_rejected_before_optimization(monkeypatch):
    def unsafe(commands):
        value = response(commands)
        value["inner_joint_position"][0, 6] = 1.99
        return value

    monkeypatch.setattr(planning, "minimize", lambda *_a, **_k: pytest.fail("unsafe warm must not optimize"))
    arguments = inputs(unsafe)
    arguments["desired"][1, 0, 3] = 0.03
    with pytest.raises(ValueError, match="Incomplete warm"):
        solve(**arguments)


def test_failed_rays_share_exact_response_cache_across_all_three_stages(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        value = response(commands)
        if commands[1, 0] != 0:
            value["pose"][0, 0, 3] = 0.011
        return value

    def direction(_fun, start, *, callback, **_kw):
        trial = start.copy()
        trial[6] = 0.002
        callback(trial)

    monkeypatch.setattr(planning, "minimize", direction)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments)
    assert result["termination_reason"] == "trust_region_stalled"
    assert result["total_backtracking_attempts"] == 27
    assert result["total_backtracking_new_responses"] == 8
    assert result["response_evaluations"] == 10 and result["total_iterations"] == 3
    for phase in result["phases"][1:]:
        assert all(item["outcome"] == "cache_hit" for item in phase["ray_attempts"])
        assert phase["evaluations_before"] == phase["evaluations_after"] == 10
    assert_ledger(result, calls)


def test_normal_final_response_cannot_supply_ordinary_anchor_without_callback(monkeypatch):
    def returned(_fun, start, **_kw):
        candidate = start.copy()
        candidate[6] = 0.004
        return optimizer_result(candidate, nit=1)

    monkeypatch.setattr(planning, "minimize", returned)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.03
    result = solve(**arguments, max_stages=1)
    phase = result["phases"][0]
    assert phase["optimizer_returned"] and phase["optimizer_iterations"] == 1
    assert phase["callback_count"] == 0 and phase["captured_callback"] is None
    assert phase["ray_termination"] == "no_direction" and not phase["accepted"]
    assert result["selected"]["commands"][1, 0] == 0.004
    assert not result["final_selected_is_last_anchor"]
