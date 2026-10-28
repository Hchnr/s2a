"""Exercise prefix repair against causal synthetic responses and solver boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import s2a.prefix_preserving_trajectory_plan as planning
from s2a.osc import OSCInterface


def response(commands):
    """A command changes this and subsequent endpoints, never earlier ones."""
    deltas = np.c_[commands, np.zeros(len(commands))]
    endpoints = np.cumsum(deltas, axis=0)
    starts = np.vstack((np.zeros(7), endpoints[:-1]))
    inner = starts[:, None, :] + (np.arange(1, 26) / 25)[None, :, None] * deltas[:, None, :]
    velocities = deltas / 0.05
    poses = np.tile(np.eye(4), (len(commands), 1, 1))
    poses[:, :3, 3] = endpoints[:, :3]
    poses[:, :3, :3] = Rotation.from_rotvec(endpoints[:, 3:6]).as_matrix()
    return dict(
        inner_joint_position=inner.reshape(-1, 7),
        inner_joint_velocity=np.repeat(velocities, 25, axis=0),
        joint_position=endpoints,
        joint_velocity=velocities,
        pose=poses,
        twist=velocities[:, :6],
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


def optimizer_result(x, *, nit=1):
    return SimpleNamespace(success=True, status=0, message="synthetic optimizer return", nit=nit, x=x.copy())


def test_repair_can_move_an_exact_prefix_within_original_caps_without_mutating_inputs():
    arguments = inputs()
    arguments["desired"][:, 0, 3] = [0.0, 0.275]
    before = deepcopy(arguments)
    result = planning.solve_prefix_preserving_trajectory_response(**arguments)

    assert result["initial_prefix_length"] == 1
    assert result["witness_found"] and result["selected_prefix_length"] == 2
    commands = result["selected"]["commands"]
    # Fixing u0=0, or fixing its original zero error, limits endpoint1 to .25.
    # With movable u0=.009, u1=.258, both endpoints and the .25 rate are feasible.
    assert commands[0, 0] >= 0.0075 - 1e-10
    assert 0 < result["selected"]["tcp_errors"][0, 0] ** 2 <= 0.01**2
    assert np.all(result["selected"]["normalized_squared_errors"] <= 1.0)
    assert result["selected"]["audit"]["feasible"]
    assert np.max(abs(np.diff(np.vstack((arguments["previous"], commands)), axis=0))) <= 0.25 + 1e-12
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(arguments[name], value)
    for name in ("input_min", "input_max", "output_min", "output_max"):
        np.testing.assert_array_equal(getattr(arguments["interface"], name), getattr(before["interface"], name))


def test_complete_warm_needs_one_response_and_no_optimizer_even_inside_safety_buffer(monkeypatch):
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        record = response(commands)
        # Within the existing terminal audit tolerance, outside optimizer buffer.
        record["joint_velocity"][-1, 6] = 5e-11
        record["inner_joint_velocity"][-25:, 6] = 5e-11
        return record

    def forbidden_optimizer(*_args, **_kwargs):
        pytest.fail("A fully audited warm witness must not invoke optimization")

    monkeypatch.setattr(planning, "minimize", forbidden_optimizer)
    arguments = inputs(predict)
    arguments["terminal_normals"] = np.eye(7)[6:]
    result = planning.solve_prefix_preserving_trajectory_response(**arguments)
    assert result["witness_found"] and result["selected"]["audit"]["feasible"]
    assert result["selected"]["audit"]["groups"]["terminal"]["minimum"] == -5e-11
    assert result["response_evaluations"] == len(calls) == 1
    assert result["total_iterations"] == 0 and result["phases"] == []


@pytest.mark.parametrize("slack", [-0.01, 5e-9])
def test_nonwitness_warm_requires_full_safety_and_optimizer_buffer(monkeypatch, slack):
    def predict(commands):
        record = response(commands)
        record["inner_joint_position"][32, 6] = 2.0 - 0.05 - slack
        return record

    def forbidden_optimizer(*_args, **_kwargs):
        pytest.fail("Prefix repair must not silently add a safety restoration phase")

    monkeypatch.setattr(planning, "minimize", forbidden_optimizer)
    arguments = inputs(predict)
    arguments["desired"][1, 0, 3] = 0.1
    with pytest.raises(ValueError):
        planning.solve_prefix_preserving_trajectory_response(**arguments)


@pytest.mark.parametrize("failure", ["prefix", "inner", "braking", "terminal", "rate"])
def test_optimizer_success_cannot_override_prefix_or_full_horizon_contract(monkeypatch, failure):
    attempts = []

    def predict(commands):
        record = response(commands)
        active = bool(np.any(commands))
        if failure == "inner" and active:
            record["inner_joint_position"][57, 6] = 1.99
        if failure == "braking" and active:
            record["joint_velocity"][1, 6] = 20.0
        if failure == "terminal":
            record["joint_velocity"][-1, 6] = 0.1 if active else -0.01
        return record

    arguments = inputs(predict, horizon=3)
    arguments["desired"][:, 0, 3] = [0, 0.1, 0.2]
    if failure == "terminal":
        arguments["terminal_normals"] = np.eye(7)[6:]
    if failure == "rate":
        arguments["desired"][:, 0, 3] = [0, 0.3, 0.2]

    def misleading_success(_objective, start, *, constraints, callback, **_kwargs):
        attempts.append(start.copy())
        final = start.copy()
        final[[0, 6, 12]] = [0.0, 0.1, 0.1]
        if failure == "prefix":
            final[[0, 6, 12]] = [0.04, 0.06, 0.1]
        elif failure == "rate":
            final[[0, 6, 12]] = [0.0, 0.3, -0.1]
        constraints[1]["fun"](final)
        callback(final)
        return optimizer_result(final, nit=3)

    monkeypatch.setattr(planning, "minimize", misleading_success)
    result = planning.solve_prefix_preserving_trajectory_response(**arguments)
    assert len(attempts) == len(result["phases"]) == 1
    assert result["phases"][0]["success"]
    assert result["total_iterations"] == 3
    assert not result["witness_found"] and result["selected_prefix_length"] == 1
    assert result["selected"]["audit"]["feasible"]
    assert np.all(result["selected"]["normalized_squared_errors"][:1] <= 1.0)
    rejected = result["evaluated_candidates"][-1]
    assert not rejected["feasible"]
    if failure != "prefix":
        np.testing.assert_allclose(rejected["tcp_errors"], 0.0, atol=1e-15)
        assert not rejected["joint_feasible"]


def test_disconnected_good_points_do_not_extend_prefix_or_restart_a_stalled_stage(monkeypatch):
    attempts = []

    def unchanged(_objective, start, **_kwargs):
        attempts.append(start.copy())
        return optimizer_result(start, nit=0)

    monkeypatch.setattr(planning, "minimize", unchanged)
    arguments = inputs(horizon=3)
    arguments["desired"][:, 0, 3] = [0, 0.02, 0]
    result = planning.solve_prefix_preserving_trajectory_response(**arguments)
    assert result["initial_prefix_length"] == result["selected_prefix_length"] == 1
    assert result["phases"][0]["target_cursor"] == 1
    assert len(attempts) == len(result["phases"]) == 1
    assert result["response_evaluations"] == 1 and result["total_iterations"] == 0
    assert not result["witness_found"]


@pytest.mark.parametrize("distance,expected_witness", [(0.01, True), (np.nextafter(0.01, np.inf), False)])
def test_pose_caps_do_not_inherit_optimizer_or_audit_tolerances(monkeypatch, distance, expected_witness):
    monkeypatch.setattr(planning, "minimize", lambda _objective, start, **_kwargs: optimizer_result(start))
    arguments = inputs()
    arguments["desired"][1, 0, 3] = distance
    result = planning.solve_prefix_preserving_trajectory_response(**arguments)
    assert result["witness_found"] is expected_witness
    assert result["selected_prefix_length"] == (2 if expected_witness else 1)


def test_full_fd_uses_box_contained_one_sided_steps_at_command_boundaries():
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return response(np.zeros_like(commands))

    arguments = inputs(predict)
    arguments["previous"][:2] = [0.9, -0.9]
    arguments["warm"][:, :2] = [1.0, -1.0]
    arguments["desired"][1, 0, 3] = 0.1
    result = planning.solve_prefix_preserving_trajectory_response(**arguments, max_evaluations=13)
    assert result["response_evaluations"] == len(calls) == 13
    commands = np.asarray(calls)
    assert np.all(commands >= -1.0) and np.all(commands <= 1.0)
    assert np.all(abs(commands[:, 0] - arguments["previous"]) <= 0.25)
    perturbations = commands[1:].reshape(12, 12) - arguments["warm"].ravel()
    np.testing.assert_allclose(abs(perturbations), np.eye(12) * 1e-4, atol=1e-15)
    assert perturbations[0, 0] < 0 and perturbations[1, 1] > 0
    assert perturbations[6, 6] < 0 and perturbations[7, 7] > 0
    assert not result["witness_found"]


def test_prefix_growth_cannot_reset_an_exhausted_iteration_budget(monkeypatch):
    attempts = []

    def exhaust_iterations(objective, start, *, callback, options, **_kwargs):
        attempts.append(start.copy())
        assert options["maxiter"] == 2
        callback(start)
        partial = start.copy()
        partial[6] = 0.1
        objective(partial)
        callback(partial)
        pytest.fail("Second callback should report prefix growth")

    monkeypatch.setattr(planning, "minimize", exhaust_iterations)
    arguments = inputs(horizon=3)
    arguments["desired"][:, 0, 3] = [0, 0.1, 0.2]
    result = planning.solve_prefix_preserving_trajectory_response(**arguments, max_iterations=2)
    assert len(attempts) == len(result["phases"]) == 1
    assert result["total_iterations"] == result["phases"][0]["callback_count"] == 2
    assert result["selected_prefix_length"] == 2 and not result["witness_found"]
    assert result["termination_reason"] == "iteration_budget"


@pytest.mark.parametrize("budget,interrupt", [(40, False), (38, True), (39, True)])
def test_stages_share_cache_fd_budget_and_callback_accounting(monkeypatch, budget, interrupt):
    calls, options_seen, jacobians = [], [], []

    def predict(commands):
        calls.append(commands.copy())
        record = response(commands)
        # A nonlinear terminal response makes positive probes around the .1
        # checkpoint unsafe, while the .0, .1 and .2 plans remain safe. Thus
        # the next stage must use the exact checkpoint whose FD is cached.
        total = commands.sum()
        terminal_velocity = 10 * (total - 0.1) * (0.2 - total) - 1e-8
        record["joint_velocity"][-1, 6] = terminal_velocity
        record["inner_joint_velocity"][-25:, 6] = terminal_velocity
        commands[:] = 999.0  # The predictor must not own the cached command array.
        return record

    def staged_optimizer(objective, start, *, constraints, callback, options, **_kwargs):
        phase = len(options_seen)
        options_seen.append(options.copy())
        nonlinear = constraints[1]
        objective(start)
        shifted_epigraph = start.copy()
        shifted_epigraph[-1] += 0.3
        change = nonlinear["fun"](shifted_epigraph) - nonlinear["fun"](start)
        np.testing.assert_allclose(change[:-2], 0.0, atol=1e-14)
        np.testing.assert_allclose(change[-2:], 0.3, atol=1e-12)
        jacobians.append(nonlinear["jac"](start))
        if phase == 0:
            # The regularizer includes the already-protected nonzero first error.
            assert objective(start) - start[-1] == pytest.approx(1e-4 * (0.25 + 100 + 400))
            assert len(calls) == 19
            first_repair = start.copy()
            first_repair[6], first_repair[-1] = 0.1, 0.0
            objective(first_repair)
            before_advance = nonlinear["jac"](first_repair)
            assert len(calls) == 38
            # Cache this same command's full derivative before changing stages.
            jacobians.append(before_advance)
            callback(first_repair)
            pytest.fail("A completed callback must advance a genuinely repaired prefix")
        assert phase == 1 and len(calls) == 38
        assert options["maxiter"] == options_seen[0]["maxiter"] - 1
        # Same command bytes, different target cursor: assembled rows must change.
        assert jacobians[-1].shape[0] == jacobians[-2].shape[0] + 2
        assert jacobians[-2][-2, 12] == 0.0
        assert jacobians[-1][-2, 12] == pytest.approx(1999.0, abs=1e-7)
        full = start.copy()
        full[12], full[-1] = 0.1, 0.0
        objective(full)
        if interrupt:
            another = full.copy()
            another[1] += 0.001
            objective(another)
            pytest.fail("The next unique response must respect the shared budget")
        callback(full)
        pytest.fail("The full witness must stop the phase at its callback")

    monkeypatch.setattr(planning, "minimize", staged_optimizer)
    arguments = inputs(predict, horizon=3)
    arguments["desired"][:, 0, 3] = [0.005, 0.1, 0.2]
    arguments["terminal_normals"] = np.eye(7)[6:]
    result = planning.solve_prefix_preserving_trajectory_response(**arguments, max_evaluations=budget, max_iterations=5)
    assert len(options_seen) == len(result["phases"]) == (1 if budget == 38 else 2)
    assert result["phases"][0]["iterations"] == result["phases"][0]["callback_count"] == 1
    assert result["phases"][0]["evaluations_after"] == 38
    if budget > 38:
        assert result["phases"][1]["evaluations_before"] == 38
    assert result["response_evaluations"] == len(calls) == len(result["evaluated_candidates"]) == min(budget, 39)
    assert len({command.tobytes() for command in calls}) == len(calls)
    assert all(command.shape == (3, 6) for command in calls)
    assert result["total_iterations"] == (1 if interrupt else 2)
    assert result["witness_found"] is (budget >= 39)
    assert result["selected_prefix_length"] == (3 if budget >= 39 else 2)
    if interrupt and budget > 38:
        assert result["phases"][1]["status"] == "response_budget"
    for candidate in result["evaluated_candidates"]:
        assert np.max(abs(candidate["commands"])) < 1.0
