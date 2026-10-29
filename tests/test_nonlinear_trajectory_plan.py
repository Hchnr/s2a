"""Check full-trajectory feasibility against causal synthetic robot responses."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import s2a.nonlinear_trajectory_plan as planning
from s2a.osc import OSCInterface


def response(commands, translation_scale=0.1):
    joint_deltas = np.c_[commands, np.zeros(len(commands))]
    endpoints = np.cumsum(joint_deltas, axis=0)
    starts = np.vstack((np.zeros(7), endpoints[:-1]))
    fractions = np.arange(1, 26) / 25.0
    inner = starts[:, None, :] + fractions[None, :, None] * joint_deltas[:, None, :]
    velocities = joint_deltas / 0.05
    poses = np.tile(np.eye(4), (len(commands), 1, 1))
    poses[:, :3, 3] = translation_scale * endpoints[:, :3]
    poses[:, :3, :3] = Rotation.from_rotvec(endpoints[:, 3:6]).as_matrix()
    return dict(
        inner_joint_position=inner.reshape(-1, 7),
        inner_joint_velocity=np.repeat(velocities, 25, axis=0),
        joint_position=endpoints,
        joint_velocity=velocities,
        pose=poses,
        twist=np.c_[translation_scale * velocities[:, :3], velocities[:, 3:6]],
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


def test_multiple_periods_track_translation_and_rotation_with_tcp_offset():
    arguments = inputs(horizon=3)
    target_commands = np.tile([0.16, -0.07, 0.05, 0.07, 0.04, -0.06], (3, 1))
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    offset[:3, 3] = [0.05, -0.04, 0.03]
    arguments.update(desired=response(target_commands)["pose"] @ offset, site_from_tcp=offset)
    result = planning.solve_trajectory_response(**arguments)
    assert result["witness_found"] and result["pose_caps_horizon"] == 3
    assert result["all_pose_caps_enforced"]
    selected = result["selected"]
    tcp = response(selected["commands"])["pose"] @ offset
    assert np.all(np.linalg.norm(arguments["desired"][:, :3, 3] - tcp[:, :3, 3], axis=1) <= 0.01)
    rotation_error = arguments["desired"][:, :3, :3] @ np.swapaxes(tcp[:, :3, :3], 1, 2)
    assert np.all(Rotation.from_matrix(rotation_error).magnitude() <= 0.05)
    assert selected["rollout"]["pose"][-1, 0, 3] > 0.035
    changes = np.diff(np.vstack((arguments["previous"], selected["commands"])), axis=0)
    assert np.max(abs(changes)) <= 0.25 + 1e-12
    assert result["max_evaluations"] == 4000 and result["response_evaluations"] <= 4000
    assert result["total_iterations"] <= 60


def test_future_rotation_failure_rejects_plan_even_with_exact_first_pose():
    def predict(commands):
        record = response(np.zeros_like(commands))
        return record

    arguments = inputs(predict)
    arguments["desired"][1, :3, :3] = Rotation.from_rotvec([0, 0, 0.2]).as_matrix()
    result = planning.solve_trajectory_response(**arguments)
    assert not result["witness_found"]
    assert result["selected"]["audit"]["feasible"]
    np.testing.assert_array_equal(result["selected"]["tcp_errors"][0], np.zeros(6))
    assert np.linalg.norm(result["selected"]["tcp_errors"][1, 3:]) == pytest.approx(0.2)


def test_interperiod_rate_limit_cannot_be_expanded_to_hit_unreachable_future_pose():
    def predict(commands):
        return response(commands, translation_scale=1.0)

    arguments = inputs(predict)
    arguments["desired"][0, 0, 3] = 0.2
    result = planning.solve_trajectory_response(**arguments)
    assert not result["witness_found"]
    selected = result["selected"]
    assert selected["audit"]["feasible"]
    changes = np.diff(np.vstack((arguments["previous"], selected["commands"])), axis=0)
    assert np.max(abs(changes)) <= 0.25 + 1e-12
    assert np.max(np.linalg.norm(selected["tcp_errors"][:, :3], axis=1)) > 0.01
    assert selected["commands"][1, 0] - selected["commands"][0, 0] == pytest.approx(-0.25, abs=1e-7)


def test_terminal_velocity_is_restored_only_at_final_period():
    def predict(commands):
        record = response(commands)
        record["joint_velocity"][:, 6] = 2.0
        record["joint_velocity"][-1, 6] += 10.0 * commands[-1, 0]
        record["inner_joint_velocity"][:, 6] = np.repeat(record["joint_velocity"][:, 6], 25)
        return record

    arguments = inputs(predict)
    arguments["terminal_normals"] = np.eye(7)[6:]
    arguments["desired"][-1, 0, 3] = -0.02
    result = planning.solve_trajectory_response(**arguments)
    assert not result["initial"]["audit"]["feasible"]
    assert [phase["phase"] for phase in result["phases"]] == ["safety", "pose"]
    assert result["witness_found"]
    velocity = result["selected"]["rollout"]["joint_velocity"]
    assert velocity[0, 6] == 2.0
    assert velocity[-1, 6] <= 1e-10
    assert result["selected"]["commands"][-1, 0] <= -0.2 + 1e-10


def test_each_period_braking_projection_is_enforced_during_safety_recovery():
    def predict(commands):
        record = response(commands)
        record["inner_joint_position"][:, 6] = 0.8
        record["joint_position"][:, 6] = 0.8
        record["joint_velocity"][:, 6] = 2.0 + 10.0 * commands[:, 0]
        record["inner_joint_velocity"][:, 6] = np.repeat(record["joint_velocity"][:, 6], 25)
        return record

    arguments = inputs(predict)
    arguments["upper"][6] = 1.0
    arguments["desired"][:, 0, 3] = [-0.01, -0.02]
    result = planning.solve_trajectory_response(**arguments)
    assert result["phases"][0]["phase"] == "safety" and result["witness_found"]
    selected = result["selected"]
    rollout = selected["rollout"]
    braking = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
    assert np.all(braking <= arguments["upper"] - 0.05 + 1e-10)
    assert np.all(braking >= arguments["lower"] + 0.05 - 1e-10)
    assert np.all(selected["commands"][:, 0] <= -0.05 + 1e-9)


def test_inner_excursion_is_checked_even_when_all_endpoints_are_safe():
    def predict(commands):
        record = response(commands)
        record["inner_joint_position"][8, 6] = 0.98 + commands[0, 1]
        return record

    arguments = inputs(predict)
    arguments["upper"][6] = 1.0
    result = planning.solve_trajectory_response(**arguments)
    assert not result["initial"]["audit"]["feasible"]
    assert result["initial"]["audit"]["groups"]["braking_upper"]["minimum"] > 0
    assert result["witness_found"]
    selected = result["selected"]
    assert selected["commands"][0, 1] <= -0.03 + 1e-9
    assert np.all(selected["rollout"]["inner_joint_position"] <= arguments["upper"] - 0.05 + 1e-10)


def test_profile_uses_inner_time_for_first_period_and_history_peak_for_future():
    def predict(commands):
        record = response(commands)
        record["inner_joint_position"][:25, 6] = np.linspace(0.88, 0.8, 25)
        record["inner_joint_position"][25:, 6] = 0.88 + commands[1, 0]
        record["joint_position"][:, 6] = record["inner_joint_position"][[24, 49], 6]
        return record

    arguments = inputs(predict)
    arguments["upper"][6] = 1.0
    arguments["past_inner_error_rad"][:, 6] = np.linspace(0, 0.1, 25)
    before = arguments["past_inner_error_rad"].copy()
    result = planning.solve_trajectory_response(**arguments)
    assert not result["initial"]["audit"]["feasible"]
    assert result["witness_found"]
    selected = result["selected"]
    q = selected["rollout"]["inner_joint_position"]
    assert q[0, 6] == 0.88
    assert selected["commands"][1, 0] <= -0.03 + 1e-9
    assert np.all(q[:25] <= arguments["upper"] - 0.05 - before + 1e-10)
    assert np.all(q[25:] <= arguments["upper"] - 0.05 - before.max(axis=0) + 1e-10)
    np.testing.assert_array_equal(arguments["past_inner_error_rad"], before)
    np.testing.assert_array_equal(result["past_inner_error_rad"], before)


def test_boundary_finite_differences_respect_box_and_first_command_rate():
    calls = []

    def predict(commands):
        calls.append(commands.copy())
        return response(np.zeros_like(commands))

    arguments = inputs(predict)
    arguments["previous"][:2] = [0.9, -0.9]
    arguments["warm"][:, :2] = [1.0, -1.0]
    arguments["desired"][:, 0, 3] = 1.0
    result = planning.solve_trajectory_response(**arguments, max_evaluations=13)
    assert result["response_evaluations"] == len(calls) == 13
    commands = np.asarray(calls)
    assert np.all(commands >= -1.0) and np.all(commands <= 1.0)
    assert np.all(abs(commands[:, 0] - arguments["previous"]) <= 0.25)
    perturbations = commands[1:].reshape(12, 12) - arguments["warm"].ravel()
    np.testing.assert_allclose(abs(perturbations), np.eye(12) * 1e-4, atol=1e-15)
    assert perturbations[0, 0] < 0 and perturbations[1, 1] > 0
    assert perturbations[6, 6] < 0 and perturbations[7, 7] > 0


@pytest.mark.parametrize("failure", ["future_pose", "inner_safety", "interperiod_rate"])
def test_optimizer_success_never_substitutes_for_complete_trajectory_audit(monkeypatch, failure):
    def misleading_success(_objective, start, *, constraints, **_kwargs):
        final = start.copy()
        if failure == "interperiod_rate":
            final[6] = 0.3
            constraints[1]["fun"](final)
        return SimpleNamespace(success=True, status=0, message="claimed success", nit=1, x=final)

    def predict(commands):
        record = response(np.zeros_like(commands))
        if failure == "inner_safety":
            record["inner_joint_position"][32, 6] = 1.99
        elif failure == "interperiod_rate":
            record["pose"][1, 0, 3] = commands[1, 0]
        return record

    arguments = inputs(predict)
    if failure == "future_pose":
        arguments["desired"][1, :3, :3] = Rotation.from_rotvec([0, 0, 0.08]).as_matrix()
    elif failure == "interperiod_rate":
        arguments["desired"][1, 0, 3] = 0.3
    monkeypatch.setattr(planning, "minimize", misleading_success)
    result = planning.solve_trajectory_response(**arguments)
    assert all(phase["success"] for phase in result["phases"])
    assert not result["witness_found"]
    if failure == "inner_safety":
        assert result["selected"] is None
    else:
        assert result["selected"]["audit"]["feasible"]
    if failure == "interperiod_rate":
        exact_pose_probe = result["evaluated_candidates"][-1]
        np.testing.assert_array_equal(exact_pose_probe["tcp_errors"], np.zeros((2, 6)))
        assert not exact_pose_probe["joint_feasible"] and not exact_pose_probe["feasible"]


def test_safety_and_pose_phases_share_budget_for_all_full_trajectory_queries(monkeypatch):
    calls, phases_entered = [], []

    def predict(commands):
        calls.append(commands.copy())
        record = response(commands)
        record["inner_joint_position"][31, 6] = 1.98 + commands[1, 0]
        return record

    def requesting_optimizer(objective, start, *, constraints, **_kwargs):
        phases_entered.append(start.copy())
        constraints[1]["jac"](start)
        final = start.copy()
        if len(phases_entered) == 1:
            final[6], final[12] = -0.1, 0.0
        else:
            final[1] = 0.01
        objective(final)
        constraints[1]["fun"](final)
        return SimpleNamespace(success=True, status=0, message="requested evaluations", nit=1, x=final)

    monkeypatch.setattr(planning, "minimize", requesting_optimizer)
    arguments = inputs(predict)
    arguments["desired"][:, 0, 3] = 1.0
    result = planning.solve_trajectory_response(**arguments, max_evaluations=26)
    assert len(phases_entered) == 2
    assert result["response_evaluations"] == len(calls) == len(result["evaluated_candidates"]) == 26
    assert result["phases"][0]["evaluations_after"] == 14
    assert result["phases"][1]["evaluations_before"] == 14
    assert result["phases"][1]["status"] == "response_budget"
    assert not result["witness_found"] and result["selected"]["audit"]["feasible"]
