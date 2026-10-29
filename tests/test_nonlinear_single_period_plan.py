"""Exercise command feasibility against explicit inner states and SE(3) targets."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import s2a.nonlinear_single_period_plan as planning
from s2a.osc import OSCInterface


def pose(command, translation_scale=0.1):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(command[3:]).as_matrix()
    transform[:3, 3] = translation_scale * command[:3]
    return transform


def response(command):
    return dict(
        joint_position=np.tile(np.r_[command, 0.0], (25, 1)),
        joint_velocity=np.zeros((25, 7)),
        endpoint_pose=pose(command),
    )


def inputs(predict=response):
    return dict(
        predict=predict,
        previous=np.zeros(6),
        warm=np.zeros(6),
        interface=OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6)),
        lower=-np.ones(7),
        upper=np.ones(7),
        desired=np.eye(4),
        site_from_tcp=np.eye(4),
        past_inner_error_rad=np.zeros((25, 7)),
    )


def test_reachable_translation_rotation_and_tcp_offset_are_checked_together():
    arguments = inputs()
    target_command = np.array([0.14, -0.12, 0.09, 0.17, -0.15, 0.11])
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    offset[:3, 3] = [0.07, -0.03, 0.04]
    arguments.update(desired=pose(target_command) @ offset, site_from_tcp=offset)
    result = planning.solve_single_period_response(**arguments)
    assert result["witness_found"]
    selected = result["selected"]
    actual_tcp = pose(selected["command"]) @ offset
    assert np.linalg.norm(actual_tcp[:3, 3] - arguments["desired"][:3, 3]) <= 0.01
    rotation_difference = arguments["desired"][:3, :3] @ actual_tcp[:3, :3].T
    assert Rotation.from_matrix(rotation_difference).magnitude() <= 0.05
    assert np.max(abs(selected["command"])) <= 0.25
    assert result["max_evaluations"] == 600
    assert result["response_evaluations"] <= 600 and result["total_iterations"] <= 60


def test_recovery_from_unsafe_braking_projection_precedes_pose_search():
    def predict(command):
        record = response(command)
        record["joint_position"][:, 0] = 0.8
        record["joint_velocity"][:, 0] = 2.0 + 10.0 * command[0]
        return record

    arguments = inputs(predict)
    arguments["desired"] = pose(np.array([-0.1, 0, 0, 0, 0, 0]))
    result = planning.solve_single_period_response(**arguments)
    assert not result["initial"]["audit"]["feasible"]
    assert [phase["phase"] for phase in result["phases"]] == ["safety", "pose"]
    assert result["phases"][1]["initial_safety_min"] >= -1e-10
    assert result["witness_found"]
    selected = result["selected"]
    predicted = selected["response"]
    braking = predicted["joint_position"][-1] + 0.1 * predicted["joint_velocity"][-1]
    assert np.all(braking <= arguments["upper"] - 0.05 + 1e-10)
    assert np.all(braking >= arguments["lower"] + 0.05 - 1e-10)
    assert selected["command"][0] <= -0.05 + 1e-9


def test_joint_excursion_inside_period_must_be_corrected_even_when_endpoint_is_safe():
    def predict(command):
        record = response(command)
        record["joint_position"][9, 6] = 0.98 + command[0]
        return record

    result = planning.solve_single_period_response(**inputs(predict))
    assert result["initial"]["audit"]["groups"]["braking_upper"]["minimum"] > 0
    assert not result["initial"]["audit"]["feasible"]
    assert result["witness_found"]
    selected = result["selected"]
    assert selected["command"][0] <= -0.03 + 1e-9
    assert np.max(selected["response"]["joint_position"]) <= 0.95 + 1e-10


def test_causal_inner_error_profile_tightens_each_sample_without_mutating_history():
    def predict(command):
        record = response(command)
        record["joint_position"][:, 6] = 0.9 + 0.1 * command[0]
        record["endpoint_pose"] = pose(command, translation_scale=0.05)
        return record

    arguments = inputs(predict)
    arguments["past_inner_error_rad"][:, 6] = np.linspace(0.0, 0.06, 25)
    before = arguments["past_inner_error_rad"].copy()
    result = planning.solve_single_period_response(**arguments)
    assert not result["initial"]["audit"]["feasible"]
    assert result["witness_found"]
    selected = result["selected"]
    assert selected["command"][0] <= -0.1 + 1e-9
    margin = 0.05 + before
    q = selected["response"]["joint_position"]
    assert np.all(q <= arguments["upper"] - margin + 1e-10)
    assert np.all(q >= arguments["lower"] + margin - 1e-10)
    np.testing.assert_array_equal(arguments["past_inner_error_rad"], before)
    np.testing.assert_array_equal(result["past_inner_error_rad"], before)


def test_boundary_differences_and_all_search_calls_preserve_box_and_rate_limits():
    calls = []

    def predict(command):
        calls.append(command.copy())
        record = response(command)
        record["joint_position"][:] = 0
        return record

    arguments = inputs(predict)
    arguments["previous"][:2] = [0.9, -0.9]
    arguments["warm"][:2] = [1.0, -1.0]
    arguments["desired"][0, 3] = 1.0
    result = planning.solve_single_period_response(**arguments, max_evaluations=7)
    assert result["response_evaluations"] == len(calls) == 7
    commands = np.asarray(calls)
    assert np.all(commands >= -1.0) and np.all(commands <= 1.0)
    assert np.all(abs(commands - arguments["previous"]) <= 0.25)
    perturbations = commands[1:] - arguments["warm"]
    np.testing.assert_allclose(abs(perturbations), np.eye(6) * 1e-4, atol=1e-15)
    assert perturbations[0, 0] < 0 and perturbations[1, 1] > 0
    assert not result["witness_found"]
    assert result["selected"]["audit"]["feasible"]
    assert result["phases"][-1]["status"] == "response_budget"


def test_unreachable_target_returns_best_safe_command_without_enlarging_rate_limit():
    arguments = inputs()
    arguments["desired"][0, 3] = 0.06
    result = planning.solve_single_period_response(**arguments)
    assert not result["witness_found"]
    selected = result["selected"]
    assert selected["audit"]["feasible"]
    assert selected["command"][0] == pytest.approx(0.25, abs=1e-8)
    assert np.linalg.norm(selected["tcp_error"][:3]) == pytest.approx(0.035, abs=1e-8)
    assert selected["maximum_pose_cap_violation"] == pytest.approx(2.5, abs=1e-6)


@pytest.mark.parametrize("failure", ["pose", "safety"])
def test_optimizer_success_does_not_certify_pose_or_joint_safety(monkeypatch, failure):
    def misleading_success(_objective, start, **_kwargs):
        return SimpleNamespace(success=True, status=0, message="claimed success", nit=1, x=start)

    def predict(command):
        record = response(command)
        record["endpoint_pose"] = np.eye(4)
        if failure == "safety":
            record["joint_position"][7, 6] = 0.99
        return record

    arguments = inputs(predict)
    if failure == "pose":
        arguments["desired"][:3, :3] = Rotation.from_rotvec([0.0, 0.0, 0.08]).as_matrix()
    monkeypatch.setattr(planning, "minimize", misleading_success)
    result = planning.solve_single_period_response(**arguments)
    assert all(phase["success"] for phase in result["phases"])
    assert not result["witness_found"]
    assert result["status"] == "finite_search_without_witness"
    if failure == "pose":
        assert result["selected"]["audit"]["feasible"]
        assert result["selected"]["maximum_pose_cap_violation"] == pytest.approx(0.6)
    else:
        assert result["selected"] is None
        assert not result["closest_safety_attempt"]["audit"]["feasible"]


def test_recovery_and_pose_phases_share_one_response_budget(monkeypatch):
    calls = []
    phases_entered = []

    def predict(command):
        calls.append(command.copy())
        record = response(command)
        record["joint_position"][:, 6] = 0.98 + command[0]
        return record

    def requesting_optimizer(objective, start, *, constraints, **_kwargs):
        phases_entered.append(start.copy())
        constraints["jac"](start)
        final = start.copy()
        if len(phases_entered) == 1:
            final[0], final[6] = -0.1, 0.0
        else:
            final[1] = 0.01
        objective(final)
        constraints["fun"](final)
        return SimpleNamespace(success=True, status=0, message="requested evaluations", nit=1, x=final)

    monkeypatch.setattr(planning, "minimize", requesting_optimizer)
    arguments = inputs(predict)
    arguments["desired"][0, 3] = 1.0
    result = planning.solve_single_period_response(**arguments, max_evaluations=14)
    assert len(phases_entered) == 2
    assert result["response_evaluations"] == len(calls) == 14
    assert result["phases"][0]["evaluations_after"] == 8
    assert result["phases"][1]["evaluations_before"] == 8
    assert result["phases"][1]["status"] == "response_budget"
    assert not result["witness_found"] and result["selected"]["audit"]["feasible"]


@pytest.mark.parametrize("invalid", ["negative_profile", "decreasing_profile", "short_rollout"])
def test_incomplete_or_noncausal_safety_inputs_are_rejected(invalid):
    arguments = inputs()
    if invalid == "negative_profile":
        arguments["past_inner_error_rad"][:] = -0.01
    elif invalid == "decreasing_profile":
        arguments["past_inner_error_rad"][:, 0] = np.linspace(0.02, 0.0, 25)
    else:

        def incomplete(command):
            record = response(command)
            record["joint_position"] = record["joint_position"][:24]
            return record

        arguments["predict"] = incomplete
    with pytest.raises(ValueError):
        planning.solve_single_period_response(**arguments)
