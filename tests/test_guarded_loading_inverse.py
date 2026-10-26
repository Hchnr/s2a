import copy

import guarded_loading_support_independent as independent
import numpy as np
import pytest
import qualified_inverse_independent as kkt

import s2a.support_aware_inverse as producer
from s2a.loading_phase_inverse import LoadingPhaseSupport
from s2a.tracking_guarded_loading_inverse import tracking_guarded_loading_translation
from s2a.training_support import TrainingSupport


@pytest.mark.parametrize("route,weight", [("native_close_load", 0), ("model_tracking", 0.5)])
def test_route_switch_and_independent_command_audit(monkeypatch, route, weight):
    condition = np.zeros((1, 89))
    monkeypatch.setattr(producer, "raw_condition", lambda _: condition)
    monkeypatch.setattr(independent, "conditions", lambda _: condition)
    matrix = np.zeros((12, 3))
    matrix[:3] = np.eye(3)
    matrix[6:9] = np.eye(3)

    def prediction(row, _):
        return (matrix @ row["action_goal_delta"][0, :3])[None, :]

    monkeypatch.setattr(independent, "prediction", prediction)
    monkeypatch.setattr(kkt, "prediction", prediction)
    data = dict(action_command=np.zeros((1, 7)), action_goal_delta=np.zeros((1, 6)))
    params = dict(state_mean=np.zeros(89), state_scale=np.ones(89))
    contract = dict(input_min=-np.ones(6), input_max=np.ones(6), output_min=-np.ones(6), output_max=np.ones(6))
    physical = np.array([-0.2, 0, 0.1])
    support = TrainingSupport(np.zeros((1, 89)), 1)
    solved = tracking_guarded_loading_translation(
        data,
        params,
        np.ones(3),
        np.zeros(3),
        support=support,
        route=route,
        physical_command=physical,
        previous_command=np.zeros(3),
        predictor=prediction,
        **{k: v[:3] for k, v in contract.items()},
    )
    assert solved["support_decision"]["learned_weight"] == weight
    if route == "native_close_load":
        np.testing.assert_allclose(solved["command"], physical, atol=1e-14)
    args = (data, params, contract, np.zeros(3), np.ones(3), np.zeros(3), physical, np.zeros((1, 89)), 1)
    independent.verify_support_inverse(*args, solved, expected_route=route)
    wrong = copy.deepcopy(solved)
    wrong["arm_loading_route"] = "model_tracking" if route == "native_close_load" else "native_close_load"
    with pytest.raises(AssertionError):
        independent.verify_support_inverse(*args, wrong, expected_route=route)


def test_unknown_route_rejected():
    with pytest.raises(ValueError, match="route"):
        LoadingPhaseSupport(TrainingSupport(np.zeros((1, 89)), 1), "future_success")


def test_loading_projection_uses_actual_previous_command_at_interface_edge(monkeypatch):
    condition = np.zeros((1, 89))
    monkeypatch.setattr(producer, "raw_condition", lambda _: condition)
    data = dict(action_command=np.zeros((1, 7)), action_goal_delta=np.zeros((1, 6)))
    matrix = np.zeros((12, 3))
    matrix[:3] = np.eye(3)

    def prediction(row, _):
        return (matrix @ row["action_goal_delta"][0, :3])[None, :]

    support = TrainingSupport(np.zeros((1, 89)), 1)
    previous = np.array([0.9, -0.9, 0.0])
    solved = tracking_guarded_loading_translation(
        data,
        dict(state_mean=np.zeros(89), state_scale=np.ones(89)),
        np.zeros(3),
        np.zeros(3),
        support=support,
        route="native_close_load",
        physical_command=np.array([-1.0, 1.0, 0]),
        input_min=-np.ones(3),
        input_max=np.ones(3),
        output_min=-np.ones(3),
        output_max=np.ones(3),
        previous_command=previous,
        predictor=prediction,
    )
    np.testing.assert_allclose(solved["pre_rate_command"], [0.65, -0.65, 0], atol=1e-14)
    np.testing.assert_allclose(solved["command"], [0.65, -0.65, 0], atol=1e-14)
    np.testing.assert_allclose(solved["full_predicted_response"], matrix @ solved["command"], atol=1e-14)
    assert solved["normalized_translation_rate"] == 5
    assert not solved["tracking_guard"]["tracking_guard_feasible"]
    np.testing.assert_array_equal(previous, [0.9, -0.9, 0])


@pytest.mark.parametrize(
    "condition_value,reason,expected", [(0, "guard_evaluated", 0.19), (3, "outside_training_support", 0.25)]
)
def test_independent_forward_map_selects_root_or_bypasses_extrapolation(monkeypatch, condition_value, reason, expected):
    condition = np.zeros((1, 89))
    condition[0, 0] = condition_value
    monkeypatch.setattr(producer, "raw_condition", lambda _: condition)
    monkeypatch.setattr(independent, "conditions", lambda _: condition)
    matrix = np.zeros((12, 3))
    matrix[0, 0] = 0.05

    def prediction(row, _):
        return (matrix @ row["action_goal_delta"][0, :3])[None, :]

    monkeypatch.setattr(independent, "prediction", prediction)
    monkeypatch.setattr(kkt, "prediction", prediction)
    data = dict(action_command=np.zeros((1, 7)), action_goal_delta=np.zeros((1, 6)))
    params = dict(state_mean=np.zeros(89), state_scale=np.ones(89))
    contract = dict(input_min=-np.ones(6), input_max=np.ones(6), output_min=-np.ones(6), output_max=np.ones(6))
    points = np.zeros((1, 89))
    physical = np.array([0.25, 0, 0])
    target = np.array([0.0145, 0, 0])
    solved = tracking_guarded_loading_translation(
        data,
        params,
        target,
        np.zeros(3),
        support=TrainingSupport(points, 1),
        route="native_close_load",
        physical_command=physical,
        previous_command=np.zeros(3),
        predictor=prediction,
        **{k: v[:3] for k, v in contract.items()},
    )
    assert solved["tracking_guard"]["reason"] == reason
    np.testing.assert_allclose(solved["command"], [expected, 0, 0], atol=1e-12)
    args = (data, params, contract, np.zeros(3), target, np.zeros(3), physical, points, 1)
    independent.verify_support_inverse(*args, solved, expected_route="native_close_load")
    wrong = copy.deepcopy(solved)
    wrong["tracking_guard"]["change"] += 0.01
    with pytest.raises(AssertionError):
        independent.verify_support_inverse(*args, wrong, expected_route="native_close_load")
