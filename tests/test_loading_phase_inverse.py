import copy

import loading_support_independent as independent
import numpy as np
import pytest
import qualified_inverse_independent as kkt

import s2a.support_aware_inverse as producer
from s2a.loading_phase_inverse import LoadingPhaseSupport, loading_phase_translation
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
    solved = loading_phase_translation(
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
