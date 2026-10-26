import numpy as np

import s2a.support_aware_inverse as module
from s2a.residual_supported_inverse import residual_supported_translation
from s2a.training_support import TrainingSupport


def test_executed_mixture_reprediction_is_distinct_from_learned_optimum(monkeypatch):
    monkeypatch.setattr(module, "raw_condition", lambda _: np.full((1, 89), 3.0))
    data = dict(action_command=np.zeros((1, 7)), action_goal_delta=np.zeros((1, 6)))
    parameters = dict(state_mean=np.zeros(89), state_scale=np.ones(89))
    matrix = np.zeros((12, 3))
    matrix[:3] = np.eye(3)
    matrix[6:9] = np.eye(3)

    def predictor(row, _):
        return (matrix @ row["action_goal_delta"][0, :3])[None, :]

    result = residual_supported_translation(
        data,
        parameters,
        np.ones(3),
        np.ones(3) * 20,
        support=TrainingSupport(np.zeros((1, 89)), 1),
        physical_command=np.array([-1, 1, 0]),
        input_min=-np.ones(3),
        input_max=np.ones(3),
        output_min=-np.ones(3),
        output_max=np.ones(3),
        previous_command=np.zeros(3),
        predictor=predictor,
    )
    assert result["support_decision"]["learned_weight"] == 0
    np.testing.assert_allclose(result["command"], [-0.25, 0.25, 0])
    np.testing.assert_allclose(result["learned_solution"]["command"], [0.25, 0.25, 0.25])
    expected = matrix @ np.array([-0.25, 0.25, 0])
    np.testing.assert_allclose(result["full_predicted_response"], expected)
    assert result["squared_objective"] > result["learned_solution"]["squared_objective"]
    np.testing.assert_array_equal(data["action_command"], np.zeros((1, 7)))


def test_independent_checker_rejects_wrong_support_weight_and_wrong_mixture(monkeypatch):
    import copy

    import pytest
    import qualified_inverse_independent as kkt
    import residual_support_independent as independent

    condition = np.zeros((1, 89))
    condition[0, 0] = 1.5
    monkeypatch.setattr(module, "raw_condition", lambda _: condition)
    monkeypatch.setattr(independent, "conditions", lambda _: condition)
    matrix = np.zeros((12, 3))
    matrix[:3] = np.eye(3)
    matrix[6:9] = 0.5 * np.eye(3)

    def predictor(row, _):
        return (matrix @ row["action_goal_delta"][0, :3])[None, :]

    monkeypatch.setattr(independent, "prediction", predictor)
    monkeypatch.setattr(kkt, "prediction", predictor)
    data = dict(action_command=np.zeros((1, 7)), action_goal_delta=np.zeros((1, 6)))
    parameters = dict(state_mean=np.zeros(89), state_scale=np.ones(89))
    contract = dict(input_min=-np.ones(6), input_max=np.ones(6), output_min=-np.ones(6), output_max=np.ones(6))
    points = np.zeros((1, 89))
    physical = np.array([-0.2, 0.1, 0])
    target, velocity = np.array([0.1, 0, 0]), np.zeros(3)
    solved = residual_supported_translation(
        data,
        parameters,
        target,
        velocity,
        support=TrainingSupport(points, 1),
        physical_command=physical,
        previous_command=np.zeros(3),
        predictor=predictor,
        **{k: v[:3] for k, v in contract.items()},
    )
    arguments = (data, parameters, contract, np.zeros(3), target, velocity, physical, points, 1)
    independent.verify_support_inverse(*arguments, solved)
    wrong_weight = copy.deepcopy(solved)
    wrong_weight["support_decision"]["learned_weight"] = 1
    with pytest.raises(AssertionError):
        independent.verify_support_inverse(*arguments, wrong_weight)
    wrong_command = copy.deepcopy(solved)
    wrong_command["command"][0] += 0.02
    with pytest.raises(AssertionError):
        independent.verify_support_inverse(*arguments, wrong_command)
    wrong_physical = copy.deepcopy(solved)
    wrong_physical["physical_candidate"][0] += 0.02
    with pytest.raises(AssertionError):
        independent.verify_support_inverse(*arguments, wrong_physical)
