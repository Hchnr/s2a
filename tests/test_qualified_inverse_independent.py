import numpy as np
import pytest
import qualified_inverse_independent as audit

from s2a.bounded_response_inverse import translation_command


def test_kkt_accepts_constrained_optimum_and_rejects_wrong_feasible_command(monkeypatch):
    j = np.zeros((12, 3))
    j[:3] = np.diag([1.0, 2.0, 3.0])
    j[6:9] = np.eye(3) * 0.5
    offset = np.arange(12) * 0.0001
    monkeypatch.setattr(audit, "prediction", lambda row, p: (offset + j @ row["action_goal_delta"][0, :3])[None, :])
    data = dict(action_command=np.zeros((1, 7)), action_goal_delta=np.zeros((1, 6)))
    contract = dict(
        input_min=-np.ones(6), input_max=np.ones(6), output_min=-np.ones(6) * 0.05, output_max=np.ones(6) * 0.05
    )
    target = np.array([0.05, -0.05, 0.0])
    velocity = np.zeros(3)
    previous = np.zeros(3)
    bounds = {k: np.asarray(v)[:3] for k, v in contract.items()}
    solution = translation_command(
        offset[[0, 1, 2, 6, 7, 8]],
        j[[0, 1, 2, 6, 7, 8]],
        target,
        velocity,
        dt=0.05,
        previous_command=previous,
        max_rate=5,
        **bounds,
    )
    checked = audit.verify_inverse(data, {}, contract, previous, target, velocity, solution["command"])
    np.testing.assert_allclose(checked["objective"], solution["squared_objective"], atol=1e-14)
    assert np.isclose(solution["command"][0], 0.25) and np.isclose(solution["command"][1], -0.25)
    wrong = solution["command"].copy()
    wrong[0] = 0
    with pytest.raises(AssertionError, match="KKT"):
        audit.verify_inverse(data, {}, contract, previous, target, velocity, wrong)
    with pytest.raises(AssertionError):
        audit.verify_inverse(data, {}, contract, previous, target, velocity, np.ones(3))
