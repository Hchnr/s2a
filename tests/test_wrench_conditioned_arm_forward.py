import copy

import numpy as np
import pytest
from test_nonlinear_conditioned_arm_forward import setup as original_setup

from s2a.nonlinear_conditioned_arm_forward import design as original_design
from s2a.nonlinear_conditioned_arm_forward import predict as original_predict
from s2a.wrench_conditioned_arm_forward import design, predict


def setup():
    data, previous = original_setup()
    data["x_current_wrench_world"] = np.array([[0, 0, 5, 0, 0, 0], [2, 3, 30, 1, 0, 0]], dtype=float)
    p = dict(
        previous,
        state_mean=np.r_[previous["state_mean"], np.zeros(6)],
        state_scale=np.r_[previous["state_scale"], np.ones(6)],
        coefficients=np.zeros((872, 12)),
    )
    columns = np.concatenate([np.arange(j * 218, j * 218 + 212) for j in range(4)])
    p["coefficients"][columns] = previous["coefficients"]
    return data, p, previous, columns


def test_original_prediction_is_preserved_when_new_coefficients_are_zero():
    data, p, previous, columns = setup()
    np.testing.assert_array_equal(design(data, p)[:, columns], original_design(data, previous))
    np.testing.assert_allclose(predict(data, p), original_predict(data, previous), rtol=0, atol=1e-13)


def test_wrench_changes_gain_without_breaking_action_affinity():
    a, p, _, _ = setup()
    p["coefficients"][218 + 212, 0] = 0.01
    a["x_current_wrench_world"][:, 0] = 3
    b = copy.deepcopy(a)
    a["action_goal_delta"][:, :3] = [-0.02, 0.01, 0.03]
    b["action_goal_delta"][:, :3] = [0.01, -0.01, -0.02]
    middle = copy.deepcopy(a)
    middle["action_goal_delta"][:, :3] = (a["action_goal_delta"][:, :3] + b["action_goal_delta"][:, :3]) / 2
    np.testing.assert_allclose(predict(middle, p), (predict(a, p) + predict(b, p)) / 2, rtol=0, atol=1e-12)
    altered = copy.deepcopy(a)
    altered["x_current_wrench_world"][:, 0] += 2
    assert not np.allclose(predict(a, p), predict(altered, p))


def test_future_wrench_and_labels_do_not_affect_current_prediction():
    data, p, _, _ = setup()
    p["coefficients"][212, 0] = 0.01
    before = predict(data, p)
    later = copy.deepcopy(data)
    later["x_current_wrench_world"][1] = 1e4
    later["y_twist"] = np.full((2, 6), np.nan)
    np.testing.assert_array_equal(before[0], predict(later, p)[0])


def test_missing_measurement_is_not_silently_replaced():
    data, p, _, _ = setup()
    del data["x_current_wrench_world"]
    with pytest.raises(ValueError, match="Current measured"):
        predict(data, p)
