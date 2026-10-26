import copy

import numpy as np
import pytest
from test_joint_wrench_condition_forward import prepared

from s2a.causal_wrench_change_forward import attach_wrench_change, predict
from s2a.joint_wrench_condition_forward import predict as previous_predict


def test_difference_requires_whole_episode_and_resets_without_cross_episode_state():
    data, _, _ = prepared()
    data["x_previous_command_available"] = np.array([False, True])
    output = attach_wrench_change(data)
    np.testing.assert_array_equal(output["x_wrench_change_world"][0], np.zeros(6))
    np.testing.assert_array_equal(
        output["x_wrench_change_world"][1], data["x_current_wrench_world"][1] - data["x_current_wrench_world"][0]
    )
    another = copy.deepcopy(data)
    another["x_current_wrench_world"] += 100
    np.testing.assert_array_equal(
        attach_wrench_change(another)["x_wrench_change_world"], output["x_wrench_change_world"]
    )
    invalid = {k: v[1:] for k, v in data.items() if isinstance(v, np.ndarray)}
    with pytest.raises(ValueError, match="Complete single-trajectory"):
        attach_wrench_change(invalid)


def test_embedding_affinity_future_isolation_and_explicit_history_requirement():
    data, old, _ = prepared()
    data["x_previous_command_available"] = np.array([False, True])
    data = attach_wrench_change(data)
    columns = np.concatenate([np.arange(i * 288, i * 288 + 282) for i in range(4)])
    p = dict(old, coefficients=np.zeros((1152, 12)))
    p["coefficients"][columns] = old["coefficients"]
    np.testing.assert_allclose(predict(data, p), previous_predict(data, old), rtol=0, atol=1e-13)
    p["coefficients"][282:288, 0] = 0.01
    a = predict(data, p)
    future = copy.deepcopy(data)
    future["x_current_wrench_world"][1] += 1000
    future["y_twist"] = np.full((2, 6), np.nan)
    np.testing.assert_array_equal(a[0], predict(attach_wrench_change(future), p)[0])
    b = copy.deepcopy(data)
    b["action_goal_delta"][:, :3] += 0.02
    midpoint = copy.deepcopy(data)
    midpoint["action_goal_delta"] = (data["action_goal_delta"] + b["action_goal_delta"]) / 2
    p["coefficients"][288 + 282 : 288 + 288, 0] = 0.01
    np.testing.assert_allclose(predict(midpoint, p), (predict(data, p) + predict(b, p)) / 2, rtol=0, atol=1e-12)
    del data["x_wrench_change_world"]
    with pytest.raises(ValueError, match="Explicit causal"):
        predict(data, p)


def test_independent_feature_assembly_agrees_with_explicit_nonzero_history():
    from arm_wrench_change_independent import feature_matrix

    from s2a.causal_wrench_change_forward import design

    data, parameters, _ = prepared()
    data["x_previous_command_available"] = np.array([False, True])
    data = attach_wrench_change(data)
    np.testing.assert_allclose(design(data, parameters), feature_matrix(data, parameters), rtol=0, atol=1e-12)
    assert np.linalg.norm(data["x_wrench_change_world"][1]) > 0
