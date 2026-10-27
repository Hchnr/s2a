import copy

import numpy as np
from test_wrench_conditioned_arm_forward import setup

from s2a.joint_wrench_condition_forward import design, predict
from s2a.wrench_conditioned_arm_forward import predict as previous_predict


def prepared():
    data, old, _, _ = setup()
    p = dict(old, joint_frequencies=np.random.Generator(np.random.PCG64(42)).normal(size=(32, 89)) / np.sqrt(89))
    columns = np.concatenate([np.arange(i * 282, i * 282 + 218) for i in range(4)])
    p["coefficients"] = np.zeros((1128, 12))
    p["coefficients"][columns] = old["coefficients"]
    return data, p, old


def test_old_prediction_embeds_exactly_and_new_features_match_scalar_phases():
    data, p, old = prepared()
    np.testing.assert_allclose(predict(data, p), previous_predict(data, old), rtol=0, atol=1e-13)
    from arm_joint_wrench_independent import feature_matrix

    np.testing.assert_allclose(design(data, p), feature_matrix(data, p), rtol=0, atol=1e-12)


def test_joint_gain_remains_action_affine_and_uses_only_current_inputs():
    a, p, _ = prepared()
    p["coefficients"][282 + 218 : 282 + 250, 0] = 0.01
    b = copy.deepcopy(a)
    b["action_goal_delta"][:, :3] += 0.01
    middle = copy.deepcopy(a)
    middle["action_goal_delta"] = (a["action_goal_delta"] + b["action_goal_delta"]) / 2
    np.testing.assert_allclose(predict(middle, p), (predict(a, p) + predict(b, p)) / 2, rtol=0, atol=1e-12)
    before = predict(a, p)
    later = copy.deepcopy(a)
    later["x_current_wrench_world"][1] += 100
    later["y_twist"] = np.full((2, 6), np.nan)
    np.testing.assert_array_equal(before[0], predict(later, p)[0])
    changed = copy.deepcopy(a)
    changed["x_current_wrench_world"][0] += 1
    assert not np.allclose(before[0], predict(changed, p)[0])
