import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.joint_wrench_condition_forward import with_history
from s2a.runtime_response_history import RuntimeResponseHistory


def test_completed_runtime_matches_training_prefix_and_rejects_wrong_endpoint():
    rng = np.random.default_rng(42)
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, 3] = rng.normal(size=(4, 3)) * 0.01
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(size=(4, 3)) * 0.03).as_matrix()
    data = dict(
        x_site_pose=poses,
        x_site_twist=rng.normal(size=(4, 6)) * 0.1,
        action_goal_delta=rng.normal(size=(4, 6)) * 0.02,
        x_previous_command_available=np.array([False, True, True, True]),
    )
    p = dict(dt=0.05, kp=[150.0] * 6, kd=[24.0] * 6)
    expected = with_history(data, p)["x_response_residual"]
    history = RuntimeResponseHistory()
    for i in range(4):
        row = {k: v[i : i + 1] for k, v in data.items()}
        first = history.attach(row)
        np.testing.assert_allclose(first["x_response_residual"][0], expected[i], rtol=0, atol=1e-14)
        first["x_response_residual"][0, 0] = 99
        np.testing.assert_allclose(history.attach(row)["x_response_residual"][0], expected[i], rtol=0, atol=1e-14)
        assert history.completed_periods == i
        if i < 3:
            history.observe_completed(row, poses[i + 1], data["x_site_twist"][i + 1], p)
    wrong = {k: v[0:1] for k, v in data.items()}
    with pytest.raises(ValueError, match="differs"):
        history.attach(wrong)
    history.reset()
    np.testing.assert_array_equal(history.attach(wrong)["x_response_residual"], np.zeros((1, 12)))
    assert history.completed_periods == 0
