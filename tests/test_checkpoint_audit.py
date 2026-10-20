import numpy as np
from scipy.spatial.transform import Rotation

from s2a.checkpoint_audit import extrapolate_history, state_metrics
from s2a.geometry import transform


def test_extrapolation_uses_past_world_motion_with_nonidentity_anchor():
    times = np.arange(9)
    r0 = Rotation.from_euler("xyz", [0.4, -0.2, 0.7]).as_matrix()
    poses = transform(
        [0.1, -0.3, 0.8] + times[:, None] * np.array([0.002, -0.001, 0.003]),
        Rotation.from_rotvec(times[:, None] * np.array([0.003, 0.001, -0.004])).as_matrix() @ r0,
    )
    prediction = extrapolate_history(poses[:5], 4)
    np.testing.assert_allclose(prediction, poses[5:], atol=1e-12)
    np.testing.assert_array_equal(extrapolate_history(poses[:1], 4), np.repeat(poses[:1], 4, axis=0))


def test_state_error_uses_so3_geodesic_across_pi():
    reference = transform(np.zeros((1, 3)), Rotation.from_rotvec([[0, 0, np.deg2rad(179)]]).as_matrix())
    prediction = transform(np.array([[0.003, 0.004, 0]]), Rotation.from_rotvec([[0, 0, np.deg2rad(-179)]]).as_matrix())
    result = state_metrics(reference, prediction, [0.2], [0.8])
    np.testing.assert_allclose(result["position_rmse_m"], 0.005)
    np.testing.assert_allclose(result["rotation_rmse_rad"], np.deg2rad(2))
    np.testing.assert_allclose(result["opening_mae"], 0.6)
