import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import matrix_to_rot6d
from s2a.trajectory_audit import trajectory_diagnostics


def test_masked_terminal_is_excluded_and_local_rotations_affect_integrated_translation():
    ref = np.zeros((3, 128))
    ref[:, 3:9] = [1, 0, 0, 0, 1, 0]
    ref[:, 9] = 1
    pred = ref.copy()
    pred[0, 3:9] = matrix_to_rot6d(Rotation.from_euler("z", 90, degrees=True).as_matrix())
    pred[1, 0] = ref[1, 0] = 0.01
    pred[2, :10] = 0  # An invalid terminal cannot become an identity-pose label.
    pred[2, 9] = 100
    mask = np.zeros_like(ref, dtype=bool)
    mask[:2, :10] = True
    result = trajectory_diagnostics(pred, ref, mask)
    assert result["pose_steps"] == result["valid_prefix_steps"] == 2
    assert result["translation_rmse_m"] == 0
    np.testing.assert_allclose(result["integrated_final_position_error_m"], np.sqrt(2) * 0.01)
    np.testing.assert_allclose(result["integrated_final_rotation_error_rad"], np.pi / 2)
    assert result["opening_mae_raw"] == result["opening_outside_fraction"] == 0
