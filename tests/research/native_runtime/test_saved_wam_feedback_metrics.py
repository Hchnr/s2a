import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error
from scripts.verification.reference_ports import audit_saved_wam_feedback as audit


def test_logged_pose_error_recovers_noncommuting_actual_rotations():
    actual = np.repeat(np.eye(4)[None], 3, axis=0)
    actual[:, :3, :3] = Rotation.from_euler("xyz", [[0.2, -0.3, 0.5], [-0.1, 0.7, -0.2], [0.5, 0.2, -0.4]]).as_matrix()
    actual[:, :3, 3] = [[0.1, 0.2, 0.3], [0.2, -0.1, 0.4], [-0.2, 0.1, 0.5]]
    target = actual.copy()
    target[:, :3, :3] = Rotation.from_euler("zyx", [[0.3, 0.1, -0.2], [0.2, -0.4, 0.5], [-0.3, 0.1, 0.4]]).as_matrix()
    target[:, :3, 3] += [0.01, -0.02, 0.03]
    recovered = audit.actual_from_logged_error(target, pose_error(target, actual))
    np.testing.assert_allclose(recovered, actual, atol=1e-12, rtol=0)


def test_logged_pose_error_rejects_misaligned_or_nonfinite_inputs():
    targets = np.repeat(np.eye(4)[None], 2, axis=0)
    with pytest.raises(ValueError, match="paired TCP"):
        audit.actual_from_logged_error(targets, np.zeros((1, 6)))
    errors = np.zeros((2, 6))
    errors[0, 0] = np.nan
    with pytest.raises(ValueError, match="paired TCP"):
        audit.actual_from_logged_error(targets, errors)


def test_saved_replay_decomposes_wam_and_correct_reference_errors():
    result = audit.load_audit()
    assert result["steps"] == 8
    assert result["replayed_prefix_inner_samples"] == 2000
    assert result["replayed_continuation_inner_samples"] == 200
    assert result["actual_wam"]["pose_caps_against_reference"] == 8
    assert result["actual_wam"]["pose_caps_against_correct"] == 2
    assert result["actual_wam"]["actual_vs_reference"]["position_rmse_m"] == pytest.approx(0.00240518, abs=1e-8)
    assert result["actual_wam"]["actual_vs_correct"]["position_rmse_m"] == pytest.approx(0.01311812, abs=1e-8)
    assert result["original_wam_worker_success"] is False
    assert result["initial_noise_ledger_available"] is False
    assert result["new_physics_steps"] == result["new_model_requests"] == 0
