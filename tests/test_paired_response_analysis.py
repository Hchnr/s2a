import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.paired_response_analysis import axis_response


def packet(u, axis=1):
    pose = np.eye(4)
    pose[:3, 3] = np.array([0.02, 0.01, -0.03]) * u
    pose[:3, :3] = (
        Rotation.from_rotvec(np.array([0.001, -0.002, 0.003]) * u).as_matrix()
        @ Rotation.from_rotvec([0, 0.8, 0]).as_matrix()
    )
    twist = np.r_[np.array([0.4, 0.2, -0.6]) * u, np.array([0.01, 0.02, 0.03]) * u]
    command = np.zeros(7)
    command[axis] = u
    return dict(pose=pose, twist=twist, command=command)


def test_coupled_response_world_so3_and_physical_interface_units():
    result = axis_response(packet(0), packet(-0.1), packet(0.1), 1, 0.05)
    for side in ("minus", "plus", "central"):
        measured = result[side]["per_normalized_input"]
        np.testing.assert_allclose(measured["position"], [0.02, 0.01, -0.03], atol=1e-14)
        np.testing.assert_allclose(measured["so3_increment"], [0.001, -0.002, 0.003], atol=1e-14)
        np.testing.assert_allclose(measured["linear_velocity"], [0.4, 0.2, -0.6], atol=1e-14)
        np.testing.assert_allclose(result[side]["per_physical_goal_delta"]["position"], [0.4, 0.2, -0.6], atol=1e-14)


def test_clipped_side_is_unavailable_and_actual_separation_is_used():
    result = axis_response(packet(1), packet(0.9), packet(1), 1, 0.05)
    assert not result["plus"]["available"] and result["plus"]["per_normalized_input"] is None
    assert result["central"]["actual_normalized_delta"] == pytest.approx(0.1)
    np.testing.assert_allclose(result["central"]["per_normalized_input"]["position"], [0.02, 0.01, -0.03], atol=1e-14)
    assert result["one_sided_disagreement"] is None


def test_other_command_interference_is_rejected():
    plus = packet(0.1)
    plus["command"][0] = 0.01
    with pytest.raises(ValueError):
        axis_response(packet(0), packet(-0.1), plus, 1, 0.05)


def test_nonfinite_measurement_and_missing_scale_cannot_form_response():
    plus = packet(0.1)
    plus["twist"][0] = np.nan
    with pytest.raises(ValueError):
        axis_response(packet(0), packet(-0.1), plus, 1, 0.05)
    with pytest.raises(ValueError):
        axis_response(packet(0), packet(-0.1), packet(0.1), 1, 0)
