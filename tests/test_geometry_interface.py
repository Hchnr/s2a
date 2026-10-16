import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from s2a.geometry import (
    advance,
    euler_pose_to_transform,
    integrate_local,
    inverse,
    matrix_to_rot6d,
    pose9_to_transform,
    pose_error,
    rot6d_to_matrix,
    transform,
)
from s2a.interface import DroidReferenceInterface


def test_column_rot6d_composition_and_integration():
    rng = np.random.default_rng(42)
    rotations = Rotation.random(7, random_state=rng).as_matrix()
    positions = rng.normal(size=(7, 3))
    t = transform(positions, rotations)
    deltas = inverse(t[:-1]) @ t[1:]
    encoded = np.c_[deltas[:, :3, 3], matrix_to_rot6d(deltas[:, :3, :3])]
    assert_allclose(pose9_to_transform(encoded), deltas, atol=2e-15)
    assert_allclose(integrate_local(t[0], encoded), t, atol=3e-15)
    assert_allclose(t @ inverse(t), np.broadcast_to(np.eye(4), t.shape), atol=2e-15)
    # Cross-check concatenated columns rather than flattened row-major 3x2.
    assert_allclose(rot6d_to_matrix([0, 1, 0, -1, 0, 0]), Rotation.from_euler("z", np.pi / 2).as_matrix(), atol=1e-15)


@pytest.mark.parametrize("value", [[0] * 6, [1, 0, 0, 2, 0, 0], [np.nan, 0, 0, 0, 1, 0]])
def test_rot6d_rejects_degenerate_predictions(value):
    with pytest.raises(ValueError):
        rot6d_to_matrix(value)


def test_tcp_lever_arm_and_base_rotation_error():
    h = euler_pose_to_transform([0, 0, 0.1438, 0, 0, np.pi / 2])
    flange = euler_pose_to_transform([0.3, 0.1, 0.5, 0.2, -0.3, 0.7])
    tcp = flange @ h
    tcp_next = advance(tcp, [0, 0, 0, 0.2, -0.1, 0], 0.1)
    flange_next = tcp_next @ inverse(h)
    assert np.linalg.norm(flange_next[:3, 3] - flange[:3, 3]) > 0.003
    assert_allclose(flange_next @ h, tcp_next, atol=1e-15)
    assert_allclose(pose_error(tcp_next, tcp)[3:], [0.02, -0.01, 0], atol=1e-15)


def test_droid_reference_algebra_matches_composed_euler_and_vector_limits():
    iface = DroidReferenceInterface()
    measured = euler_pose_to_transform([0.4, 0, 0.5, 0.7, 0.4, -0.3])
    delta = Rotation.from_euler("xyz", [0.04, -0.03, 0.02])
    target = transform(measured[:3, 3] + [0.015, -0.0075, 0], delta.as_matrix() @ measured[:3, :3])
    expected = [0.2, -0.1, 0, 0.04 / 0.15, -0.03 / 0.15, 0.02 / 0.15]
    command = iface.encode_target(target, measured)
    assert_allclose(command, expected, atol=1e-15)
    assert_allclose(iface.decode_target(command, measured), target, atol=1e-15)
    assert_allclose(iface.limit([3, 4, 0, 0, 0, 2]), [0.6, 0.8, 0, 0, 0, 1])
    assert_allclose(iface.gripper_position([0, 0.25, 1]), [1, 0.75, 0])
    with pytest.raises(ValueError):
        DroidReferenceInterface(gripper_action_space="velocity")


def test_physical_velocity_bridge_does_not_use_future_dt():
    iface = DroidReferenceInterface()
    t = np.eye(4)
    command = iface.encode_velocity([0.01, 0, 0, 0, 0, 0.1], t, 0.05)
    assert_allclose(command, [0.0005 / 0.075, 0, 0, 0, 0, 0.005 / 0.15], atol=1e-15)
    for bad in [0, -1, np.nan]:
        with pytest.raises(ValueError):
            iface.encode_velocity(np.zeros(6), t, bad)


def test_final_command_rate_bound_after_large_inverse_and_vector_projection():
    interface = DroidReferenceInterface()
    previous = np.array([0.5, 0, 0, 0, 0, 0.5])
    command = interface.limit_rate([-100, 100, 0, 100, 0, -100], previous, 0.07, 2.0, 4.0)
    assert np.linalg.norm(command[:3] - previous[:3]) <= 0.14 + 1e-14
    assert np.linalg.norm(command[3:] - previous[3:]) <= 0.28 + 1e-14
    assert np.linalg.norm(command[:3]) <= 1
    assert np.linalg.norm(command[3:]) <= 1
