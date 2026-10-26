import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.temporal_rotation_residual import decode, encode


def test_noncommuting_rotation_roundtrip_and_world_frame_equivariance():
    rng = np.random.default_rng(42)
    packet = rng.normal(size=(8, 12)) * 0.2
    twist = rng.normal(size=(8, 6))
    old_packet, old_twist = packet.copy(), twist.copy()
    latent = encode(packet, twist, 0.05)
    restored = decode(latent, twist, 0.05)
    np.testing.assert_allclose(
        Rotation.from_rotvec(restored[:, 3:6]).as_matrix(), Rotation.from_rotvec(packet[:, 3:6]).as_matrix(), atol=1e-14
    )
    for part in (slice(0, 3), slice(6, 12)):
        np.testing.assert_array_equal(restored[:, part], packet[:, part])
    q = Rotation.from_rotvec([0.4, -0.3, 0.2]).as_matrix()
    rotated = np.hstack([packet[:, j : j + 3] @ q.T for j in range(0, 12, 3)])
    rotated_twist = np.hstack([twist[:, j : j + 3] @ q.T for j in (0, 3)])
    np.testing.assert_allclose(encode(rotated, rotated_twist, 0.05)[:, 3:6], latent[:, 3:6] @ q.T, atol=1e-14)
    # Multiplication order matters: vector subtraction is not the Lie residual.
    assert np.max(np.abs(latent[:, 3:6] - (packet[:, 3:6] - 0.5 * (0.05 * twist[:, 3:] + packet[:, 9:])))) > 1e-4
    np.testing.assert_array_equal(packet, old_packet)
    np.testing.assert_array_equal(twist, old_twist)


def test_zero_path_residual_uses_predicted_endpoint_angular_velocity():
    latent = np.zeros((1, 12))
    current = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    latent[0, 9:] = [0.0, 0.1, 0.0]
    result = decode(latent, current, 0.05)
    np.testing.assert_allclose(result[0, 3:6], [0.025, 0.05, 0.0], atol=1e-14)
    for dt in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            decode(latent, current, dt)
    with pytest.raises(ValueError):
        encode(latent, current[:, :3], 0.05)
    latent[0, 0] = np.inf
    with pytest.raises(ValueError):
        decode(latent, current, 0.05)
