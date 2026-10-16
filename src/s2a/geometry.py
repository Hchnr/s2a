"""Column-rot6d, SE(3), and base-frame flange errors (metres/radians)."""

import numpy as np
from scipy.spatial.transform import Rotation


def finite(value, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError(f"expected finite array of shape {shape}, got {a.shape}")
    return a


def rot6d_to_matrix(value):
    """Gram-Schmidt on concatenated first/second columns; reject degeneracy."""
    a = finite(value)
    if a.shape[-1] != 6:
        raise ValueError("rotation must have six columns-encoded components")
    x, y = a[..., :3], a[..., 3:]
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("degenerate first rotation column")
    x = x / norm
    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("degenerate second rotation column")
    y = y / norm
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def matrix_to_rot6d(matrix):
    r = finite(matrix)
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def transform(position, rotation):
    p, r = finite(position), finite(rotation)
    if p.shape[-1] != 3 or r.shape != p.shape[:-1] + (3, 3):
        raise ValueError("incompatible position/rotation shapes")
    t = np.broadcast_to(np.eye(4), p.shape[:-1] + (4, 4)).copy()
    t[..., :3, :3], t[..., :3, 3] = r, p
    return t


def validate_transform(value):
    t = finite(value)
    if t.shape[-2:] != (4, 4):
        raise ValueError("expected homogeneous transform")
    r = t[..., :3, :3]
    if not (
        np.allclose(t[..., 3, :], [0, 0, 0, 1], atol=1e-8)
        and np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-7)
        and np.allclose(np.linalg.det(r), 1, atol=1e-7)
    ):
        raise ValueError("invalid SE(3) transform")
    return t


def inverse(t):
    t = validate_transform(t)
    r = t[..., :3, :3].swapaxes(-1, -2)
    return transform(-(r @ t[..., :3, 3, None])[..., 0], r)


def pose9_to_transform(value):
    a = finite(value)
    return transform(a[..., :3], rot6d_to_matrix(a[..., 3:9]))


def euler_pose_to_transform(value):
    a = finite(value)
    if a.shape[-1] != 6:
        raise ValueError("expected xyz + extrinsic xyz Euler radians")
    return transform(a[..., :3], Rotation.from_euler("xyz", a[..., 3:]).as_matrix())


def transform_to_euler_pose(t):
    t = validate_transform(t)
    return np.concatenate([t[..., :3, 3], Rotation.from_matrix(t[..., :3, :3]).as_euler("xyz")], axis=-1)


def pose_error(desired, measured):
    """Base-frame translation and principal SO(3) log; reject pi ambiguity."""
    desired, measured = validate_transform(desired), validate_transform(measured)
    r = desired[..., :3, :3] @ measured[..., :3, :3].swapaxes(-1, -2)
    w = Rotation.from_matrix(r).as_rotvec()
    return np.concatenate([desired[..., :3, 3] - measured[..., :3, 3], w], axis=-1)


def advance(t, base_velocity, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    t, v = validate_transform(t), finite(base_velocity, (6,))
    return transform(t[:3, 3] + dt * v[:3], Rotation.from_rotvec(dt * v[3:]).as_matrix() @ t[:3, :3])


def integrate_local(anchor, deltas):
    t = validate_transform(anchor).copy()
    poses = [t.copy()]
    for delta in deltas:
        t = t @ pose9_to_transform(delta)
        poses.append(t.copy())
    return np.array(poses)


def error_summary(desired, measured):
    e = pose_error(desired, measured)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(e[..., :3] ** 2, axis=-1)))),
        "position_max_m": float(np.max(np.linalg.norm(e[..., :3], axis=-1))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(np.sum(e[..., 3:] ** 2, axis=-1)))),
        "rotation_max_rad": float(np.max(np.linalg.norm(e[..., 3:], axis=-1))),
    }
