"""Spherical-cap axis bounds contain allowed rotation samples."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.verification.response_models.audit_slide_range_table_collision import spherical_cap_axis_interval


def test_axis_interval_covers_random_rotations_and_extremes():
    assert spherical_cap_axis_interval(1.0, 0.1)[1] == pytest.approx(1.0)
    assert spherical_cap_axis_interval(-1.0, 0.1)[0] == pytest.approx(-1.0)
    rng = np.random.default_rng(42)
    for _ in range(256):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        rotation_axis = rng.normal(size=3)
        rotation_axis /= np.linalg.norm(rotation_axis)
        low, high = spherical_cap_axis_interval(float(axis @ normal), 0.05)
        rotated = Rotation.from_rotvec(rotation_axis * rng.uniform(0, 0.05)).apply(axis)
        assert low - 1e-12 <= rotated @ normal <= high + 1e-12
    with pytest.raises(ValueError):
        spherical_cap_axis_interval(0.0, np.pi + 0.1)
