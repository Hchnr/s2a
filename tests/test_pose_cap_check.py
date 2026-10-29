"""Reject candidates that only pass a dimensionless 1e-8 residual tolerance."""

import numpy as np
import pytest

from s2a.pose_cap_check import within_pose_caps


@pytest.mark.parametrize("component,cap", [(0, 0.01), (3, 0.05)])
def test_absolute_position_and_rotation_violation_is_not_hidden_by_normalization(component, cap):
    error = np.zeros(6)
    caps = np.array([0.01, 0.05])
    error[component] = cap + (5e-11 if component == 0 else 2e-10)
    assert 1 - np.linalg.norm(error[component : component + 1]) / cap >= -1e-8
    assert not within_pose_caps(error, caps)
    error[component] = cap - 1e-10
    assert within_pose_caps(error, caps)
