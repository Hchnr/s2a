"""The study must express the native slide axis in the measured TCP frame."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from scripts.tracking.response_models.directional_slide_boxes_study import slide_axis_in_tcp


def test_world_axis_is_rotated_into_tcp_without_translation():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    tcp = transform(np.array([0.4, -0.2, 0.8]), rotation)
    np.testing.assert_allclose(slide_axis_in_tcp(tcp, [0, 1, 0]), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(slide_axis_in_tcp(tcp, rotation @ [0, 0, 1]), [0, 0, 1], atol=1e-12)


def test_nonunit_or_invalid_axis_is_refused():
    with pytest.raises(ValueError, match="unit length"):
        slide_axis_in_tcp(np.eye(4), [0, 2, 0])
    with pytest.raises(ValueError):
        slide_axis_in_tcp(np.eye(4), [0, np.nan, 0])
    invalid = np.eye(4)
    invalid[0, 0] = 2
    with pytest.raises(ValueError):
        slide_axis_in_tcp(invalid, [0, 1, 0])
