import numpy as np
import pytest

from s2a.libero_audit import post_action_reference


def test_post_action_endpoints_include_first_and_last_without_reanchoring():
    position = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]])
    poses, openings = post_action_reference(position, np.zeros((3, 3)), np.array([[0.04, -0.04]] * 3))
    np.testing.assert_array_equal(poses[:, :3, 3], position)
    np.testing.assert_array_equal(openings, [1, 1, 1])
    with pytest.raises(ValueError, match="shapes"):
        post_action_reference(position[:-1], np.zeros((3, 3)), np.zeros((3, 2)))
