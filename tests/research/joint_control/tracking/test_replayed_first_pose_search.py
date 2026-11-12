import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.references.generated_twist import reference_world_twist
from scripts.tracking.trajectory_search.audit_replayed_first_pose_search import generated_horizon


def test_generated_horizon_matches_the_original_segment_clock():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, 0, 3] = np.arange(40, dtype=float) ** 2 * 1e-4
    poses[:, :3, :3] = Rotation.from_rotvec(
        np.column_stack((np.zeros(40), np.zeros(40), np.arange(40) * 0.01))
    ).as_matrix()
    anchor = np.eye(4)
    segment = np.concatenate((poses[16:], np.repeat(poses[-1:], 16, axis=0)))
    segment_anchor = np.eye(4)
    segment_anchor[1, 3] = 0.2
    expected_twists = np.asarray(
        [reference_world_twist(segment, index, segment_anchor, 0.05)["twist"] for index in range(4, 12)]
    )
    targets, twists = generated_horizon(poses, anchor, 88, 108)
    np.testing.assert_array_equal(targets, segment[4:12])
    np.testing.assert_allclose(twists, expected_twists, atol=1e-14, rtol=0)
    with pytest.raises(ValueError):
        generated_horizon(poses, anchor, 88, 121)
