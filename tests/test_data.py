import numpy as np
import pytest
from numpy.testing import assert_array_equal

from s2a.data import align_original, pose_transition_mask, split_indices


def test_pose_mask_does_not_confuse_gripper_only_terminal_or_control_mask():
    rows = {
        "action_valid": np.ones(4, dtype=bool),
        "action_dim_mask": np.ones((4, 128), dtype=bool),
        "action_control_mask": np.zeros((4, 128), dtype=bool),
        "observation.state_mask": np.ones((4, 128), dtype=bool),
        "physical_time_valid": np.array([1, 1, 1, 0], dtype=bool),
        "transition_dt": np.array([0.07, 0.08, 0.07, 0]),
        "frame_index": np.arange(4),
    }
    rows["action_dim_mask"][-1, :9] = False
    assert_array_equal(pose_transition_mask(rows), [1, 1, 1, 0])
    rows["observation.state_mask"][2, 3] = False
    assert_array_equal(pose_transition_mask(rows), [1, 0, 0, 0])
    rows["frame_index"][1] = 3
    assert not pose_transition_mask(rows)[0]


def test_join_uses_unique_provenance_ids_not_row_order_or_time():
    rows = {
        "frame_index": np.array([30, 31]),
        "raw_frame_index": np.array([7, 9]),
        "source_frame_index": np.array([7, 9]),
    }
    original = {"frame_index": np.array([9, 7, 8]), "command": np.array([0.9, 0.7, 0.8])}
    assert_array_equal(align_original(rows, original)["command"], [0.7, 0.9])
    original["frame_index"] = np.array([9, 7, 7])
    with pytest.raises(ValueError, match="duplicate"):
        align_original(rows, original)
    rows["raw_frame_index"][0] = 8
    with pytest.raises(ValueError):
        align_original(rows, original)


def test_temporal_split_excludes_cross_boundary_history_labels_and_gaps():
    f = np.arange(27, 331)
    ids = split_indices(f, [220, 260], history=3, lag=5)
    assert_array_equal(f[ids], np.arange(223, 255))
    for i in ids:
        assert f[i - 3] >= 220
        assert f[i + 5 + 1] <= 260
    f[225 - 27] = 999
    ids = split_indices(f, [220, 260], history=3, lag=5)
    assert all(999 not in f[i - 3 : i + 7] for i in ids)
