import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.coupled_gripper_preview import coupled_opening_preview

BOUNDS = dict(
    preview_steps=6,
    position_span_max_m=0.01,
    rotation_span_max_rad=0.1,
    measured_position_max_m=0.01,
    measured_rotation_max_rad=0.05,
)


def evaluate(poses, measured=None):
    return coupled_opening_preview(
        poses, np.linspace(0.2, 0.8, len(poses)), 0, poses[0] if measured is None else measured, **BOUNDS
    )


def test_accepts_small_pose_segment_without_changing_input():
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    poses[:, 0, 3] = np.linspace(0, 0.005, 7)
    before = poses.copy()
    result = evaluate(poses)
    assert result["advanced"] and result["reference"] == 0.8
    np.testing.assert_array_equal(poses, before)


@pytest.mark.parametrize(
    "failure", ["position_excursion", "rotation_excursion", "measured_position", "measured_rotation"]
)
def test_whole_segment_and_measured_guards_reject_individually(failure):
    poses = np.repeat(np.eye(4)[None], 7, axis=0)
    measured = np.eye(4)
    if failure == "position_excursion":
        poses[3, 0, 3] = 0.02
    if failure == "rotation_excursion":
        poses[3, :3, :3] = Rotation.from_euler("z", 0.2).as_matrix()
    if failure == "measured_position":
        measured[0, 3] = 0.02
    if failure == "measured_rotation":
        measured[:3, :3] = Rotation.from_euler("z", 0.1).as_matrix()
    result = evaluate(poses, measured)
    assert not result["advanced"] and result["reference"] == 0.2


def test_incompatible_endpoint_rosters_rejected():
    with pytest.raises(ValueError):
        coupled_opening_preview(np.repeat(np.eye(4)[None], 7, axis=0), [0.2], 0, np.eye(4), **BOUNDS)
