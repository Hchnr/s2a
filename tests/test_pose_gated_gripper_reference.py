import numpy as np
from scipy.spatial.transform import Rotation

from s2a.pose_gated_gripper_reference import align_closure


def run(poses, values):
    return align_closure(
        np.eye(4),
        poses,
        values,
        execute_steps=8,
        closing_threshold=0.35,
        position_tolerance_m=0.02,
        rotation_tolerance_rad=0.15,
    )


def test_only_generated_deferred_closure_at_reached_pose_is_advanced():
    poses = np.tile(np.eye(4), (40, 1, 1))
    poses[12, 0, 3] = 0.01
    values = np.ones(40)
    values[12] = 0.2
    actual = run(poses, values)
    assert actual["selected_index"] == 12
    np.testing.assert_array_equal(actual["openings"][:8], np.full(8, 0.2))
    np.testing.assert_array_equal(actual["openings"][8:], values[8:])
    np.testing.assert_array_equal(values[:8], np.ones(8))


def test_position_and_so3_gates_prevent_premature_closure():
    for position, angle in ((0.03, 0), (0, 0.2)):
        poses = np.tile(np.eye(4), (40, 1, 1))
        poses[12, 0, 3] = position
        poses[12, :3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        values = np.ones(40)
        values[12] = 0.2
        actual = run(poses, values)
        assert actual["selected_index"] is None
        np.testing.assert_array_equal(actual["openings"], values)


def test_existing_executed_closure_and_no_closure_remain_unchanged():
    poses = np.tile(np.eye(4), (40, 1, 1))
    for index in (None, 3):
        values = np.ones(40)
        if index is not None:
            values[index] = 0.2
        actual = run(poses, values)
        assert actual["selected_index"] is None
        np.testing.assert_array_equal(actual["openings"], values)
