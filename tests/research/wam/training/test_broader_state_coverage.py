"""A joint match must use one real task-matched anchor for all dimensions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.wam.training.broader_state_coverage import first_joint_anchor


def anchor(name, x, angle, opening, *, task=0):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    pose[0, 3] = x
    return dict(anchor_id=name, source_kind="training", task_index=task, tcp_pose=pose, opening=opening)


def test_all_descriptors_must_match_one_anchor_from_the_same_task():
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.5)
    rows = [
        anchor("near_bad_rotation", 0.01, 0.8, 0.5),
        anchor("far_good_rotation", 0.08, 0.1, 0.5),
        anchor("wrong_task", 0.0, 0.0, 0.5, task=1),
        anchor("near_bad_opening", 0.03, 0.1, 0.9),
    ]
    assert first_joint_anchor(query, rows, max_position_m=0.05, max_rotation_rad=0.5, max_opening_abs=0.2) is None
    result = first_joint_anchor(query, rows, max_position_m=0.1, max_rotation_rad=0.5, max_opening_abs=0.2)
    assert result["anchor_id"] == "far_good_rotation"
    assert result["position_m"] == pytest.approx(0.08)
    assert result["rotation_rad"] == pytest.approx(0.1)
    assert result["opening_abs"] == 0


def test_first_matching_anchor_is_stable_and_limits_are_checked():
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.5)
    rows = [anchor("first", 0.04, 0.49, 0.5), anchor("closer", 0.01, 0.1, 0.5)]
    assert (
        first_joint_anchor(query, rows, max_position_m=0.05, max_rotation_rad=0.5, max_opening_abs=0.2)["anchor_id"]
        == "first"
    )
    with pytest.raises(ValueError, match="limits"):
        first_joint_anchor(query, rows, max_position_m=float("nan"), max_rotation_rad=0.5, max_opening_abs=0.2)
    with pytest.raises(ValueError, match="same-task"):
        first_joint_anchor(
            dict(query, task_index=2), rows, max_position_m=0.05, max_rotation_rad=0.5, max_opening_abs=0.2
        )


@pytest.mark.parametrize("angle", [0.0, 0.5, np.pi - 1e-8, np.pi])
def test_angle_agrees_with_scipy_away_from_threshold_ambiguities(angle):
    query = dict(task_index=0, tcp_pose=np.eye(4), opening=0.5)
    row = anchor("actual", 0.01, angle, 0.5)
    result = first_joint_anchor(query, [row], max_position_m=0.05, max_rotation_rad=np.pi, max_opening_abs=0.2)
    assert result["rotation_rad"] == pytest.approx(Rotation.from_matrix(row["tcp_pose"][:3, :3]).magnitude(), abs=1e-12)
