"""Independently read free-joint qpos and aggregate contact normals."""

import numpy as np
import pytest

from scripts.verification.response_models.audit_early_box_face_bowl_effect import (
    direct_normal_forces,
    raw_free_pose,
    raw_motion,
)


def test_raw_free_joint_pose_and_relative_motion():
    state = np.zeros(1 + 41)
    state[1:8] = [1, 2, 3, 1, 0, 0, 0]
    position, rotation = raw_free_pose(state, 0)
    np.testing.assert_array_equal(position, [1, 2, 3])
    np.testing.assert_array_equal(rotation, np.eye(3))
    result = raw_motion(
        (position, rotation),
        (position, rotation),
        (position + [0.01, 0, 0], rotation),
        (position + [0.01, 0, 0], rotation),
    )
    assert result["bowl_world"]["translation_m"] == pytest.approx(0.01)
    assert result["cheese_relative_to_bowl"]["translation_m"] == 0
    with pytest.raises(ValueError):
        raw_free_pose(state, 40)


def test_direct_force_counts_each_sample_once():
    contact = dict(active_constraint=True, geom1="finger", geom2="bowl", wrench_contact_frame=[2, 0, 0, 0, 0, 0])
    summary = direct_normal_forces([{"contacts": [contact, contact]}, {"contacts": []}], {"finger", "bowl"})
    assert summary["active_inner_samples"] == 1 and summary["active_contact_records"] == 2
    assert summary["maximum_sum_normal_n_per_sample"] == 4
