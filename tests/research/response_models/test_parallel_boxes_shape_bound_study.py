"""Reject invalid kinematic assumptions before using a gripper range as a bound."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tracking.response_models.parallel_boxes_shape_bound_study import (
    body_path,
    single_slide_topology,
)


def model():
    return SimpleNamespace(
        nbody=4,
        body_parentid=np.array([0, 0, 1, 1]),
        body_jntadr=np.array([-1, -1, 0, -1]),
        body_jntnum=np.array([0, 0, 1, 0]),
        jnt_type=np.array([2]),
        jnt_limited=np.array([1]),
        jnt_axis=np.array([[0.0, 1.0, 0.0]]),
        jnt_range=np.array([[-0.04, 0.0]]),
    )


def test_slide_topology_accepts_one_moving_finger_and_fixed_tcp():
    native = model()
    result = single_slide_topology(native, finger_body=2, site_body=3, slide=0, slide_type=2)
    assert result["common_ancestor_body_id"] == 1
    assert result["finger_chain_body_ids"] == [2]
    assert result["site_chain_body_ids"] == [3]
    np.testing.assert_array_equal(result["joint_interval_m"], [-0.04, 0.0])


def test_slide_topology_rejects_a_moving_tcp_or_inconsistent_joint_type():
    native = model()
    native.body_jntadr[3], native.body_jntnum[3] = 0, 1
    with pytest.raises(ValueError, match="exactly one"):
        single_slide_topology(native, 2, 3, 0, 2)
    native = model()
    native.jnt_type[0] = 3
    with pytest.raises(ValueError, match="exactly one"):
        single_slide_topology(native, 2, 3, 0, 2)
    native = model()
    native.jnt_limited[0] = 0
    with pytest.raises(ValueError, match="exactly one"):
        single_slide_topology(native, 2, 3, 0, 2)


def test_slide_topology_rejects_nonunit_axis_and_invalid_ancestry():
    native = model()
    native.jnt_axis[0] *= 2
    with pytest.raises(ValueError, match="unit slide"):
        single_slide_topology(native, 2, 3, 0, 2)
    native = model()
    native.body_parentid[1] = 2
    with pytest.raises(ValueError, match="ancestry"):
        body_path(native, 2)
