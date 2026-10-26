import numpy as np
import pytest

from s2a.xml_site_kinematics import XMLSiteKinematics


def test_offset_joint_pivot_and_world_velocity():
    xml = '<mujoco><compiler angle="radian"/><worldbody><body><joint axis="0 0 1" pos="1 0 0"/><site name="tcp" pos="2 0 0"/></body></worldbody></mujoco>'
    pose, twist = XMLSiteKinematics(xml, "tcp").endpoint([0, np.pi / 2, 2])
    np.testing.assert_allclose(pose[:3, 3], [1, 1, 0], atol=1e-15)
    np.testing.assert_allclose(twist, [-2, 0, 0, 0, 0, 2], atol=1e-15)


def test_free_object_offsets_do_not_shift_robot_indices_and_joint_ref_is_respected():
    xml = '<mujoco><compiler angle="radian"/><worldbody><body><freejoint/></body><body pos="0.5 0.2 0" quat="0.7071067811865476 0 0 0.7071067811865476"><body><joint axis="0 0 1" ref="0.3"/><site name="tcp" pos="1 0 0"/></body></body></worldbody></mujoco>'
    model = XMLSiteKinematics(xml, "tcp")
    state = np.zeros(1 + model.nq + model.nv)
    state[1 + 7] = 0.3
    state[1 + model.nq + 6] = 2
    pose, twist = model.endpoint(state)
    np.testing.assert_allclose(pose[:3, 3], [0.5, 1.2, 0], atol=1e-15)
    np.testing.assert_allclose(twist, [-2, 0, 0, 0, 0, 2], atol=1e-15)


def test_unavailable_site_and_wrong_state_are_rejected():
    xml = '<mujoco><compiler angle="radian"/><worldbody><body><site name="tcp"/></body></worldbody></mujoco>'
    with pytest.raises(ValueError):
        XMLSiteKinematics(xml, "missing")
    with pytest.raises(ValueError):
        XMLSiteKinematics(xml, "tcp").endpoint([0, 1])
