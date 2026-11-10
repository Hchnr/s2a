"""The repair handles floating-point quaternion normalization, without hiding movement."""

import mujoco
import numpy as np
import pytest

from s2a.research.wam.tracking.preview.development_reference_geometry import StaticReferenceProbe
from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe


def synthetic_model_and_state():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="robot0_base"><site name="gripper0_grip_site"/>
            <body pos="-.024 0 0"><joint name="gripper0_finger_joint1" type="slide" axis="1 0 0"/>
              <geom name="gripper0_finger1_pad_collision" type="box" size=".01 .01 .01"/></body>
            <body pos=".024 0 0"><joint name="gripper0_finger_joint2" type="slide" axis="1 0 0"/>
              <geom name="gripper0_finger2_pad_collision" type="box" size=".01 .01 .01"/></body>
          </body>
          <body name="target_body"><freejoint name="object_free"/>
            <geom name="target_collision" type="sphere" size=".02"/></body>
        </worldbody></mujoco>
        """
    )
    state = np.r_[0.0, model.qpos0, np.zeros(model.nv + model.na)]
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    address = int(model.jnt_qposadr[joint])
    state[1 + address + 3 : 1 + address + 7] = [
        -0.31899323748197655,
        -0.5067675584753463,
        0.3835940640598035,
        0.7030544432231313,
    ]
    return model, state


def test_repeat_forward_normalization_is_recorded_and_bounded():
    model, state = synthetic_model_and_state()
    old = StaticReferenceProbe(model, "target")
    old.restore(state)
    with pytest.raises(AssertionError):
        old.place(old.site_transform())

    model, state = synthetic_model_and_state()
    repaired = RoundoffBoundedProbe(model, "target")
    repaired.restore(state)
    contacts = repaired.contacts()
    repaired.place(repaired.site_transform())
    assert repaired.contacts() == contacts
    assert 0 < repaired.max_object_rotation_change <= 1e-12
    assert 0 < repaired.max_qpos_change_during_placement <= 1e-12


def test_rotation_drift_larger_than_registered_bound_is_rejected():
    model, state = synthetic_model_and_state()
    probe = RoundoffBoundedProbe(model, "target")
    probe.restore(state)
    position = probe.data.xpos[probe.object_body].copy()
    rotation = probe.data.xmat[probe.object_body].copy()
    qpos = probe.data.qpos.copy()
    probe.data.xmat[probe.object_body, 0] += 1e-8
    with pytest.raises(AssertionError):
        probe.check_object_unchanged(position, rotation, qpos)
