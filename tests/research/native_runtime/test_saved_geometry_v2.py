"""Allow only tiny quaternion normalization, with every changed index recorded."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime.saved_geometry_v2 import reconstruct_endpoint


def fixture():
    model = SimpleNamespace(
        nq=7, nv=6, na=0, nsite=1, nbody=1, njnt=1, jnt_type=np.array([0]), jnt_qposadr=np.array([0])
    )
    data = SimpleNamespace(
        time=0.0,
        qpos=np.zeros(7),
        qvel=np.zeros(6),
        site_xmat=np.eye(3).reshape(1, 9),
        site_xpos=np.zeros((1, 3)),
        xpos=np.array([[0.3, 0, 0]]),
    )
    quaternion = np.array([1.0 + 2e-16, 0, 0, 0])
    state = np.concatenate(([0.5], [0.1, 0.2, 0.3], quaternion, np.zeros(6)))
    return model, data, state


def call(model, data, state, forward):
    return reconstruct_endpoint(model, data, state, site_id=0, body_id=0, site_from_tcp=np.eye(4), forward=forward)


def test_tiny_quaternion_normalization_is_recorded_and_does_not_change_other_state():
    model, data, state = fixture()

    def normalize(_model, value):
        value.qpos[3:7] /= np.linalg.norm(value.qpos[3:7])

    result = call(model, data, state, normalize)
    evidence = result["state_change"]
    assert evidence["changed_indices"] == [4]
    assert 0 < evidence["max_absolute_change"] < 1e-12
    assert evidence["quaternion_norms"][0]["joint_index"] == 0


@pytest.mark.parametrize("field", ["time", "position", "velocity", "rotation"])
def test_rejects_nonquaternion_or_non_normalization_changes(field):
    model, data, state = fixture()

    def corrupt(_model, value):
        if field == "time":
            value.time += 0.002
        elif field == "position":
            value.qpos[0] += 1e-4
        elif field == "velocity":
            value.qvel[0] += 1e-4
        else:
            value.qpos[4] += 1e-4

    with pytest.raises(ValueError, match="unchanged|normalization"):
        call(model, data, state, corrupt)


def test_real_cpu_mujoco_normalization_is_recorded_without_physics_step():
    import mujoco

    xml = (
        '<mujoco><worldbody><body name="object" pos="0.3 0 0">'
        '<geom type="sphere" size="0.02"/></body><body name="moving"><freejoint/>'
        '<geom type="sphere" size="0.02" mass="1"/><site name="tip"/></body></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    state = np.zeros(1 + model.nq + model.nv)
    state[4] = 1 + 2e-16
    result = reconstruct_endpoint(
        model,
        data,
        state,
        site_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")),
        body_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")),
        site_from_tcp=np.eye(4),
        forward=mujoco.mj_forward,
    )
    assert result["state_change"]["changed_indices"] == [4]
    assert result["state_change"]["max_absolute_change"] == pytest.approx(2.220446049250313e-16)
    assert data.time == 0
