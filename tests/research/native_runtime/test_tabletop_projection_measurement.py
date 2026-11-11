"""Native geometry reader must not step or refresh the simulation itself."""

from types import SimpleNamespace

import mujoco
import numpy as np

from scripts.simulation.runtime.tabletop_projection_measurement import CurrentFingerGeometry


def test_native_geometry_is_current_and_read_only(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <asset><mesh name="finger_mesh" vertex="0 0 0  .02 0 0  0 .02 0  0 0 .02"/></asset>
          <worldbody>
            <geom name="table" type="box" size=".2 .2 .05" pos="0 0 -.05"/>
            <body pos="0 0 .1"><site name="tcp_site"/>
              <geom name="finger" type="mesh" mesh="finger_mesh"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def state():
        return SimpleNamespace(flatten=lambda: np.r_[data.time, data.qpos.copy(), data.qvel.copy()])

    sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data), get_state=state)
    env = SimpleNamespace(sim=sim, robots=[SimpleNamespace(controller=SimpleNamespace(eef_name="tcp_site"))])
    table = np.eye(4)
    table[2, 3] = -0.05
    reader = CurrentFingerGeometry(
        env,
        site_from_tcp=np.eye(4),
        table_pose=table,
        table_half_size=[0.2, 0.2, 0.05],
        table_name="table",
        finger_name="finger",
    )
    monkeypatch.setattr(
        mujoco, "mj_forward", lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected forward"))
    )
    monkeypatch.setattr(mujoco, "mj_step", lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected step")))
    actual = reader.capture()
    assert actual["vertices_tcp"].shape == (4, 3) and actual["time_s"] == 0
    np.testing.assert_allclose(actual["tcp"][:3, 3], [0.0, 0.0, 0.1], atol=1e-12, rtol=0)
