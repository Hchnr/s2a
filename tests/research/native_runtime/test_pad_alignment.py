"""Verify box support intervals and the defined signed jaw-axis coordinates."""

import numpy as np
import pytest

from s2a.research.native_runtime.pad_alignment import alignment, project_box


def test_axis_projection_is_exact_for_a_rotated_box():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = project_box([2, 0, 0], rotation, [0.5, 0.2, 0.3], [1, 0, 0], [1, 0, 0])
    np.testing.assert_allclose(result, [0.8, 1.2])
    with pytest.raises(ValueError, match="unit projection axis"):
        project_box([0, 0, 0], rotation, [1, 1, 1], [0, 0, 0], [2, 0, 0])
    with pytest.raises(ValueError, match="proper rotation"):
        project_box([0, 0, 0], -np.eye(3), [1, 1, 1], [0, 0, 0], [1, 0, 0])


def test_signed_axis_and_perpendicular_distances_use_actual_pad_centers():
    result = alignment([-1, 0, 0], [1, 0, 0], [0.25, 0.3, 0.4], [0.1, 0, 0], np.eye(3))
    np.testing.assert_allclose(result["jaw_axis_world"], [1, 0, 0])
    assert result["pad_center_separation_m"] == pytest.approx(2)
    assert result["body_axis_projection_m"] == pytest.approx(0.25)
    assert result["body_axis_perpendicular_m"] == pytest.approx(0.5)
    np.testing.assert_allclose(result["body_from_site_in_site_m"], [0.15, 0.3, 0.4])
    with pytest.raises(ValueError, match="Distinct pad centers"):
        alignment([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], np.eye(3))


def test_real_cpu_mujoco_reconstruction_reads_box_geometry_without_physics(monkeypatch):
    import mujoco

    from scripts.simulation.runtime import diagnose_bowl_pad_alignment as diagnosis

    xml = """<mujoco><worldbody>
      <body name="object"><freejoint/><geom name="target" type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="l" pos="-.12 0 0"><geom name="left" type="box" size=".05 .05 .05"/><site name="site"/></body>
      <body name="r" pos=".12 0 0"><geom name="right" type="box" size=".05 .05 .05"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config = dict(left_pad="left", right_pad="right", target_contact_geoms=["target"])
    monkeypatch.setattr(diagnosis, "FIXED", config)
    state = np.r_[0.0, model.qpos0, np.zeros(model.nv)]
    point = diagnosis.reconstruct_endpoint(
        model,
        data,
        state,
        site_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site")),
        body_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")),
        site_from_tcp=np.eye(4),
        forward=mujoco.mj_forward,
    )
    assert point["contacts"]["bilateral"] is True
    assert point["pad_alignment"]["pad_center_separation_m"] == pytest.approx(0.24)
    np.testing.assert_allclose(point["pad_alignment"]["target_axis_interval_m"], [-0.1, 0.1])
    assert data.time == 0
