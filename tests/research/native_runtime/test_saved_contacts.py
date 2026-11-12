"""Check native contact pair semantics, including reversed pairs and real MuJoCo."""

from types import SimpleNamespace

import pytest

from s2a.research.native_runtime.saved_contacts import summarize_contacts, target_contacts


def fake_scene(pairs):
    names = ["target", "visual", "left", "right", "table"]
    model = SimpleNamespace(ngeom=5, geom_bodyid=[1, 1, 2, 3, 0], geom_group=[0, 1, 0, 0, 0])
    data = SimpleNamespace(ncon=len(pairs), contact=[SimpleNamespace(geom1=a, geom2=b, dist=d) for a, b, d in pairs])
    mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_GEOM=0),
        mj_name2id=lambda _model, _kind, name: names.index(name) if name in names else -1,
        mj_id2name=lambda _model, _kind, index: names[index],
    )
    return model, data, mujoco


def inspect(model, data, mujoco):
    return target_contacts(
        model, data, mujoco=mujoco, body_id=1, target_names=["target"], left_pad="left", right_pad="right"
    )


def test_contact_pairs_match_native_grasp_semantics_in_either_order():
    model, data, mujoco = fake_scene([(2, 0, 0.001), (0, 3, -0.002), (4, 0, 0.0), (1, 2, -0.01)])
    result = inspect(model, data, mujoco)
    assert result["bilateral"] is True
    assert result["left_pad"] == [dict(target_geom="target", other_geom="left", distance_m=0.001)]
    assert result["right_pad"] == [dict(target_geom="target", other_geom="right", distance_m=-0.002)]
    assert result["other"] == [dict(target_geom="target", other_geom="table", distance_m=0.0)]


def test_one_pad_or_table_contact_is_not_bilateral_and_mismatches_are_reported():
    model, data, mujoco = fake_scene([(0, 2, -0.001), (0, 4, -0.001)])
    result = inspect(model, data, mujoco)
    assert result["bilateral"] is False
    points = [dict(endpoint_step=7, contacts=result, target_grasp=True)]
    assert summarize_contacts(points) == dict(
        endpoint_counts=dict(left_pad=1, right_pad=0, bilateral=0, other=1), grasp_label_mismatch_steps=[7]
    )


def test_target_geometry_must_match_exact_registered_group_and_distance_must_be_finite():
    model, data, mujoco = fake_scene([(0, 2, float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        inspect(model, data, mujoco)
    model.geom_group[1] = 0
    with pytest.raises(ValueError, match="groups"):
        inspect(model, data, mujoco)
    with pytest.raises(ValueError, match="different pads"):
        target_contacts(
            model, data, mujoco=mujoco, body_id=1, target_names=["target"], left_pad="left", right_pad="left"
        )


def test_real_mujoco_forward_finds_two_pads_then_one_without_stepping():
    import mujoco

    xml = """<mujoco><worldbody>
      <body name="object"><freejoint/><geom name="target" type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="l" pos="-.12 0 0"><geom name="left" type="box" size=".05 .05 .05"/></body>
      <body name="r" pos=".12 0 0"><geom name="right" type="box" size=".05 .05 .05"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object"))
    kwargs = dict(mujoco=mujoco, body_id=body_id, target_names=["target"], left_pad="left", right_pad="right")
    mujoco.mj_forward(model, data)
    assert target_contacts(model, data, **kwargs)["bilateral"] is True
    right_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r"))
    model.body_pos[right_body, 0] = 0.7
    mujoco.mj_forward(model, data)
    result = target_contacts(model, data, **kwargs)
    assert result["left_pad"] and not result["right_pad"] and result["bilateral"] is False
    assert data.time == 0
