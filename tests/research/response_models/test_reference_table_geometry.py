"""Static target geometry must preserve state and identify real pair penetration."""

from types import SimpleNamespace

import numpy as np

from s2a.geometry import transform
from scripts.tracking.response_models import reference_table_geometry as geometry


def test_matching_contacts_filters_unrelated_or_positive_distance():
    data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=1, geom2=2, dist=-0.01),
            SimpleNamespace(geom1=2, geom2=1, dist=0.0),
            SimpleNamespace(geom1=1, geom2=2, dist=0.001),
            SimpleNamespace(geom1=1, geom2=3, dist=-0.02),
        ],
    )
    assert geometry.matching_contacts(data, 1, 2) == [-0.01, 0.0]


def test_static_target_proxy_detects_table_intersection_without_advancing_time():
    import mujoco

    from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe

    xml = """
    <mujoco><worldbody>
      <geom name="table_collision" type="box" pos="0 0 -0.05" size="0.3 0.3 0.05"/>
      <body name="object" pos="0.2 0.2 0.1"><geom name="target_collision" type="sphere" size="0.01"/></body>
      <body name="robot0_base" pos="0 0 0.15">
        <site name="gripper0_grip_site"/>
        <body name="finger1" pos="-0.05 0 0">
          <joint name="gripper0_finger_joint1" type="slide" axis="1 0 0"/>
          <geom name="gripper0_finger1_pad_collision" type="box" pos="0 0 -0.04" size="0.01 0.01 0.02" mass="1"/>
        </body>
        <body name="finger2" pos="0.05 0 0">
          <joint name="gripper0_finger_joint2" type="slide" axis="1 0 0"/>
          <geom name="gripper0_finger2_pad_collision" type="box" pos="0 0 -0.04" size="0.01 0.01 0.02" mass="1"/>
          <geom name="gripper0_finger2_collision" type="box" pos="0 0 -0.04" size="0.012 0.012 0.02" mass="1"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    probe = RoundoffBoundedProbe(model, "target", budget=2)
    state = np.r_[0.0, np.zeros(model.nq + model.nv + model.na)]
    target = transform(np.array([0.0, 0.0, 0.04]), np.eye(3))
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper0_finger2_collision")
    row = geometry.inspect_target(probe, state, target, np.eye(4), table, finger)
    assert row["current_finger_table_contact_distances_m"] == []
    assert row["target_proxy_finger_table_intersection"] is True
    assert min(row["target_proxy_finger_table_contact_distances_m"]) < 0
    assert probe.queries == 2 and probe.data.time == 0.0
    assert row["finger_pose_transform_max_difference"] <= 1e-9
    assert row["qpos_max_difference"] == row["table_position_max_difference"] == 0


def test_source_roster_contains_only_available_states_and_references():
    inputs = geometry.load_inputs()["reference_table_geometry"]
    assert len(inputs["states"]) == len(inputs["targets"]) == 47
    assert inputs["config"]["executed_periods"] == 46 and inputs["config"]["last_step"] == 126
    assert all(
        state.shape == (79,) and target.shape == (4, 4)
        for state, target in zip(inputs["states"], inputs["targets"], strict=True)
    )
