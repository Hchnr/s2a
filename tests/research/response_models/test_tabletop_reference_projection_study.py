"""Static study preserves recorded state and distinguishes refusal from clearance."""

from itertools import product

import mujoco
import numpy as np

from s2a.geometry import transform
from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe
from scripts.tracking.response_models import reference_table_geometry as geometry
from scripts.tracking.response_models import tabletop_reference_projection as study


def scene():
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
    probe = RoundoffBoundedProbe(model, "target", budget=6)
    state = np.r_[0.0, np.zeros(model.nq + model.nv + model.na)]
    target = transform(np.array([0.0, 0.0, 0.04]), np.eye(3))
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper0_finger2_collision")
    vertices = np.array(list(product((-0.012, 0.012), (-0.012, 0.012), (-0.02, 0.02))))
    prior = geometry.inspect_target(probe, state, target, np.eye(4), table, finger)
    assert prior["target_proxy_finger_table_intersection"]
    return probe, state, target, table, finger, vertices, prior


def config(limit):
    return {
        "clearance_m": 0.001,
        "numerical_margin_m": 1e-9,
        "max_translation_m": limit,
        "candidates": [
            {
                "name": "nominal",
                "position_uncertainty_m": 0.0,
                "rotation_uncertainty_rad": 0.0,
                "shape_uncertainty_m": 0.0,
            },
            {
                "name": "pose_bounded",
                "position_uncertainty_m": 0.01,
                "rotation_uncertainty_rad": 0.05,
                "shape_uncertainty_m": 0.0,
            },
        ],
    }


def test_recorded_collision_is_removed_by_accepted_static_projection():
    probe, state, target, table, finger, vertices, prior = scene()
    row = study.inspect_state(probe, state, target, np.eye(4), table, finger, vertices, config(0.05), prior)
    assert row["original_proxy_finger_table_contact_distances_m"]
    assert row["candidates"]["nominal"]["accepted"]
    assert row["candidates"]["pose_bounded"]["accepted"]
    for candidate in row["candidates"].values():
        assert candidate["projected_proxy_finger_table_contact_distances_m"] == []
        assert candidate["current_tcp_error_to_projected"] is not None
    assert (
        row["candidates"]["nominal"]["required_translation_m"]
        < row["candidates"]["pose_bounded"]["required_translation_m"]
    )
    assert probe.queries == 6 and probe.data.time == 0.0


def test_refusal_never_creates_a_projected_target_or_claims_a_contact_result():
    probe, state, target, table, finger, vertices, prior = scene()
    row = study.inspect_state(probe, state, target, np.eye(4), table, finger, vertices, config(0.005), prior)
    for candidate in row["candidates"].values():
        assert candidate["status"] == "correction_exceeds_limit"
        assert candidate["projected_tcp"] is None
        assert candidate["projected_proxy_finger_table_contact_distances_m"] is None
        assert candidate["current_tcp_error_to_projected"] is None
    assert probe.queries == 4 and probe.data.time == 0.0
