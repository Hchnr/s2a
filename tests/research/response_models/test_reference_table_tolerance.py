"""Bind the conservative certificate to the compiled collision mesh and source roster."""

import numpy as np

from s2a.geometry import transform
from scripts.tracking.response_models import reference_table_tolerance as study


def synthetic_probe():
    import mujoco

    from s2a.research.wam.tracking.preview.development_reference_geometry_roundoff import RoundoffBoundedProbe

    xml = """
    <mujoco>
      <asset><mesh name="finger_mesh"
        vertex="-0.01 -0.01 -0.02  0.01 -0.01 -0.02  0.01 0.01 -0.02  -0.01 0.01 -0.02
                -0.01 -0.01 0.02  0.01 -0.01 0.02  0.01 0.01 0.02  -0.01 0.01 0.02"
        face="0 2 1  0 3 2  4 5 6  4 6 7  0 1 5  0 5 4  3 7 6  3 6 2  0 4 7  0 7 3  1 2 6  1 6 5"/></asset>
      <worldbody>
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
            <geom name="gripper0_finger2_collision" type="mesh" mesh="finger_mesh" pos="0 0 -0.04" mass="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    probe = RoundoffBoundedProbe(model, "target", budget=4)
    state = np.r_[0.0, np.zeros(model.nq + model.nv + model.na)]
    target = transform(np.array([0.0, 0.0, 0.03]), np.eye(3))
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper0_finger2_collision")
    mesh = model.geom_dataid[finger]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float).copy()
    probe.restore(state)
    probe.place(target)
    contacts = study.geometry.matching_contacts(probe.data, table, finger)
    return probe, state, target, table, finger, vertices, contacts


def test_compiled_mesh_witness_is_inside_box_under_pose_tolerances():
    probe, state, target, table, finger, vertices, contacts = synthetic_probe()
    config = {"position_tolerance_m": 0.01, "rotation_tolerance_rad": 0.05, "certificate_margin_m": 1e-9}
    previous = {"target_proxy_finger_table_contact_distances_m": contacts}
    row = study.evaluate_target(probe, state, target, np.eye(4), table, finger, vertices, config, previous)
    proof = row["certificate"]
    assert contacts and proof["certified"] is True and proof["certificate_slack_m"] > 1e-9
    assert np.all(proof["face_margins_m"] > proof["motion_bound_m"])
    assert probe.queries == 4 and probe.data.time == 0


def test_fixed_scope_reuses_completed_states_and_previously_audited_targets():
    inputs = study.load_inputs()
    item = inputs["reference_table_tolerance"]
    assert item["config"]["steps"] == list(range(120, 127))
    assert [row["step"] for row in item["previous_rows"]] == list(range(120, 127))
    assert item["config"]["physics_steps"] == item["config"]["world_model_calls"] == 0
