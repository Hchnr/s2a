"""Bind model inputs to actual raw current states and independent XML FK."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.xml_site_kinematics import XMLSiteKinematics


def verify_inputs(xml, data, rows):
    assert len(rows) + 1 == len(data["sim_states"])
    first = rows[0]["current_wrench"]
    kin = XMLSiteKinematics(xml, first["control_site"])
    sensor = XMLSiteKinematics(xml, first["sensor_site"])
    indices = {n.get("name"): v for n, v in kin.indices.items()}
    for k, row in enumerate(rows):
        packet = {name: np.asarray(value) for name, value in row["qualified_forward_input"].items()}
        state = data["sim_states"][k]
        pose, twist = kin.endpoint(state)
        np.testing.assert_allclose(packet["x_site_pose"][0], pose, rtol=0, atol=1e-9)
        np.testing.assert_allclose(packet["x_site_twist"][0], twist, rtol=0, atol=1e-8)
        for prefix, names in (
            ("joint", [f"robot0_joint{i}" for i in range(1, 8)]),
            ("finger", [f"gripper0_finger_joint{i}" for i in (1, 2)]),
        ):
            q = [state[1 + indices[n][0]] for n in names]
            v = [state[1 + kin.nq + indices[n][1]] for n in names]
            qkey, vkey = (
                ("x_joint_position", "x_joint_velocity") if prefix == "joint" else ("x_finger_qpos", "x_finger_qvel")
            )
            np.testing.assert_allclose(packet[qkey][0], q, rtol=0, atol=1e-12)
            np.testing.assert_allclose(packet[vkey][0], v, rtol=0, atol=1e-12)
        offsets = data["gripper_current_action_offsets"]
        motor = data["gripper_current_action_flat"][offsets[k] : offsets[k + 1]]
        if len(motor) == 1:
            motor = np.repeat(motor, 2)
        np.testing.assert_allclose(packet["x_motor_before"][0], motor, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            packet["x_previous_goal_pos"][0], row["controller_goal_position_before"], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(packet["x_previous_goal_ori"][0], row["controller_goal_before"], rtol=0, atol=1e-12)
        expected = np.r_[
            np.asarray(row["controller_goal_position_after"]) - pose[:3, 3],
            Rotation.from_matrix(np.asarray(row["controller_goal_after"]) @ pose[:3, :3].T).as_rotvec(),
        ]
        np.testing.assert_allclose(packet["action_goal_delta"][0], expected, rtol=0, atol=1e-8)
        wrench = row["current_wrench"]
        assert (
            wrench["readonly_verified"]
            and wrench["control_site"] == first["control_site"]
            and wrench["sensor_site"] == first["sensor_site"]
        )
        np.testing.assert_allclose(wrench["time"], state[0], rtol=0, atol=1e-10)
        spose, _ = sensor.endpoint(state)
        np.testing.assert_allclose(wrench["sensor_pose"], spose, rtol=0, atol=1e-9)
        np.testing.assert_allclose(wrench["control_site_position"], pose[:3, 3], rtol=0, atol=1e-9)
        local = np.asarray(wrench["local"])
        force = spose[:3, :3] @ local[:3]
        world = np.r_[force, spose[:3, :3] @ local[3:] + np.cross(spose[:3, 3] - pose[:3, 3], force)]
        np.testing.assert_allclose(wrench["world_at_control_site"], world, rtol=0, atol=1e-9)
        np.testing.assert_allclose(packet["x_current_wrench_world"][0], world, rtol=0, atol=1e-9)
