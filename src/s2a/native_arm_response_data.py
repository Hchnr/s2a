"""087 causal state/action and separate next-state labels from native records."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.xml_site_kinematics import XMLSiteKinematics


def reconstruct_native_arm_data(result, goals, data):
    contract = result["controller_contract"]
    assert contract["control_freq"] == 20 and contract["control_frame"] == "MuJoCo world"
    kin = XMLSiteKinematics(result["record"]["model_xml"], contract["controller_site"])
    commands = np.asarray(data["sent_commands"], dtype=float)
    n = len(commands)
    native_steps = result["native_steps"]
    assert commands.shape == (n, 7) and n == result["physics_steps"] == native_steps + 80
    assert len(goals) == n and data["sim_states"].shape == (n + 1, 1 + kin.nq + kin.nv)
    states = np.asarray(data["sim_states"])
    q = states[:, 1 : 1 + kin.nq]
    v = states[:, 1 + kin.nq :]
    np.testing.assert_allclose(np.diff(states[:, 0]), 0.05, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(states[:, 0], data["sim_time"])
    poses = []
    twists = []
    for state in states:
        pose, twist = kin.endpoint(state)
        poses.append(pose)
        twists.append(twist)
    poses, twists = np.array(poses), np.array(twists)
    np.testing.assert_allclose(poses, data["controller_site_pose"], rtol=0, atol=1e-9)
    by_name = {node.get("name"): (qi, vi, kind) for node, (qi, vi, kind) in kin.indices.items()}
    arm = [by_name[f"robot0_joint{i}"] for i in range(1, 8)]
    fingers = [by_name[f"gripper0_finger_joint{i}"] for i in (1, 2)]
    assert all(kind == "hinge" for _, _, kind in arm) and all(kind == "slide" for _, _, kind in fingers)
    arm_q = q[:, [x[0] for x in arm]]
    arm_v = v[:, [x[1] for x in arm]]
    finger_q = q[:, [x[0] for x in fingers]]
    finger_v = v[:, [x[1] for x in fingers]]
    np.testing.assert_allclose(finger_q, data["gripper_qpos"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.clip((finger_q[:, 0] - finger_q[:, 1] - 0.001) / 0.079, 0, 1), data["opening"], rtol=0, atol=1e-12
    )
    motors = []
    motor_sizes = []
    for i in range(n + 1):
        flat = data["gripper_current_action_flat"]
        offsets = data["gripper_current_action_offsets"]
        m = np.asarray(flat[offsets[i] : offsets[i + 1]])
        assert m.shape in ((1,), (2,)) and np.isfinite(m).all()
        motor_sizes.append(len(m))
        motors.append(np.repeat(m, 2) if len(m) == 1 else m.copy())
    goal_pos = np.asarray(data["controller_goal_pos"])
    goal_rot = np.asarray(data["controller_goal_ori"])
    for k, goal in enumerate(goals):
        assert goal["step_index"] == k and type(goal["overridden"]) is bool
        np.testing.assert_array_equal(goal["action"], commands[k, :6])
        np.testing.assert_allclose(goal["actual_goal_pos"], goal_pos[k + 1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(goal["actual_goal_ori"], goal_rot[k + 1], rtol=0, atol=1e-12)
        assert goal["overridden"] == (k >= native_steps)
        if k < native_steps:
            assert goal["set_pos"] is None and goal["set_ori"] is None
        else:
            np.testing.assert_allclose(goal["set_pos"], poses[native_steps, :3, 3], rtol=0, atol=1e-9)
            np.testing.assert_allclose(goal["set_ori"], poses[native_steps, :3, :3], rtol=0, atol=1e-9)
    goal_delta = np.c_[
        goal_pos[1:] - poses[:-1, :3, 3],
        Rotation.from_matrix(goal_rot[1:] @ poses[:-1, :3, :3].transpose(0, 2, 1)).as_rotvec(),
    ]
    next_rotation = Rotation.from_matrix(poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1)).as_rotvec()
    previous = np.zeros_like(commands)
    previous[1:] = commands[:-1]
    arrays = dict(
        step=np.arange(n, dtype=np.int64),
        time=states[:-1, 0].copy(),
        x_site_pose=poses[:-1],
        x_site_twist=twists[:-1],
        x_joint_position=arm_q[:-1],
        x_joint_velocity=arm_v[:-1],
        x_finger_qpos=finger_q[:-1],
        x_finger_qvel=finger_v[:-1],
        x_motor_before=np.array(motors[:-1]),
        x_previous_command=previous,
        x_previous_command_available=np.arange(n) > 0,
        x_previous_goal_pos=goal_pos[:-1],
        x_previous_goal_ori=goal_rot[:-1],
        action_command=commands.copy(),
        action_goal_delta=goal_delta,
        action_goal_pos=goal_pos[1:],
        action_goal_ori=goal_rot[1:],
        y_motion=poses[1:, :3, 3] - poses[:-1, :3, 3],
        y_rotation_increment=next_rotation,
        y_twist=twists[1:],
        y_joint_position=arm_q[1:],
        y_joint_velocity=arm_v[1:],
        protocol_native_goal_override=np.arange(n) >= native_steps,
        protocol_original_motor_size=np.array(motor_sizes[:-1], dtype=np.int64),
    )
    assert all(len(value) == n and np.isfinite(value).all() for value in arrays.values())
    return arrays, dict(
        rows=n,
        endpoints=n + 1,
        native_steps=native_steps,
        pose_max_abs_error=float(np.max(np.abs(poses - data["controller_site_pose"]))),
        velocity_source="analytic XML site Jacobian times current raw qvel",
        actual_goal_protocol="current action goal; native-end measured SITE latch for last80; not future observed state",
    )
