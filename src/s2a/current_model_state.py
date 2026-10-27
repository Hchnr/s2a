"""Read a refreshed fixed-OSC endpoint without forwarding or executing action."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.current_wrench import read_current_wrench


def current_model_state(env, command, previous_command, *, previous_available, history):
    robot, sim = env.robots[0], env.sim
    controller = robot.controller
    if not controller.use_delta or controller.impedance_mode != "fixed":
        raise ValueError("Fixed delta OSC required")
    if controller.position_limits is not None or controller.orientation_limits is not None:
        raise ValueError("Unmodelled controller goal limits")
    command, previous = np.asarray(command, dtype=float), np.asarray(previous_command, dtype=float)
    if (
        command.shape != (7,)
        or previous.shape != (7,)
        or not np.isfinite(command).all()
        or not np.isfinite(previous).all()
        or type(previous_available) is not bool
    ):
        raise ValueError("Finite candidate/previous seven commands and explicit availability required")
    lo, hi, outlo, outhi = (
        np.asarray(getattr(controller, k), dtype=float) for k in ("input_min", "input_max", "output_min", "output_max")
    )
    if any(v.shape != (6,) for v in (lo, hi, outlo, outhi)) or np.any(lo >= hi) or np.any(outlo >= outhi):
        raise ValueError("Ordered six-axis OSC interface required")
    if np.any(command[:6] < lo) or np.any(command[:6] > hi) or abs(command[6]) > 1 or np.any(np.abs(previous) > 1):
        raise ValueError("Actual normalized command outside fixed interface")
    before = sim.get_state().flatten().copy()
    controls = np.asarray(sim.data.ctrl).copy()
    sensors = np.asarray(sim.data.sensordata).copy()
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(sim.data.get_site_xmat(controller.eef_name))
    pose[:3, 3] = np.asarray(sim.data.get_site_xpos(controller.eef_name))
    twist = np.r_[sim.data.get_site_xvelp(controller.eef_name), sim.data.get_site_xvelr(controller.eef_name)]
    motor = np.asarray(robot.gripper.current_action, dtype=float)
    if motor.shape == (1,):
        motor = np.repeat(motor, 2)
    if motor.shape != (2,):
        raise ValueError("Known native motor cache size required")
    wrench = read_current_wrench(env)
    goal_delta = outlo + (command[:6] - lo) * (outhi - outlo) / (hi - lo)
    if np.all(goal_delta[3:] == 0):
        goal_delta[3:] = Rotation.from_matrix(np.asarray(controller.goal_ori) @ pose[:3, :3].T).as_rotvec()
    row = dict(
        x_site_pose=pose[None],
        x_site_twist=twist[None],
        x_joint_position=np.asarray(sim.data.qpos)[robot._ref_joint_pos_indexes][None].copy(),
        x_joint_velocity=np.asarray(sim.data.qvel)[robot._ref_joint_vel_indexes][None].copy(),
        x_finger_qpos=np.asarray(sim.data.qpos)[robot._ref_gripper_joint_pos_indexes][None].copy(),
        x_finger_qvel=np.asarray(sim.data.qvel)[robot._ref_gripper_joint_vel_indexes][None].copy(),
        x_motor_before=motor[None].copy(),
        x_previous_command=previous[None].copy(),
        x_previous_command_available=np.array([previous_available]),
        x_previous_goal_pos=np.asarray(controller.goal_pos)[None].copy(),
        x_previous_goal_ori=np.asarray(controller.goal_ori)[None].copy(),
        action_command=command[None].copy(),
        action_goal_delta=goal_delta[None],
        x_current_wrench_world=np.asarray(wrench["world_at_control_site"])[None].copy(),
    )
    row = history.attach(row)
    np.testing.assert_array_equal(before, sim.get_state().flatten())
    np.testing.assert_array_equal(controls, sim.data.ctrl)
    np.testing.assert_array_equal(sensors, sim.data.sensordata)
    return row, wrench
