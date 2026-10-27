"""228 read robot-only current inertia/Jacobian and native rotation goal, no forwarding."""

import numpy as np

from s2a.geometry import finite
from s2a.osc import OSCInterface


def rotation_goal_from_command(command, interface, current_rotation, previous_goal, *, matrix_decoder=None):
    command = finite(command, (6,))
    current, previous = finite(current_rotation, (3, 3)), finite(previous_goal, (3, 3))
    if np.any(command < interface.input_min) or np.any(command > interface.input_max):
        raise ValueError("Feasible actual normalized arm command required")
    delta = interface.decode(command)[3:]
    if np.all(delta == 0):
        goal = previous.copy()
    else:
        if matrix_decoder is None:
            from robosuite.utils import transform_utils as transform

            def matrix_decoder(value):
                return transform.quat2mat(transform.axisangle2quat(value))

        goal = finite(matrix_decoder(delta), (3, 3)) @ current
    return goal, goal @ current.T


def read_current_dynamics(env, current_data, command):
    """Require the already refreshed endpoint used by current_model_state.

    Only robot joint blocks of M/J are used. No object pose, contact, goal update,
    simulation forward/step or controller cache refresh is performed here.
    """
    import mujoco

    robot, sim = env.robots[0], env.sim
    controller = robot.controller
    if not bool(controller.uncoupling) or not bool(controller.use_delta) or controller.impedance_mode != "fixed":
        raise ValueError("Current uncoupled fixed-delta OSC instance required")
    if controller.position_limits is not None or controller.orientation_limits is not None:
        raise ValueError("Unmodelled controller goal limits")
    if env.env.control_freq != 20 or not np.isclose(sim.model.opt.timestep, 0.002, atol=1e-12, rtol=0):
        raise ValueError("Actual20Hz/2ms fixed controller required")
    command = finite(command, (7,))
    if abs(command[-1]) > 1:
        raise ValueError("Feasible current gripper candidate required")
    before = sim.get_state().flatten().copy()
    controls, sensors = np.asarray(sim.data.ctrl).copy(), np.asarray(sim.data.sensordata).copy()
    q = finite(np.asarray(current_data["x_joint_position"])[0], (7,))
    qd = finite(np.asarray(current_data["x_joint_velocity"])[0], (7,))
    pose = np.asarray(current_data["x_site_pose"], dtype=float)
    twist = np.asarray(current_data["x_site_twist"], dtype=float)
    if pose.shape != (1, 4, 4) or twist.shape != (1, 6):
        raise ValueError("One refreshed current world SITE endpoint required")
    qids, dofs = robot._ref_joint_pos_indexes, robot._ref_joint_vel_indexes
    np.testing.assert_allclose(q, np.asarray(sim.data.qpos)[qids], atol=1e-12, rtol=0)
    np.testing.assert_allclose(qd, np.asarray(sim.data.qvel)[dofs], atol=1e-12, rtol=0)
    model, data = sim.model._model, sim.data._data
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
    if sid < 0:
        raise ValueError("Actual control SITE missing")
    np.testing.assert_allclose(data.site_xpos[sid], pose[0, :3, 3], atol=1e-9, rtol=0)
    np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), pose[0, :3, :3], atol=1e-9, rtol=0)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    full_j = np.vstack([jp, jr])
    np.testing.assert_allclose(full_j @ data.qvel, twist[0], atol=1e-8, rtol=0)
    previous_goal = np.asarray(current_data["x_previous_goal_ori"])[0]
    np.testing.assert_allclose(previous_goal, controller.goal_ori, atol=1e-12, rtol=0)
    interface = OSCInterface(
        *(getattr(controller, key) for key in ("input_min", "input_max", "output_min", "output_max"))
    )
    goal, relative = rotation_goal_from_command(command[:6], interface, pose[0, :3, :3], previous_goal)
    np.testing.assert_array_equal(before, sim.get_state().flatten())
    np.testing.assert_array_equal(controls, np.asarray(sim.data.ctrl))
    np.testing.assert_array_equal(sensors, np.asarray(sim.data.sensordata))
    return dict(
        mass=mass[np.ix_(dofs, dofs)].copy(),
        jacobian=full_j[:, dofs].copy(),
        robot_joint_position=q.copy(),
        robot_joint_velocity=qd.copy(),
        kp=np.asarray(controller.kp).copy(),
        kd=np.asarray(controller.kd).copy(),
        actual_rotation_goal=goal,
        relative_rotation_goal=relative,
        uncoupling=bool(controller.uncoupling),
        use_delta=bool(controller.use_delta),
        impedance_mode=controller.impedance_mode,
        control_freq=20,
        model_timestep=0.002,
    )
