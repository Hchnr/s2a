"""232 independent raw-state dynamics, native goal, ODE/KKT and dual-history replay."""

import numpy as np
from arm_joint_wrench_independent import nominal_prediction
from current_coupled_independent import response, verify_inverse
from scipy.spatial.transform import Rotation

from s2a.archived_forward_state import verify_forward_state


def close(actual, expected, atol=1e-8):
    actual, expected = (np.asarray(value, dtype=float) for value in (actual, expected))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def innovation(previous, pose, twist, *, scaled_velocity):
    if previous is None:
        return np.zeros(12)
    origin, prediction = previous
    rotation = (
        Rotation.from_matrix(pose[:3, :3] @ origin[:3, :3].T) * Rotation.from_rotvec(prediction[3:6]).inv()
    ).as_rotvec()
    velocity = 0.05 * twist - (prediction[6:] if scaled_velocity else 0.05 * prediction[6:])
    return np.r_[pose[:3, 3] - origin[:3, 3] - prediction[:3], rotation, velocity]


class CurrentCoupledArmLogReplay:
    def __init__(self, parameters, contract, source_from_site, xml_path, arrays):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.contract, self.parameters = contract, parameters
        self.mapping = np.asarray(source_from_site)
        self.arrays = arrays
        assert arrays["sim_states"].shape[1] == 1 + self.model.nq + self.model.nv
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        assert self.sid >= 0
        self.qids, self.dofs = [], []
        for joint in range(1, 8):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{joint}")
            assert jid >= 0
            self.qids.append(self.model.jnt_qposadr[jid])
            self.dofs.append(self.model.jnt_dofadr[jid])
        self.quaternions = []
        for joint in range(self.model.njnt):
            kind = int(self.model.jnt_type[joint])
            if kind in (0, 1):
                self.quaternions.append(
                    dict(
                        joint=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                        qpos_start=int(self.model.jnt_qposadr[joint]) + (3 if kind == 0 else 0),
                    )
                )
        self.previous = np.zeros(7)
        self.last_old, self.last_coupled = None, None
        self.steps, self.maximum_kkt = 0, 0.0
        self.normalization_checks = []

    def endpoint(self, index):
        m, d = self.model, self.data
        state = self.arrays["sim_states"][index]
        d.time = state[0]
        d.qpos[:] = state[1 : 1 + m.nq]
        d.qvel[:] = state[1 + m.nq :]
        self.mujoco.mj_forward(m, d)
        changes = verify_forward_state(
            state, np.r_[d.time, d.qpos, d.qvel], nq=m.nq, quaternion_segments=self.quaternions
        )
        if changes:
            self.normalization_checks.append(dict(endpoint=index, changes=changes))
        pose = np.eye(4)
        pose[:3, 3] = d.site_xpos[self.sid]
        pose[:3, :3] = d.site_xmat[self.sid].reshape(3, 3)
        close(pose, self.arrays["controller_site_pose"][index], 1e-9)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.mujoco.mj_jacSite(m, d, jp, jr, self.sid)
        full_j = np.vstack([jp, jr])
        mass = np.zeros((m.nv, m.nv))
        self.mujoco.mj_fullM(m, mass, d.qM)
        return (
            pose,
            full_j @ d.qvel,
            mass[np.ix_(self.dofs, self.dofs)].copy(),
            full_j[:, self.dofs].copy(),
            d.qpos[self.qids].copy(),
            d.qvel[self.dofs].copy(),
        )

    def verify(self, row, arm, desired, site_velocity):
        from robosuite.utils import transform_utils as transform

        pose, twist, mass, jacobian, q, qd = self.endpoint(self.steps)
        packet = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
        close(packet["x_site_pose"][0], pose, 1e-9)
        close(pose, np.asarray(row["measured_source"]) @ self.mapping, 1e-8)
        close(packet["x_site_twist"][0], twist, 1e-8)
        close(twist, row["measured_twist"], 1e-8)
        close(packet["x_joint_position"][0], q, 1e-12)
        close(packet["x_joint_velocity"][0], qd, 1e-12)
        close(packet["x_previous_command"][0], self.previous, 1e-12)
        assert bool(packet["x_previous_command_available"][0]) == bool(self.steps)
        command = np.asarray(row["command"])
        close(packet["action_command"][0], command, 1e-12)
        close(command[3:6], arm["command"][3:6], 1e-9)
        target = np.asarray(desired) @ self.mapping
        close(row["desired_site"], target, 1e-9)
        for key, value in site_velocity.items():
            if key == "velocity":
                close(row["site_velocity_reference"][key], value, 1e-9)
            else:
                assert row["site_velocity_reference"][key] == value
        previous_goal = self.arrays["controller_goal_ori"][self.steps]
        close(previous_goal, packet["x_previous_goal_ori"][0], 1e-12)
        close(previous_goal, row["controller_goal_before"], 1e-12)
        lo, hi, outlo, outhi = (
            np.asarray(self.contract[key]) for key in ("input_min", "input_max", "output_min", "output_max")
        )
        delta = outlo + (command[:6] - lo) * (outhi - outlo) / (hi - lo)
        goal = (
            previous_goal
            if np.all(delta[3:] == 0)
            else transform.quat2mat(transform.axisangle2quat(delta[3:])) @ pose[:3, :3]
        )
        close(goal, self.arrays["controller_goal_ori"][self.steps + 1], 1e-8)
        close(goal, row["controller_goal_after"], 1e-8)
        relative = goal @ pose[:3, :3].T
        close(packet["action_goal_delta"][0, 3:], Rotation.from_matrix(relative).as_rotvec(), 1e-8)
        dynamics = row["coupled_dynamics"]
        assert (
            dynamics["uncoupling"] is True and dynamics["use_delta"] is True and dynamics["impedance_mode"] == "fixed"
        )
        assert dynamics["control_freq"] == 20 and dynamics["model_timestep"] == 0.002
        for key, value in dict(
            mass=mass,
            jacobian=jacobian,
            robot_joint_position=q,
            robot_joint_velocity=qd,
            kp=self.contract["kp"],
            kd=self.contract["kd"],
            actual_rotation_goal=goal,
            relative_rotation_goal=relative,
        ).items():
            close(dynamics[key], value, 1e-9 if key not in ("actual_rotation_goal", "relative_rotation_goal") else 1e-8)
        independent = response(mass, jacobian, self.contract["kp"], self.contract["kd"], relative)
        produced = row["coupled_response_model"]
        assert produced["dt"] == 0.05
        for key, value in independent.items():
            close(produced[key], value)
        for key, value in dict(
            mass=mass,
            jacobian=jacobian,
            kp=self.contract["kp"],
            kd=self.contract["kd"],
            relative_rotation_goal=relative,
        ).items():
            close(produced[key], value)
        old_innovation = innovation(self.last_old, pose, twist, scaled_velocity=True)
        close(packet["x_response_residual"][0], old_innovation, 1e-9)
        close(row["response_residual"]["innovation"], old_innovation, 1e-9)
        assert row["response_residual"]["completed_periods"] == self.steps
        coupled_innovation = innovation(self.last_coupled, pose, twist, scaled_velocity=False)
        history = row["coupled_response_history"]
        close(history["innovation_before"], coupled_innovation)
        assert history["completed_periods_before"] == self.steps
        bias = np.vstack([coupled_innovation[:3], coupled_innovation[6:9] / 0.05])
        solved = row["coupled_inverse"]
        certificate = verify_inverse(
            solved,
            pose[:3, 3],
            twist,
            target[:3, 3],
            site_velocity["velocity"],
            self.previous[:3],
            self.contract,
            independent,
            bias,
        )
        self.maximum_kkt = max(self.maximum_kkt, certificate["max_stationarity_residual"])
        close(command[:3], solved["command"], 1e-12)
        close(delta[:3], solved["physical_delta"], 1e-12)
        close(delta[:3], packet["action_goal_delta"][0, :3], 1e-12)
        nominal = (
            independent["initial_velocity_response"] @ twist
            + independent["goal_forcing_response"] @ np.r_[delta[:3], independent["rotation_forcing"]]
        )
        commit = history["commit"]
        close(commit["executed_translation"], command[:3], 1e-12)
        close(commit["physical_delta"], delta[:3], 1e-12)
        close(commit["nominal_response"], nominal)
        close(row["predicted_linear_velocity"], np.asarray(solved["predicted_response"])[6:9])
        assert row["prediction_kind"] == "228 current coupled local response;completed innovation;not measured future"
        assert row["nominal_model_kind"] == "150 original uncoupled nominal;coupled nominal recorded separately"
        old_nominal = nominal_prediction(packet, self.parameters)[0]
        close(row["nominal_predicted_motion"], old_nominal[:6], 1e-9)
        close(row["nominal_predicted_linear_velocity"], old_nominal[6:9] / 0.05, 1e-9)
        after_pose, after_twist, *_ = self.endpoint(self.steps + 1)
        close(after_pose, np.asarray(row["after_source"]) @ self.mapping, 1e-8)
        expected_after = innovation((pose, nominal), after_pose, after_twist, scaled_velocity=False)
        close(history["innovation_after"], expected_after)
        assert history["completed_periods_after"] == self.steps + 1
        assert row["rate_limited"] == bool(
            np.any(np.isclose(np.abs(command[:3] - self.previous[:3]), 0.25, atol=1e-10))
        )
        self.last_old, self.last_coupled = (pose.copy(), old_nominal.copy()), (pose.copy(), nominal.copy())
        self.previous = command.copy()
        self.steps += 1
