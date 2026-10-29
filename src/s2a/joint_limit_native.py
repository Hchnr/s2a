"""Causal native adapter for the separate ideal joint-limit refinement candidate."""

import numpy as np

from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_state import rotation_goal_from_command
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import finite, pose_error
from s2a.joint_limit_inverse import refine_command
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.osc import OSCInterface


class JointLimitResponseHistory(CoupledResponseHistory):
    """Separate completed innovation for the actually executed limited prediction."""

    def commit_prediction(self, pose, twist, prediction):
        self.current(pose, twist)
        self.pending = (np.asarray(pose).copy(), finite(prediction, (12,)).copy())


def refine_native(env, dynamics, site, twist, poses, anchor, index, seed, previous, contract, history):
    """Read robot-only current state and already generated references; never step."""
    controller = env.robots[0].controller
    if not controller.uncoupling or not controller.use_delta or controller.impedance_mode != "fixed":
        raise ValueError("Original fixed uncoupled delta OSC required")
    if contract["control_freq"] != 20 or not np.isclose(env.sim.model.opt.timestep, 0.002):
        raise ValueError("Fixed20Hz/2ms controller required")
    q = np.asarray(dynamics["robot_joint_position"])
    velocity = np.asarray(dynamics["robot_joint_velocity"])
    ranges = np.asarray([env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)])
    model = build_joint_response(
        dynamics["mass"],
        dynamics["jacobian"],
        dynamics["kp"],
        dynamics["kd"],
        q,
        controller.initial_joint,
        null_kp=10.0,
        dt=0.002,
    )
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    innovation = history.current(site, twist)
    target = np.r_[pose_error(poses[index], site), reference_world_twist(poses, index, anchor, 0.05)["twist"]]
    target -= np.r_[innovation[:6], innovation[6:] / 0.05]
    result = refine_command(model, velocity, seed, previous, interface, ranges[:, 0], ranges[:, 1], target)
    command = result["command"].copy()
    lower = np.maximum(interface.input_min, previous - 0.25)
    upper = np.minimum(interface.input_max, previous + 0.25)
    reset_adjusted = False
    if np.all(interface.decode(command)[3:] == 0):
        for axis in range(3, 6):
            for sign in (1, -1):
                trial = command.copy()
                trial[axis] += sign * 1e-10
                if lower[axis] <= trial[axis] <= upper[axis] and np.any(interface.decode(trial)[3:] != 0):
                    command, reset_adjusted = trial, True
                    break
            if reset_adjusted:
                break
        if not reset_adjusted:
            raise ValueError("No feasible reset representative")
    delta = interface.decode(command)
    nominal = predict_joint_response(
        model, velocity, delta, ranges[:, 0], ranges[:, 1], substeps=25, enforce_limits=True
    )
    prediction = np.r_[nominal["task_displacement"], nominal["task_velocity"]]
    goal, _ = rotation_goal_from_command(command, interface, site[:3, :3], controller.goal_ori)
    result.update(
        command=command,
        nominal_response=prediction,
        history_before=innovation,
        expected_goal_pos=site[:3, 3] + delta[:3],
        expected_goal_ori=goal,
        reset_representative_adjusted=reset_adjusted,
        model_scope="frozen joint OSC and ideal unilateral limits; no contact model",
    )
    return result
