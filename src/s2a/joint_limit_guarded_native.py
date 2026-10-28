"""Native refinement with exact same-state TCP position and SO(3) safeguards."""

import numpy as np

from s2a.current_coupled_state import rotation_goal_from_command
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import pose_error
from s2a.joint_limit_native import refine_native
from s2a.joint_limit_pose_guard import guard_command, predicted_tcp
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.osc import OSCInterface


def refine_native_guarded(
    env, dynamics, site, twist, poses, anchor, index, seed, previous, contract, history, *, site_from_tcp
):
    result = refine_native(env, dynamics, site, twist, poses, anchor, index, seed, previous, contract, history)
    if result["route"] == "unchanged_free_feasible":
        result["pose_guard"] = None
        return result
    controller = env.robots[0].controller
    q, velocity = dynamics["robot_joint_position"], dynamics["robot_joint_velocity"]
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
    bias = result["history_before"]
    target = np.r_[pose_error(poses[index], site), reference_world_twist(poses, index, anchor, 0.05)["twist"]]
    target -= np.r_[bias[:6], bias[6:] / 0.05]
    weights = np.array([100.0] * 3 + [20.0] * 3 + [5.0] * 3 + [1.0] * 3)

    def response_fn(command):
        prediction = predict_joint_response(
            model, velocity, interface.decode(command), ranges[:, 0], ranges[:, 1], substeps=25, enforce_limits=True
        )
        return np.r_[prediction["task_displacement"], prediction["task_velocity"]]

    def score_fn(response):
        residual = weights * (response - target)
        return float(residual @ residual)

    lower, upper = np.maximum(interface.input_min, previous - 0.25), np.minimum(interface.input_max, previous + 0.25)
    desired_tcp = poses[index] @ site_from_tcp
    guarded = guard_command(
        seed, result["command"], lower, upper, response_fn, score_fn, site, site_from_tcp, desired_tcp, bias
    )
    command, nominal = guarded["command"].copy(), guarded["nominal_response"].copy()
    reset_fallback = bool(np.all(interface.decode(command)[3:] == 0))
    if reset_fallback:
        # Preserve the original seed's checked reset semantics instead of moving
        # a tight active pose constraint after optimization.
        command = np.asarray(seed).copy()
        nominal = response_fn(command)
        guarded.update(
            command=command,
            nominal_response=nominal,
            chosen_tcp_error=pose_error(desired_tcp, predicted_tcp(site, site_from_tcp, nominal, bias)),
            candidate_objective=score_fn(nominal),
            route="pose_guard_retained_seed",
        )
    if not np.any(interface.decode(command)[3:] != 0):
        raise ValueError("Guarded seed lacks native reset representative")
    goal, _ = rotation_goal_from_command(command, interface, site[:3, :3], controller.goal_ori)
    result.update(
        unguarded_route=result["route"],
        route=guarded["route"],
        pose_guard=guarded,
        command=command,
        nominal_response=nominal,
        expected_goal_pos=site[:3, 3] + interface.decode(command)[:3],
        expected_goal_ori=goal,
        native_reset_seed_fallback=reset_fallback,
        candidate_objective=guarded["candidate_objective"],
    )
    return result
