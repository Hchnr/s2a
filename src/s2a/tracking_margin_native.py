"""Robot-only native adapter for the bounded four-period joint preview candidate."""

import time

import numpy as np

from s2a.current_coupled_state import rotation_goal_from_command
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import pose_error
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.joint_limit_preview_objective import preview_objective
from s2a.osc import OSCInterface
from s2a.tracking_margin_hybrid import refine_long_plan as refine_joint_plan


def refine_native_preview(
    env, dynamics, site, twist, poses, anchor, index, seed_plan, previous, contract, history, *, site_from_tcp
):
    started = time.perf_counter()
    controller = env.robots[0].controller
    if not controller.uncoupling or not controller.use_delta or controller.impedance_mode != "fixed":
        raise ValueError("Original uncoupled fixed delta OSC required")
    if contract["control_freq"] != 20 or not np.isclose(env.sim.model.opt.timestep, 0.002, atol=1e-12, rtol=0):
        raise ValueError("Original20Hz/2ms required")
    seed_plan = np.asarray(seed_plan, dtype=float)
    if seed_plan.ndim != 2 or seed_plan.shape[1] != 6 or not 1 <= len(seed_plan) <= min(4, len(poses) - index):
        raise ValueError("Complete original up-to4-period plan into available references required")
    horizon = min(8, len(poses) - index)
    seed_plan = np.vstack((seed_plan, np.tile(seed_plan[-1], (horizon - len(seed_plan), 1))))
    bias_for_audit = history.current(site, twist)
    q, velocity = dynamics["robot_joint_position"], dynamics["robot_joint_velocity"]
    limits = np.asarray([env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)])
    model = build_joint_preview(
        dynamics["mass"],
        dynamics["jacobian"],
        dynamics["kp"],
        dynamics["kd"],
        q,
        controller.initial_joint,
        null_kp=10.0,
        dt=0.002,
    )
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    deltas = np.asarray([interface.decode(c) for c in seed_plan])
    if np.all(deltas[0, 3:] == 0):
        raise ValueError("Original first command must have a checked reset representative")
    unsupported = bool(np.any(np.all(deltas[:, 3:] == 0, axis=1)))
    inverse, guard = None, None
    if unsupported:
        route = "unsupported_future_latch_seed"
    else:
        free = rollout_joint_preview(model, velocity, deltas, site, limits[:, 0], limits[:, 1], enforce_limits=False)
        risk = (
            np.any(q < limits[:, 0] + 0.05)
            or np.any(q > limits[:, 1] - 0.05)
            or np.any(free["inner_joint_position"] < limits[:, 0] + 0.05)
            or np.any(free["inner_joint_position"] > limits[:, 1] - 0.05)
        )
        route = "unchanged_free_feasible"
        if risk:
            horizon = len(seed_plan)
            tcp_poses, tcp_anchor = poses @ site_from_tcp, anchor @ site_from_tcp
            desired = tcp_poses[index : index + horizon]
            twists = np.asarray(
                [reference_world_twist(tcp_poses, j, tcp_anchor, 0.05)["twist"] for j in range(index, index + horizon)]
            )
            inverse = refine_joint_plan(
                model,
                velocity,
                seed_plan,
                previous,
                interface,
                site,
                site_from_tcp,
                desired,
                twists,
                limits[:, 0],
                limits[:, 1],
                initial_guess=history.shifted(horizon),
            )
            seed_rollout = rollout_joint_preview(
                model,
                velocity,
                deltas,
                site,
                limits[:, 0],
                limits[:, 1],
                enforce_limits=inverse["seed_guard_model"] != "free",
            )
            seed_score = preview_objective(seed_rollout, site_from_tcp, desired, twists)
            chosen_score = preview_objective(inverse["rollout"], site_from_tcp, desired, twists)
            divisor = np.array([100.0] * 3 + [20.0] * 3)
            guard = dict(
                seed_tcp_errors=seed_score["residual"].reshape(horizon, 12)[:, :6] / divisor,
                chosen_tcp_errors=chosen_score["residual"].reshape(horizon, 12)[:, :6] / divisor,
            )
            route = inverse["selection"] if inverse["accepted"] else inverse["selection"] + "_retained_seed"
    plan = seed_plan.copy() if inverse is None else inverse["planned_commands"].copy()
    command = plan[0].copy()
    history.stage_plan(plan)
    # Only this actual first command is committed. Unsupported future latch
    # periods never become an unannounced one-period optimization candidate.
    nominal = rollout_joint_preview(
        model, velocity, np.asarray([interface.decode(command)]), site, limits[:, 0], limits[:, 1], enforce_limits=True
    )
    if inverse is not None:
        np.testing.assert_allclose(nominal["pose"][0], inverse["rollout"]["pose"][0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(nominal["twist"][0], inverse["rollout"]["twist"][0], atol=1e-10, rtol=0)
    prediction = np.r_[pose_error(nominal["pose"][0], site), nominal["twist"][0]]
    goal, _ = rotation_goal_from_command(command, interface, site[:3, :3], controller.goal_ori)
    optimization = (
        None
        if inverse is None
        else {k: v for k, v in inverse.items() if k not in ("rollout", "seed_rollout", "planned_commands", "command")}
    )
    return dict(
        command=command,
        planned_commands=plan,
        nominal_response=prediction,
        route=route,
        evaluations=0 if inverse is None else inverse["response_evaluations"],
        plan_guard=guard,
        guard_seed_model="ideal_limit" if inverse is None else inverse["seed_guard_model"],
        optimization=optimization,
        history_before=bias_for_audit,
        innovation_applied=False,
        expected_goal_pos=site[:3, 3] + interface.decode(command)[:3],
        expected_goal_ori=goal,
        compute_seconds=time.perf_counter() - started,
        scope="current robot state, prior seed/history and generated references only; ideal hard-limit model",
    )
