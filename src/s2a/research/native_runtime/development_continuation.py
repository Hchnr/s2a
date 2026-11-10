"""Run or replay a short continuation without supplying future observations to actions."""

import numpy as np

from s2a.geometry import pose_error


def current_joint_site_state(env):
    """Read refreshed robot q, limits and site Jacobian without advancing or forwarding."""
    import mujoco

    robot, sim = env.robots[0], env.sim
    model, data = sim.model._model, sim.data._data
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
    if min(joint_ids) < 0:
        raise ValueError("Seven named robot joints required")
    limits = np.asarray([model.jnt_range[i] for i in joint_ids]).copy()
    q = np.asarray(sim.data.qpos)[robot._ref_joint_pos_indexes].copy()
    dofs = robot._ref_joint_vel_indexes
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.controller.eef_name)
    if sid < 0:
        raise ValueError("Fixed controller site required")
    before = np.asarray(sim.get_state().flatten()).copy()
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    np.testing.assert_array_equal(before, sim.get_state().flatten())
    return q, limits, np.vstack((jp, jr))[:, dofs].copy()


def current_robot_dynamics(env, jacobian):
    """Read current robot mass, qvel and site without forwarding or stepping."""
    import mujoco

    robot, sim = env.robots[0], env.sim
    model, data = sim.model._model, sim.data._data
    dofs = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
    full_mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full_mass, data.qM)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.controller.eef_name)
    if sid < 0 or dofs.shape != (7,) or np.asarray(jacobian).shape != (6, 7):
        raise ValueError("Fixed site, seven robot dofs and 6x7 Jacobian required")
    site = np.eye(4)
    site[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    site[:3, 3] = data.site_xpos[sid]
    velocity = np.asarray(sim.data.qvel)[dofs].copy()
    return full_mass[np.ix_(dofs, dofs)], velocity, site


def run_continuation(
    env,
    terminal,
    observers,
    receipt,
    *,
    start_step,
    previous_command,
    targets,
    tool,
    target_object,
    choose_action,
    stop_on_outside,
    expected_periods=None,
    historical_states=None,
    state_reader=None,
    task_observer=None,
    on_completed=None,
    progress=None,
    continuation_limit=8,
):
    """Dispatch a registered eight- or sixteen-step fixed-reference continuation."""
    targets, tool = np.asarray(targets, dtype=float), np.asarray(tool, dtype=float)
    previous = np.asarray(previous_command, dtype=float).copy()
    if type(continuation_limit) is not int or continuation_limit not in (8, 16):
        raise ValueError("An explicit eight- or sixteen-step continuation is required")
    if targets.shape != (continuation_limit, 4, 4) or tool.shape != (4, 4) or previous.shape != (7,):
        raise ValueError("Registered already-generated targets, tool and previous command required")
    if not np.isfinite(targets).all() or not np.isfinite(tool).all() or not np.isfinite(previous).all():
        raise ValueError("Finite continuation inputs required")
    if expected_periods is not None and not 0 <= len(expected_periods) <= continuation_limit:
        raise ValueError("A bounded recorded continuation is required")
    if historical_states is not None and np.asarray(historical_states).shape != (continuation_limit, 79):
        raise ValueError("Exactly the registered historical comparison states are required")
    if task_observer is None:
        from s2a.research.wam.tracking.runtime.checkpoint_simulation import object_progress

        task_observer = object_progress
    if state_reader is None:
        state_reader = current_joint_site_state

    robot, sim = env.robots[0], env.sim
    original_step = sim.step
    periods = []
    partial = {"periods": periods, "stop": None}
    receipt["continuation_partial"] = partial
    count = continuation_limit if expected_periods is None else len(expected_periods)
    if terminal["endpoint_receipt"]["endpoint_step"] != start_step:
        raise ValueError("Continuation begins at the verified prefix endpoint")
    try:
        for offset in range(count):
            step = start_step + offset
            q, limits, jacobian = state_reader(env)
            selected = choose_action(q.copy(), limits.copy(), jacobian.copy(), previous.copy(), offset)
            if not isinstance(selected, dict):
                raise ValueError("Action selection must return a decision")
            if "stop" in selected:
                stop = selected["stop"]
                if (
                    not isinstance(stop, dict)
                    or not isinstance(stop.get("reason"), str)
                    or not stop["reason"]
                    or "command" in selected
                    or selected.get("newly_selected", False) is not False
                    or expected_periods is not None
                ):
                    raise ValueError("A pre-dispatch stop must contain a reason and no selected command")
                partial["stop"] = {**stop, "at_step": step, "next_action_dispatched": False}
                if progress is not None:
                    progress(receipt)
                break
            newly_selected = selected.get("newly_selected", False)
            if not isinstance(newly_selected, bool):
                raise ValueError("Explicit new-command accounting must be boolean")
            if newly_selected:
                receipt["newly_selected_commands"] += 1
            command = np.asarray(selected["command"], dtype=float)
            if command.shape != (7,) or not np.isfinite(command).all() or np.max(abs(command)) > 1 + 1e-12:
                raise ValueError("Finite controller-box command required")
            if np.max(abs(command[:6] - previous[:6])) > 0.25 + 1e-12:
                raise ValueError("Original arm command rate exceeded")
            if step % 8 == 0:
                observers.render_cameras(env, 256)
            inner = []

            def observed_step(*args, **kwargs):
                value = original_step(*args, **kwargs)
                inner.append(
                    {
                        "time": float(sim.data.time),
                        "q": np.asarray(sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                        "qvel": np.asarray(sim.data.qvel)[robot._ref_joint_vel_indexes].copy(),
                    }
                )
                return value

            sim.step = observed_step
            try:
                receipt["control_steps_attempted"] += 1
                env.step(command.copy())
                receipt["control_steps_completed"] += 1
            finally:
                sim.step = original_step
            if len(inner) != 25:
                raise ValueError("Exactly 25 native inner steps required")
            qinner = np.asarray([x["q"] for x in inner])
            margins = np.minimum(qinner - limits[:, 0], limits[:, 1] - qinner)
            np.testing.assert_allclose(np.diff([x["time"] for x in inner]), 0.002, atol=1e-12, rtol=0)
            observers.current(env)
            sample, endpoint_receipt = observers.endpoint(env, step + 1)
            error = pose_error(targets[offset], np.asarray(sample["source_pose"]) @ tool)
            objects = task_observer(env)
            period = {
                "step": step,
                "command": command.copy(),
                "decision": {key: value for key, value in selected.items() if key != "command"},
                "actual_state": np.asarray(sample["sim_states"]).copy(),
                "actual_inner": inner,
                "minimum_inner_margin_rad": float(margins.min()),
                "inner_outside_samples": int(np.sum(np.any(margins < 0, axis=1))),
                "inner_below_buffer_samples": int(np.sum(np.any(margins < 0.05, axis=1))),
                "tcp_error": error,
                "pose_caps_satisfied": bool(np.linalg.norm(error[:3]) <= 0.01 and np.linalg.norm(error[3:]) <= 0.05),
                "task_success": bool(objects["success"]),
                "target_grasp": bool(objects["objects"][target_object]["grasp_contact"]),
                "endpoint_receipt": endpoint_receipt,
            }
            if historical_states is not None:
                np.testing.assert_allclose(period["actual_state"], historical_states[offset], atol=1e-10, rtol=0)
                period["historical_state_exact"] = bool(
                    np.array_equal(period["actual_state"], historical_states[offset])
                )
            if expected_periods is not None:
                expected = expected_periods[offset]
                np.testing.assert_array_equal(period["command"], expected["command"])
                np.testing.assert_allclose(period["actual_state"], expected["actual_state"], atol=1e-10, rtol=0)
                for key in ("q", "qvel", "time"):
                    np.testing.assert_allclose(
                        [item[key] for item in inner],
                        [item[key] for item in expected["actual_inner"]],
                        atol=1e-10,
                        rtol=0,
                    )
                np.testing.assert_allclose(period["tcp_error"], expected["tcp_error"], atol=1e-10, rtol=0)
                for key in (
                    "inner_outside_samples",
                    "inner_below_buffer_samples",
                    "pose_caps_satisfied",
                    "task_success",
                    "target_grasp",
                ):
                    if period[key] != expected[key]:
                        raise ValueError(f"Independent continuation label differs: {key}")
            periods.append(period)
            if on_completed is not None:
                period["completed_feedback"] = on_completed(env, command.copy(), offset)
            previous = command.copy()
            if progress is not None:
                progress(receipt)
            if stop_on_outside and period["inner_outside_samples"]:
                partial["stop"] = {
                    "reason": "actual_inner_joint_limit_violation",
                    "after_step": step,
                    "next_action_dispatched": False,
                }
                break
    finally:
        sim.step = original_step
    if expected_periods is not None and len(periods) != len(expected_periods):
        raise ValueError("Replay stopped before all recorded actions")
    partial["steps_completed"] = len(periods)
    return partial
