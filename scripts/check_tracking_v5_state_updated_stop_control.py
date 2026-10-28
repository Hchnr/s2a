"""Independently reconstruct causal history and every stopped-state search response."""

import os
from pathlib import Path

import numpy as np
from check_tracking_v5_state_updated_response import native_functions
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/tracking_v5_state_updated_stop_control_check_20261028"
PRODUCER = REPO / "outputs/tracking_v5_state_updated_stop_control_20261028/verified.json"
NUMERIC_TOLERANCE = 1e-10
TORQUE_TOLERANCE = 1e-8


def main():
    strict_native_environment()
    require(not ROOT.exists(), "No overwrite or automatic retry")
    references = {}

    def remember(ref):
        checked(ref)
        require(ref["path"] not in references or references[ref["path"]] == ref, "Unique source fingerprint")
        references[ref["path"]] = ref
        return ref

    def load(ref):
        return read(remember(ref))

    def closed_owner(ref, expected_result=None):
        owner = load(ref)
        require(owner["complete"] is owner["passed"] is True, "Closed prerequisite owner")
        exited, released = load(owner["owned_command"]["exit"]), load(owner["owned_command"]["release"])
        require(
            exited["actual_wait"] is True
            and exited["returncode"] == 0
            and released["reaped"] is True
            and released["returncode"] == 0,
            "Prerequisite actual wait0/reap0",
        )
        if expected_result is not None:
            require(owner["result_ref"] == expected_result, "Owner result is the exact registered input")
        value = load(owner["result_ref"])
        require(value["complete"] is value["passed"] is True, "Prerequisite result passed")
        return owner, value

    producer_ref = remember(reference(PRODUCER))
    producer_owner_ref = remember(
        reference(REPO / "outputs/tracking_v5_state_updated_stop_control_execution_20261028/result.json")
    )
    _, producer = closed_owner(producer_owner_ref, producer_ref)
    require(
        producer["physics_steps"] == producer["world_model_calls"] == producer["fresh_cases"] == 0
        and producer["control_qualified"] is False,
        "Numerical search only",
    )
    reg = load(producer["registration_ref"])
    require(
        reg["seed"] == 42
        and reg["history_steps"] == list(range(368, 377))
        and reg["endpoint"] == 377
        and reg["cursor"] == 9
        and reg["automatic_retry"] is False,
        "Fixed seed/history/stopped endpoint",
    )
    require(
        reg["search"]["total_iterations"] == 60
        and reg["search"]["response_budget"] == 600
        and reg["search"]["difference_step"] == 1e-4
        and reg["search"]["optimization_safety_buffer"] == 1e-8
        and reg["search"]["ftol"] == 1e-12
        and reg["search"]["starts"] == 1,
        "Fixed search contract",
    )
    for name, expected in dict(
        rate=0.25, joint_margin=0.05, braking_time=0.1, position_cap=0.01, rotation_cap=0.05
    ).items():
        require(reg["constraints"][name] == expected, "Original constraints")
    for ref in reg["source_refs"] + reg["native_source_refs"]:
        remember(ref)
    _, prior_audit = closed_owner(reg["independent_response_owner_ref"], reg["prior_response_audit_ref"])
    require(load(prior_audit["registration_ref"])["producer_ref"] == reg["prior_response_ref"], "Exact prior response")
    prior = load(reg["prior_response_ref"])
    prior_reg = load(prior["registration_ref"])
    native = load(reg["native_ref"])
    nreg = load(native["registration_ref"])
    require(
        native["complete"] is native["passed"] is True
        and native["continuation_steps"] == 9
        and native["stopped"]["step"] == 377
        and native["stopped"]["action_dispatched"] is False,
        "Original actual branch stopped before cursor9",
    )
    _, native_audit = closed_owner(prior_reg["independent_owner_ref"], prior_reg["independent_ref"])
    require(
        load(native_audit["registration_ref"])["producer_ref"] == reg["native_ref"], "Independent same-branch replay"
    )
    for name in ("native_ref", "states_ref", "periods_ref", "xml_ref", "control_ref", "tool_ref", "initial_ref"):
        require(reg[name] == prior_reg[name], "Unchanged original response inputs")
    require(reg["states_ref"] == native["states_ref"] and reg["periods_ref"] == native["periods_ref"], "Actual labels")
    require(reg["chunks_ref"] == nreg["chunks_ref"], "Same already generated reference")
    original_stop = load(reg["original_stop_ref"])
    require(original_stop["complete"] is original_stop["passed"] is True, "Original stopped-state audit")
    stop_reg = load(original_stop["registration_ref"])
    require(stop_reg["prior_native_ref"] == reg["native_ref"], "Original stop is the same branch")
    manifest = load(nreg["manifest_ref"])
    control_sources = [ref for ref in reg["native_source_refs"] if ref["path"].endswith("/utils/control_utils.py")]
    require(len(control_sources) == 1, "One native control utility source")
    functions, control_locations = native_functions(
        control_sources[0], ("orientation_error", "opspace_matrices", "nullspace_torques")
    )
    transform_ref = remember(reference(Path(control_sources[0]["path"]).with_name("transform_utils.py")))
    require(reg["native_transform_ref"] == transform_ref, "Exact registered native goal arithmetic source")
    transformations, transform_locations = native_functions(
        transform_ref, ("axisangle2quat", "quat2mat"), include_eps=True
    )
    for ref in [*reg["native_source_refs"], transform_ref]:
        require(manifest["execution_sha256"][ref["path"]] == ref["sha256"], "Pinned native execution source")
    helper_ref = remember(reference(REPO / "scripts/check_tracking_v5_state_updated_response.py"))
    require(
        helper_ref == load(prior_audit["registration_ref"])["source_ref"],
        "AST helper is the exact previously executed independent checker",
    )
    checker_ref = remember(reference(Path(__file__)))
    initial = np.asarray(load(reg["initial_ref"])["initial_joint"])
    control_record, tool = load(reg["control_ref"]), load(reg["tool_ref"])
    control = control_record["contract"]
    require(control["control_freq"] == 20 and control["model_timestep"] == 0.002, "Original control timing")
    site_from_tcp = np.linalg.solve(np.asarray(control_record["source_from_site"]), np.asarray(tool["source_from_tcp"]))
    kp, kd = np.asarray(control["kp"]), np.asarray(control["kd"])
    input_min, input_max = np.asarray(control["input_min"]), np.asarray(control["input_max"])
    output_min, output_max = np.asarray(control["output_min"]), np.asarray(control["output_max"])
    rows = load(reg["periods_ref"])
    commands = np.asarray(load(native["commands_ref"]))
    with np.load(checked(remember(reg["states_ref"])), allow_pickle=False) as archive:
        states = archive["sim_states"].copy()
    require(
        [row["step"] for row in rows] == [item["step"] for item in producer["history"]] == list(range(368, 377)),
        "All nine history entries, in order",
    )
    require(commands.shape == (377, 7) and len(states) == 378, "Complete original commands/endpoints")
    chunk = load(reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Chunk46 already available before branch")
    desired = np.asarray(chunk["effective_targets_tcp"])[9]
    previous = np.asarray(rows[-1]["command"])
    warm = np.asarray(native["stopped"]["answer"]["closest_linear_feasible_attempt"]["commands"])[0]
    np.testing.assert_array_equal(previous, original_stop["previous_command"])
    np.testing.assert_array_equal(warm, np.asarray(original_stop["original_candidate"]["commands"])[0])
    answer = producer["answer"]
    trace = answer["evaluated_candidates"]
    require(1 <= len(trace) == answer["response_evaluations"] <= 600, "All response evaluations included")
    registration = publish(
        ROOT / "registration.json",
        dict(
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            producer_owner_ref=producer_owner_ref,
            independent_response_owner_ref=reg["independent_response_owner_ref"],
            independent_native_owner_ref=prior_reg["independent_owner_ref"],
            source_ref=checker_ref,
            ast_helper_ref=helper_ref,
            native_control_source_ref=control_sources[0],
            native_transform_source_ref=transform_ref,
            native_control_function_lines=control_locations,
            native_transform_function_lines=transform_locations,
            seed=42,
            history_steps=list(range(368, 377)),
            endpoint=377,
            cursor=9,
            expected_trace_count=len(trace),
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            dynamics="Every query resets independent MjData to XML defaults, sets only current fingers and predicted arm q/v, calls mj_forward; no mj_step",
            equations="Archived native control utilities extracted by AST; independent recurrence; native quat2mat compiled with numba.njit(cache=False), axisangle2quat kept Python",
            independence="No import of new response, goal, dynamics query, causal history, constraint audit or planner implementations",
            history="Complete each predicted25-step period before reading its archived inner labels; update own monotonically accumulated empirical profile",
            trace="Recompute every unique evaluated command, all joint/action/pose predicates and deterministic selection ranks",
            qualification="Numerical reconstruction only; passed is not actual control, task or fresh qualification",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        history=[],
        trace_checks=[],
        numerical_inner_steps=0,
        mj_forward_queries=0,
        max_abs_difference={},
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
    )

    def compare(name, actual, expected, tolerance=NUMERIC_TOLERANCE):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(
            actual.shape == expected.shape and np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)),
            f"Finite matching shape: {name}",
        )
        difference = float(np.max(abs(actual - expected))) if actual.size else 0.0
        result["max_abs_difference"][name] = max(result["max_abs_difference"].get(name, 0.0), difference)
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0, err_msg=name)
        return difference

    try:
        guard = load(CONTROL_REF)["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        for ref in [guard["exact_probe_ref"], *guard["runtime_source_refs"]]:
            remember(ref)
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        import mujoco
        import numba

        quat2mat_jit = numba.njit(cache=False)(transformations["quat2mat"])
        require(hasattr(quat2mat_jit, "nopython_signatures"), "Native quaternion path must enable actual JIT")
        model = mujoco.MjModel.from_xml_path(str(checked(remember(reg["xml_ref"]))))
        data = mujoco.MjData(model)
        require(
            model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and model.opt.timestep == 0.002,
            "Native Euler2ms model",
        )
        result["native_runtime"] = dict(mujoco=mujoco.__version__, numpy=np.__version__, numba=numba.__version__)
        require(mujoco.__version__ == prior["native_mujoco_version"], "Unchanged native dynamics runtime")
        joints = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
        )
        require(
            np.all(joints >= 0) and np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE), "Seven arm joints"
        )
        qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, control["controller_site"])
        require(sid >= 0, "Native control SITE")
        fingers = {}
        for k in (1, 2):
            name = f"gripper0_finger_joint{k}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            require(jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE, "Two scalar finger coordinates")
            fingers[name] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
        aids = []
        for joint in joints:
            matched = np.flatnonzero((model.actuator_trnid[:, 0] == joint) & (model.actuator_trntype == 0))
            require(len(matched) == 1, "One direct arm motor")
            aids.append(matched[0])
        aids = np.asarray(aids)
        np.testing.assert_array_equal(model.actuator_gear[aids], np.tile([1, 0, 0, 0, 0, 0], (7, 1)))
        require(
            np.all(model.actuator_gaintype[aids] == 0)
            and np.all(model.actuator_gainprm[aids, 0] == 1)
            and np.all(model.actuator_biastype[aids] == 0)
            and np.all(model.actuator_dyntype[aids] == 0)
            and np.all(model.actuator_ctrllimited[aids])
            and not np.any(model.actuator_forcelimited[aids]),
            "Native unit torque actuators",
        )
        torque_limits, limits = model.actuator_ctrlrange[aids].copy(), model.jnt_range[joints].copy()
        lower, upper = limits[:, 0], limits[:, 1]
        require(states.shape == (378, 1 + model.nq + model.nv), "Every saved physical state coordinate")
        for key, value in dict(
            initial_joint=initial,
            kp=kp,
            kd=kd,
            site_from_tcp=site_from_tcp,
            desired_tcp=desired,
            previous_command=previous,
            warm=warm,
            torque_limits=torque_limits,
            joint_limits=limits,
        ).items():
            compare(key, value, producer[key])
        quaternion_blocks = []
        for joint in range(model.njnt):
            kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
            if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                quaternion_blocks.append(np.arange(address + 3, address + 7))
            elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
                quaternion_blocks.append(np.arange(address, address + 4))
        quaternion_ids = np.concatenate(quaternion_blocks) if quaternion_blocks else np.empty(0, dtype=int)
        scalar_ids = np.setdiff1d(np.arange(model.nq), quaternion_ids)

        def condition(step):
            state = states[step]
            fixed = {name: [state[1 + qi], state[1 + model.nq + vi]] for name, (qi, vi) in fingers.items()}
            return state[1 + qids].copy(), state[1 + model.nq + dofs].copy(), fixed

        def query(q, velocity, fixed):
            mujoco.mj_resetData(model, data)
            for name, (qi, vi) in fingers.items():
                data.qpos[qi], data.qvel[vi] = fixed[name]
            data.qpos[qids], data.qvel[dofs] = q, velocity
            before_q, before_v = data.qpos.copy(), data.qvel.copy()
            mujoco.mj_forward(model, data)
            result["mj_forward_queries"] += 1
            np.testing.assert_array_equal(data.qpos[scalar_ids], before_q[scalar_ids])
            for indices in quaternion_blocks:
                compare(
                    "quaternion_normalization",
                    data.qpos[indices],
                    before_q[indices] / np.linalg.norm(before_q[indices]),
                )
            np.testing.assert_array_equal(data.qvel, before_v)
            require(data.time == 0, "Quantity query never integrates time")
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            pose = np.eye(4)
            pose[:3, 3], pose[:3, :3] = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)
            damping = model.dof_damping[dofs].copy()
            return dict(
                mass=mass[np.ix_(dofs, dofs)],
                jacobian=np.vstack((jp, jr))[:, dofs],
                pose=pose,
                bias=data.qfrc_bias[dofs].copy(),
                damping=damping,
                passive_extra=data.qfrc_passive[dofs].copy() + damping * velocity,
            )

        def reconstruct(q0, v0, fixed, command, latched_rotation_goal=None):
            command = np.asarray(command, dtype=float)
            require(command.shape == (6,) and np.all(np.isfinite(command)), "Finite six-dimensional arm command")
            delta = output_min + (np.clip(command, input_min, input_max) - input_min) * (
                (output_max - output_min) / (input_max - input_min)
            )
            start = query(q0, v0, fixed)
            goal = start["pose"].copy()
            goal[:3, 3] += delta[:3]
            difference = 0.0
            if np.any(delta[3:] != 0):
                quaternion = transformations["axisangle2quat"](delta[3:])
                increment = quat2mat_jit(quaternion)
                python_increment = transformations["quat2mat"](quaternion)
                difference = float(np.max(abs(increment - python_increment)))
                result["max_jit_vs_python_goal_difference"] = max(
                    result["max_jit_vs_python_goal_difference"], difference
                )
                result["jit_python_goal_differences"] += int(difference > 0)
                goal[:3, :3] = increment @ start["pose"][:3, :3]
            else:
                latch = np.asarray(latched_rotation_goal)
                require(
                    latch.shape == (3, 3) and np.all(np.isfinite(latch)),
                    "Exact-zero command needs known historical orientation goal",
                )
                goal[:3, :3] = latch
            q, velocity = q0.copy(), v0.copy()
            records = []
            for inner in range(25):
                current = start if inner == 0 else query(q, velocity, fixed)
                mass, jacobian = current["mass"], current["jacobian"]
                _, lambda_pos, lambda_ori, null = functions["opspace_matrices"](
                    mass, jacobian, jacobian[:3], jacobian[3:]
                )
                rotation_error = functions["orientation_error"](goal[:3, :3], current["pose"][:3, :3])
                force = kp[:3] * (goal[:3, 3] - current["pose"][:3, 3]) - kd[:3] * (jacobian[:3] @ velocity)
                moment = kp[3:] * rotation_error - kd[3:] * (jacobian[3:] @ velocity)
                requested = jacobian.T @ np.r_[lambda_pos @ force, lambda_ori @ moment]
                requested += functions["nullspace_torques"](mass, null, initial, q, velocity, joint_kp=10)
                requested += current["bias"]
                applied = np.minimum(torque_limits[:, 1], np.maximum(torque_limits[:, 0], requested))
                velocity = np.linalg.solve(
                    mass + np.diag(0.002 * current["damping"]),
                    mass @ velocity + 0.002 * (applied - current["bias"] + current["passive_extra"]),
                )
                q = q + 0.002 * velocity
                records.append(
                    dict(
                        joint_position=q.copy(),
                        joint_velocity=velocity.copy(),
                        requested_torque=requested,
                        applied_torque=applied,
                        clipped=requested != applied,
                        mass_change=float(np.max(abs(mass - start["mass"]))),
                        jacobian_change=float(np.max(abs(jacobian - start["jacobian"]))),
                    )
                )
            endpoint = query(q, velocity, fixed)
            result["numerical_inner_steps"] += 25
            return (
                dict(
                    **{key: np.asarray([record[key] for record in records]) for key in records[0]},
                    goal=goal,
                    endpoint_pose=endpoint["pose"],
                    endpoint_twist=endpoint["jacobian"] @ velocity,
                    update_matrices=True,
                    dynamics_queries=26,
                ),
                start,
                difference,
            )

        def compare_response(actual, archived):
            require(set(actual) == set(archived), "Complete native-goal response fields")
            require(
                archived["update_matrices"] is True and archived["dynamics_queries"] == 26, "25x2ms updated response"
            )
            for key, value in actual.items():
                if key in ("update_matrices", "dynamics_queries"):
                    continue
                if key == "clipped":
                    np.testing.assert_array_equal(value, archived[key])
                else:
                    compare(
                        "response_" + key,
                        value,
                        archived[key],
                        TORQUE_TOLERANCE if "torque" in key else NUMERIC_TOLERANCE,
                    )

        raw_profile, maximum = np.zeros((25, 7)), np.zeros(7)

        def check_history(snapshot, completed):
            require(
                snapshot["available"] is (completed > 0)
                and snapshot["completed_periods"] == completed
                and snapshot["last_completed_step"] == (367 + completed if completed else None)
                and snapshot["future_guarantee"] is False,
                "Causal history count/ownership/scope",
            )
            compare("history_maximum", maximum, snapshot["position_error_rad"])
            compare("history_profile", np.maximum.accumulate(raw_profile, axis=0), snapshot["inner_position_error_rad"])

        for index, (item, row) in enumerate(zip(producer["history"], rows, strict=True)):
            step = row["step"]
            q0, v0, fixed = condition(step)
            np.testing.assert_array_equal(commands[step], row["command"])
            np.testing.assert_array_equal(commands[step], item["command"])
            np.testing.assert_array_equal(q0, item["q0"])
            np.testing.assert_array_equal(v0, item["v0"])
            require(set(fixed) == set(item["fixed_gripper_state"]), "Only current finger condition")
            for name in fixed:
                np.testing.assert_array_equal(fixed[name], item["fixed_gripper_state"][name])
            check_history(item["history_before"], index)
            prediction, start, difference = reconstruct(q0, v0, fixed, commands[step, :6])
            compare_response(prediction, item["response"])
            current = row["current_dynamics"]
            require(
                current["uncoupling"] is True and current["use_delta"] is True and current["impedance_mode"] == "fixed",
                "Original uncoupled fixed-delta controller",
            )
            for name in ("mass", "jacobian"):
                compare("native_start_" + name, start[name], current[name])
            np.testing.assert_array_equal(q0, current["robot_joint_position"])
            np.testing.assert_array_equal(v0, current["robot_joint_velocity"])
            np.testing.assert_array_equal(kp, current["kp"])
            np.testing.assert_array_equal(kd, current["kd"])
            # Only now consume this completed period's labels and update history.
            actual = np.asarray([sample["q"] for sample in row["actual_inner"]])
            actual_velocity = np.asarray([sample["qvel"] for sample in row["actual_inner"]])
            require(actual.shape == actual_velocity.shape == (25, 7), "Exactly25 actual inner labels")
            np.testing.assert_array_equal(actual, item["actual_inner_q"])
            np.testing.assert_array_equal(actual[-1], states[step + 1, 1 + qids])
            np.testing.assert_array_equal(actual_velocity[-1], states[step + 1, 1 + model.nq + dofs])
            compare(
                "actual_inner_times",
                [sample["time"] for sample in row["actual_inner"]],
                states[step, 0] + 0.002 * np.arange(1, 26),
            )
            residual = abs(actual - prediction["joint_position"])
            raw_profile = np.maximum(raw_profile, residual)
            maximum = np.maximum(maximum, residual.max(axis=0))
            check_history(item["history_after"], index + 1)
            result["history"].append(dict(step=step, inner_samples=25, native_jit_vs_python_goal_difference=difference))
        profile = np.maximum.accumulate(raw_profile, axis=0)
        latched_rotation_goal = prediction["goal"][:3, :3].copy()
        compare("historical_latched_rotation_goal", latched_rotation_goal, producer["latched_rotation_goal"])
        check_history(producer["history_before_search"], 9)
        compare("search_profile", profile, producer["past_inner_error_rad"])
        compare("answer_profile", profile, answer["past_inner_error_rad"])
        q0, v0, fixed = condition(377)
        np.testing.assert_array_equal(q0, producer["current_q"])
        np.testing.assert_array_equal(v0, producer["current_velocity"])
        require(set(fixed) == set(producer["fixed_gripper_state"]), "Current377 finger condition")
        for name in fixed:
            np.testing.assert_array_equal(fixed[name], producer["fixed_gripper_state"][name])
        current = query(q0, v0, fixed)
        require(set(current) == set(producer["current_quantities"]), "All current robot quantities")
        for name, value in current.items():
            compare("current_" + name, value, producer["current_quantities"][name])
        for name, value in dict(
            current_q=q0,
            current_velocity=v0,
            current_site=current["pose"],
            current_mass=current["mass"],
            current_jacobian=current["jacobian"],
        ).items():
            compare("original_" + name, value, original_stop[name])
        lo, hi = np.maximum(input_min, previous[:6] - 0.25), np.minimum(input_max, previous[:6] + 0.25)
        start = np.clip(warm, lo, hi)
        compare("command_lower", lo, answer["lower_command"])
        compare("command_upper", hi, answer["upper_command"])
        np.testing.assert_array_equal(warm, answer["supplied_warm"])
        np.testing.assert_array_equal(start, answer["used_start"])
        np.testing.assert_array_equal(start - warm, answer["start_roundoff_clip"])
        require(np.max(abs(start - warm)) <= 1e-12, "Only historical bound roundoff corrected")

        def classify(command, response, history_profile):
            q, velocity = response["joint_position"], response["joint_velocity"]
            margin = 0.05 + history_profile
            braking = q[-1] + 0.1 * velocity[-1]
            change = command - previous[:6]
            groups = dict(
                box_lower=(command[None] - input_min, 1e-12),
                box_upper=(input_max - command[None], 1e-12),
                rate_lower=((0.25 + change)[None], 1e-12),
                rate_upper=((0.25 - change)[None], 1e-12),
                inner_lower=(q - lower - margin, 1e-10),
                inner_upper=(upper - margin - q, 1e-10),
                braking_lower=((braking - lower - 0.05)[None], 1e-10),
                braking_upper=((upper - 0.05 - braking)[None], 1e-10),
                terminal=(np.empty(0), 1e-10),
            )
            joint_feasible = all(np.all(slack >= -tol) for slack, tol in groups.values())
            safety = np.concatenate(
                [groups[name][0].ravel() for name in ("inner_lower", "inner_upper", "braking_lower", "braking_upper")]
            )
            tcp = response["endpoint_pose"] @ site_from_tcp
            error = np.r_[
                desired[:3, 3] - tcp[:3, 3], Rotation.from_matrix(desired[:3, :3] @ tcp[:3, :3].T).as_rotvec()
            ]
            normalized = np.array([np.sum(error[:3] ** 2) / 0.01**2, np.sum(error[3:] ** 2) / 0.05**2])
            return dict(
                groups=groups,
                safety_slack=safety,
                safety_min=float(np.min(safety)),
                joint_feasible=bool(joint_feasible),
                tcp_error=error,
                normalized_squared_errors=normalized,
                maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
                pose_loss=float(np.sum(normalized)),
                feasible=bool(joint_feasible and np.all(normalized <= 1.0)),
            )

        def check_audit(archived, metrics):
            require(archived["horizon"] == 1 and archived["joint_count"] == 7, "H1 seven-joint audit")
            require(archived["activity_tolerance"] == 1e-8, "Original activity threshold")
            compare("audit_margin", 0.05 + profile, archived["position_margin_rad"])
            require(set(archived["groups"]) == set(metrics["groups"]), "All nine audit groups")
            for name, (slack, tolerance) in metrics["groups"].items():
                group = archived["groups"][name]
                saved = np.asarray(group["slack"]).reshape(slack.shape)
                compare("audit_" + name, slack, saved)
                require(group["tolerance"] == tolerance, "Original feasibility tolerance")
                if slack.size:
                    compare("audit_minimum_" + name, np.min(slack), group["minimum"])
                else:
                    require(group["minimum"] is None, "Empty terminal group has no minimum")
                # Activity is diagnostic; certify its exact archived arithmetic,
                # while independently checking all slack values and feasibility.
                for field, indices in dict(
                    active_indices=np.argwhere(saved <= 1e-8), violated_indices=np.argwhere(saved < -tolerance)
                ).items():
                    np.testing.assert_array_equal(np.asarray(group[field], dtype=int).reshape(-1, saved.ndim), indices)
                if name != "terminal":
                    compare("audit_first_" + name, np.min(slack), archived["first_period_minima"][name])
                    require(archived["future_period_minima"][name] is None, "H1 has no future minimum")
            require(archived["feasible"] is metrics["joint_feasible"], "Independent joint feasibility")

        full_entries = {
            name: answer[name] for name in ("initial", "selected", "closest_safety_attempt") if answer[name] is not None
        }
        wanted = {np.asarray(item["command"], dtype=float).tobytes() for item in full_entries.values()}
        retained, trace_keys, classified, archived_rank = {}, [], [], []
        for index, item in enumerate(trace):
            command = np.asarray(item["command"], dtype=float)
            require(
                command.shape == (6,) and np.all(command >= lo - 1e-12) and np.all(command <= hi + 1e-12),
                "Trace command inside box/rate",
            )
            key = command.tobytes()
            require(key not in trace_keys, "Each counted response has a unique exact command")
            trace_keys.append(key)
            response, _, difference = reconstruct(q0, v0, fixed, command, latched_rotation_goal)
            metrics = classify(command, response, profile)
            compare("trace_tcp_error", metrics["tcp_error"], item["tcp_error"])
            compare("trace_safety_minimum", metrics["safety_min"], item["safety_min"])
            require(
                item["joint_feasible"] is metrics["joint_feasible"] and item["feasible"] is metrics["feasible"],
                "Every trace feasibility label independently certified",
            )
            saved_error = np.asarray(item["tcp_error"])
            normalized = np.array([np.sum(saved_error[:3] ** 2) / 0.01**2, np.sum(saved_error[3:] ** 2) / 0.05**2])
            archived_rank.append((float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)), float(np.sum(normalized))))
            classified.append(metrics)
            if key in wanted:
                retained[key] = (response, metrics)
            result["trace_checks"].append(
                dict(
                    index=index,
                    joint_feasible=metrics["joint_feasible"],
                    feasible=metrics["feasible"],
                    safety_min=metrics["safety_min"],
                    normalized_squared_errors=metrics["normalized_squared_errors"],
                    native_jit_vs_python_goal_difference=difference,
                )
            )
        require(wanted <= set(retained), "Initial/selected/closest all belong to actual trace")
        np.testing.assert_array_equal(trace[0]["command"], start)
        for name, entry in full_entries.items():
            key = np.asarray(entry["command"], dtype=float).tobytes()
            response, metrics = retained[key]
            compare_response(response, entry["response"])
            check_audit(entry["audit"], metrics)
            for field in (
                "tcp_error",
                "normalized_squared_errors",
                "safety_slack",
                "maximum_pose_cap_violation",
                "pose_loss",
            ):
                compare("full_" + field, metrics[field], entry[field])
            require(entry["feasible"] is metrics["feasible"], "Full candidate feasibility agrees")
            if name == "initial":
                require(key == trace_keys[0], "Initial candidate is the single declared start")
        safe = [i for i, item in enumerate(trace) if item["joint_feasible"]]
        witnesses = [i for i in safe if trace[i]["feasible"]]
        require(answer["witness_found"] is bool(witnesses), "Witness means an independently certified cached command")
        require(
            answer["status"] == ("witness" if witnesses else "finite_search_without_witness"),
            "Truthful finite-search status",
        )
        if safe:
            expected = min(witnesses or safe, key=lambda i: archived_rank[i])
            require(answer["selected"] is not None, "Best safe command retained")
            np.testing.assert_array_equal(answer["selected"]["command"], trace[expected]["command"])
            result["selected_trace_index"] = expected
        else:
            require(answer["selected"] is None, "No safe response cannot produce a selected command")
            result["selected_trace_index"] = None
        expected_closest = min(
            range(len(trace)), key=lambda i: (max(0.0, -trace[i]["safety_min"]), archived_rank[i][1])
        )
        np.testing.assert_array_equal(answer["closest_safety_attempt"]["command"], trace[expected_closest]["command"])
        result["closest_safety_trace_index"] = expected_closest
        phases = answer["phases"]
        require(
            [p["phase"] for p in phases] in ([], ["safety"], ["pose"], ["safety", "pose"]),
            "At most one safety and one pose phase",
        )
        require(
            answer["max_evaluations"] == 600
            and answer["max_iterations"] == 60
            and answer["finite_difference_step"] == 1e-4
            and answer["optimization_safety_buffer"] == 1e-8,
            "Published fixed search budgets",
        )
        used = 0
        previous_phase_end = 1
        needs_safety = trace[0]["safety_min"] < 1e-8
        require(
            bool(phases and phases[0]["phase"] == "safety") is needs_safety,
            "Safety restoration triggered by initial slack",
        )
        for phase in phases:
            require(
                type(phase["iterations"]) is int and 0 <= phase["iterations"] <= phase["max_iterations"],
                "Phase iteration budget",
            )
            require(
                phase["max_iterations"] == (20 if phase["phase"] == "safety" else 60 - used),
                "Registered iteration allocation",
            )
            require(
                1 <= phase["evaluations_before"] <= phase["evaluations_after"] <= len(trace),
                "Phase response accounting",
            )
            require(phase["evaluations_before"] == previous_phase_end, "Phases share one response ledger")
            phase_key = np.asarray(phase["initial_command"], dtype=float).tobytes()
            require(
                phase_key in trace_keys[: phase["evaluations_before"]], "Phase starts from an already evaluated command"
            )
            before_index = trace_keys.index(phase_key)
            if phase["phase"] == "safety":
                require(before_index == 0, "Safety phase uses only the declared warm start")
            else:
                prior_safe = [i for i in range(phase["evaluations_before"]) if trace[i]["joint_feasible"]]
                require(bool(prior_safe), "Pose phase starts from a verified safe cached command")
                require(
                    before_index == min(prior_safe, key=lambda i: archived_rank[i]),
                    "Deterministic best safe phase start",
                )
            compare("phase_initial_safety", classified[before_index]["safety_min"], phase["initial_safety_min"])
            initial_epigraph = (
                max(0.0, 1e-8 - trace[before_index]["safety_min"])
                if phase["phase"] == "safety"
                else max(0.0, float(np.max(classified[before_index]["normalized_squared_errors"])) - 1.0)
            ) + 1e-6
            compare("phase_initial_epigraph", initial_epigraph, phase["initial_epigraph"])
            used += phase["iterations"]
            previous_phase_end = phase["evaluations_after"]
        require(used == answer["total_iterations"] <= 60, "Total iterations respected")
        require(
            result["numerical_inner_steps"] == 25 * (9 + len(trace)), "All history and search predictions reconstructed"
        )
        require(bool(quat2mat_jit.nopython_signatures), "Archived native quaternion function actually compiled")
        for ref in references.values():
            checked(ref)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            history_periods=9,
            actual_history_labels=225,
            trace_count=len(trace),
            full_candidates_checked=list(full_entries),
            unique_full_candidates=len(wanted),
            native_quaternion_jit_signatures=[str(value) for value in quat2mat_jit.nopython_signatures],
            witness_found=bool(witnesses),
            total_iterations=used,
            past_inner_error_rad=profile,
            source_refs_unchanged=list(references.values()),
            evidence="Independent numerical reconstruction of every evaluated command; no new physical control or task qualification",
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        publish(ROOT / "verified.json", result)


if __name__ == "__main__":
    main()
