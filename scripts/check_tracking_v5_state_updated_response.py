"""Independently reconstruct archived nonlinear robot response; no physics steps."""

import ast
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.joint_limit_response import build_joint_response
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/tracking_v5_state_updated_response_check_20261028"
PRODUCER = REPO / "outputs/tracking_v5_state_updated_response_20261028/verified.json"
MODES = (
    ("fixed_002", False, 0.002, 25),
    ("updated_002", True, 0.002, 25),
    ("fixed_001", False, 0.001, 50),
    ("updated_001", True, 0.001, 50),
)
NUMERIC_TOLERANCE = 1e-10
TORQUE_TOLERANCE = 1e-8


def native_functions(source_ref, names, *, include_eps=False):
    """Execute explicitly selected archived functions, without decorators."""
    path = checked(source_ref)
    names = set(names)
    parsed = ast.parse(path.read_text())
    definitions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
    selected = [node for node in definitions if node.name in names]
    require(len(selected) == len(names) and {node.name for node in selected} == names, "Exact native functions")
    locations = {node.name: dict(first_line=node.lineno, last_line=node.end_lineno) for node in selected}
    for node in selected:
        node.decorator_list = []
    constants = []
    if include_eps:
        constants = [
            node
            for node in parsed.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "EPS"
        ]
        require(len(constants) == 1, "One archived EPS assignment")
        locations["EPS"] = dict(first_line=constants[0].lineno, last_line=constants[0].end_lineno)
    module = ast.fix_missing_locations(ast.Module(body=constants + selected, type_ignores=[]))
    namespace = {"np": np, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}, locations


def main():
    strict_native_environment()
    require(not ROOT.exists(), "No overwrite or automatic retry")
    references = {}

    def remember(ref):
        checked(ref)
        require(ref["path"] not in references or references[ref["path"]] == ref, "No conflicting source SHA")
        references[ref["path"]] = ref
        return ref

    def load(ref):
        return read(remember(ref))

    producer_ref = remember(reference(PRODUCER))
    producer = load(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Complete numerical producer")
    require(
        producer["physics_steps"] == producer["world_model_calls"] == producer["fresh_cases"] == 0
        and producer["control_qualified"] is False,
        "Producer remains a numerical diagnosis without control qualification",
    )
    reg = load(producer["registration_ref"])
    require(reg["steps"] == list(range(368, 377)) and reg["modes"] == [list(mode) for mode in MODES], "Fixed roster")
    require(reg["seed"] == 42 and reg["automatic_retry"] is False, "Seed and no-retry contract")
    require(
        reg["budget"]
        == dict(
            periods=9,
            analytic_inner_steps=225,
            numerical_inner_steps=1350,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
        ),
        "Complete registered comparison budget",
    )
    for ref in reg["source_refs"] + reg["native_source_refs"]:
        remember(ref)
    control_sources = [ref for ref in reg["native_source_refs"] if ref["path"].endswith("/utils/control_utils.py")]
    require(len(control_sources) == 1, "Single archived native control utility source")
    functions, locations = native_functions(
        control_sources[0], ("orientation_error", "opspace_matrices", "nullspace_torques")
    )
    transform_ref = remember(reference(Path(control_sources[0]["path"]).with_name("transform_utils.py")))
    transformations, transform_locations = native_functions(
        transform_ref, ("axisangle2quat", "quat2mat"), include_eps=True
    )
    native = load(reg["native_ref"])
    require(
        native["complete"] is native["passed"] is True and native["continuation_steps"] == 9,
        "Exactly nine archived native periods",
    )
    nreg = load(native["registration_ref"])
    manifest = load(nreg["manifest_ref"])
    for ref in reg["native_source_refs"]:
        require(manifest["execution_sha256"][ref["path"]] == ref["sha256"], "Native source matches actual run")
    require(
        manifest["execution_sha256"][transform_ref["path"]] == transform_ref["sha256"],
        "Native float32 goal source matches actual run",
    )
    independent_owner = load(reg["independent_owner_ref"])
    require(independent_owner["complete"] is independent_owner["passed"] is True, "Closed independent replay owner")
    exit_record = load(independent_owner["owned_command"]["exit"])
    release = load(independent_owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Replay actual wait0/reap0",
    )
    require(independent_owner["result_ref"] == reg["independent_ref"], "Exact independent replay result")
    independent = load(reg["independent_ref"])
    require(independent["complete"] is independent["passed"] is True, "Independent native replay passed")
    require(load(independent["registration_ref"])["producer_ref"] == reg["native_ref"], "Same native branch")
    numeric = load(nreg["numeric_ref"])
    numeric_reg = load(numeric["registration_ref"])
    original = load(numeric_reg["producer_ref"])
    original_reg = load(original["registration_ref"])
    require(reg["initial_ref"] == original_reg["braking_numeric_ref"], "Original nullspace target source")
    for name in ("xml_ref", "control_ref", "tool_ref"):
        require(reg[name] == numeric_reg[name], "Exact native geometry and control source")
    for name in ("states_ref", "periods_ref"):
        require(reg[name] == native[name], "Exact native observation labels")
    initial = np.asarray(load(reg["initial_ref"])["initial_joint"])
    np.testing.assert_array_equal(producer["initial_joint"], initial)
    contract_record, tool = load(reg["control_ref"]), load(reg["tool_ref"])
    contract = contract_record["contract"]
    site_from_tcp = np.linalg.solve(
        np.asarray(contract_record["source_from_site"]), np.asarray(tool["source_from_tcp"])
    )
    np.testing.assert_allclose(site_from_tcp, producer["site_from_tcp"], atol=NUMERIC_TOLERANCE, rtol=0)
    rows = load(reg["periods_ref"])
    commands = np.asarray(load(native["commands_ref"]))
    with np.load(checked(remember(reg["states_ref"])), allow_pickle=False) as archive:
        states = archive["sim_states"].copy()
    require(len(states) == 378 and commands.shape == (377, 7), "Complete native prefix and endpoints")
    require(
        [row["step"] for row in rows] == [entry["step"] for entry in producer["entries"]] == list(range(368, 377)),
        "All nine entries in original order",
    )
    remember(reg["xml_ref"])
    checker_source = remember(reference(Path(__file__)))
    registration = publish(
        ROOT / "registration.json",
        dict(
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            native_ref=reg["native_ref"],
            independent_owner_ref=reg["independent_owner_ref"],
            xml_ref=reg["xml_ref"],
            source_ref=checker_source,
            native_control_source_ref=control_sources[0],
            native_transform_source_ref=transform_ref,
            native_transform_function_lines=transform_locations,
            native_function_lines=locations,
            seed=42,
            steps=list(range(368, 377)),
            modes=[list(mode) for mode in MODES],
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            dynamics="Independent MjModel/MjData, reset XML defaults on every query, restore only fixed current finger and predicted arm q/v, mj_forward; no mj_step",
            equations="Three archived native NumPy utility functions via AST without decorators; independent held-goal torque and implicit-damping recurrence",
            analytic_scope="Original analytical transition reused to reconstruct q/v; q additionally checked against prior native predicted_inner. Not an independent analytical derivation.",
            goal_precision="Archived native quat2mat casts quaternion to float32; producer uses double-precision SciPy Exp in both numerical arms. Record this semantic rounding difference separately; reconstruct the declared double-precision model without changing its tolerance.",
            qualification="Numerical reconstruction only; no new actual-control, task or fresh qualification",
            expected_analytic_steps=225,
            expected_numerical_steps=1350,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        entries=[],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        analytic_inner_steps=0,
        numerical_inner_steps=0,
        actual_inner_labels=0,
        mj_forward_queries=0,
        max_abs_difference={},
        max_native_goal_precision_difference=0.0,
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

        model = mujoco.MjModel.from_xml_path(str(checked(reg["xml_ref"])))
        data = mujoco.MjData(model)
        require(mujoco.__version__ == producer["native_mujoco_version"], "Same pinned native MuJoCo version")
        require(
            model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and model.opt.timestep == 0.002,
            "Native Euler2ms XML",
        )
        joints = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
        )
        require(
            np.all(joints >= 0) and np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE),
            "Seven scalar arm coordinates",
        )
        qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        require(sid >= 0, "Named control SITE")
        fingers = {}
        for k in (1, 2):
            name = f"gripper0_finger_joint{k}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            require(jid >= 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE, "Named scalar finger")
            fingers[name] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
        actuator_ids = []
        for joint in joints:
            selected = np.flatnonzero((model.actuator_trnid[:, 0] == joint) & (model.actuator_trntype == 0))
            require(len(selected) == 1, "Exactly one direct arm motor")
            actuator_ids.append(selected[0])
        aids = np.asarray(actuator_ids)
        np.testing.assert_array_equal(model.actuator_gear[aids], np.tile([1, 0, 0, 0, 0, 0], (7, 1)))
        require(
            np.all(model.actuator_gaintype[aids] == 0)
            and np.all(model.actuator_gainprm[aids, 0] == 1)
            and np.all(model.actuator_biastype[aids] == 0)
            and np.all(model.actuator_dyntype[aids] == 0)
            and np.all(model.actuator_ctrllimited[aids])
            and not np.any(model.actuator_forcelimited[aids]),
            "Unit direct torque without additional actuator dynamics",
        )
        torque_limits, limits = model.actuator_ctrlrange[aids], model.jnt_range[joints]
        np.testing.assert_array_equal(torque_limits, producer["torque_limits"])
        np.testing.assert_array_equal(limits, producer["joint_limits"])
        require(states.shape == (378, 1 + model.nq + model.nv), "All native state coordinates")
        quaternion_blocks = []
        for joint in range(model.njnt):
            kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
            if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                quaternion_blocks.append(np.arange(address + 3, address + 7))
            elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
                quaternion_blocks.append(np.arange(address, address + 4))
        quaternion_ids = np.concatenate(quaternion_blocks) if quaternion_blocks else np.empty(0, dtype=int)
        scalar_ids = np.setdiff1d(np.arange(model.nq), quaternion_ids)

        for entry, row in zip(producer["entries"], rows, strict=True):
            step, current = row["step"], row["current_dynamics"]
            q0, v0 = states[step, 1 + qids], states[step, 1 + model.nq + dofs]
            np.testing.assert_array_equal(q0, current["robot_joint_position"])
            np.testing.assert_array_equal(v0, current["robot_joint_velocity"])
            np.testing.assert_array_equal(q0, entry["q0"])
            np.testing.assert_array_equal(v0, entry["v0"])
            require(
                current["uncoupling"] is True and current["use_delta"] is True and current["impedance_mode"] == "fixed",
                "Fixed uncoupled delta OSC",
            )
            np.testing.assert_array_equal(commands[step], row["command"])
            np.testing.assert_array_equal(commands[step], entry["command"])
            kp, kd = np.asarray(current["kp"]), np.asarray(current["kd"])
            np.testing.assert_array_equal(kp, entry["kp"])
            np.testing.assert_array_equal(kd, entry["kd"])
            input_min, input_max = np.asarray(contract["input_min"]), np.asarray(contract["input_max"])
            output_min, output_max = np.asarray(contract["output_min"]), np.asarray(contract["output_max"])
            require(
                np.all(commands[step, :6] >= input_min) and np.all(commands[step, :6] <= input_max),
                "Issued arm command within fixed interface",
            )
            delta = output_min + (commands[step, :6] - input_min) * (output_max - output_min) / (input_max - input_min)
            compare("decoded_command", delta, entry["delta"])
            require(np.any(delta[3:] != 0), "Orientation reset has a nonzero command")
            fixed = {
                name: [states[step, 1 + qi], states[step, 1 + model.nq + vi]] for name, (qi, vi) in fingers.items()
            }
            require(set(fixed) == set(entry["fixed_gripper_state"]), "Only two fixed current fingers")
            for name in fixed:
                np.testing.assert_array_equal(fixed[name], entry["fixed_gripper_state"][name])

            def query(q, velocity):
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
                require(data.time == 0, "No integration during independent quantity query")
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
                    passive_extra=data.qfrc_passive[dofs].copy() + damping * data.qvel[dofs],
                )

            start = query(q0, v0)
            for name, value in start.items():
                compare(f"initial_{name}", value, entry["initial_quantities"][name])
            for name in ("mass", "jacobian"):
                compare(f"native_current_{name}", start[name], current[name])
            goal = start["pose"].copy()
            goal[:3, 3] += delta[:3]
            goal[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ goal[:3, :3]
            native_goal = goal.copy()
            native_goal[:3, :3] = (
                transformations["quat2mat"](transformations["axisangle2quat"](delta[3:])) @ start["pose"][:3, :3]
            )
            require(np.all(np.isfinite(native_goal)), "Finite native float32 goal diagnostic")
            goal_precision_difference = float(np.max(abs(native_goal - goal)))
            result["max_native_goal_precision_difference"] = max(
                result["max_native_goal_precision_difference"], goal_precision_difference
            )
            require(set(entry["methods"]) == {"analytic", *(mode[0] for mode in MODES)}, "Every registered arm")
            local = build_joint_response(start["mass"], start["jacobian"], kp, kd, q0, initial, null_kp=10.0, dt=0.002)
            z, analytic_q, analytic_v = np.r_[np.zeros(7), v0], [], []
            for _ in range(25):
                z = local["transition"] @ z + local["input"] @ delta + local["affine"]
                analytic_q.append(q0 + z[:7])
                analytic_v.append(z[7:])
            compare("analytic_archived_q", analytic_q, row["predicted_inner"])
            analytic = entry["methods"]["analytic"]
            require(analytic["dt"] == 0.002, "Original analytic timestep")
            compare("analytic_q", analytic_q, analytic["joint_position"])
            compare("analytic_velocity", analytic_v, analytic["joint_velocity"])
            result["analytic_inner_steps"] += 25
            method_checks = {}
            for mode, update, dt, count in MODES:
                archived = entry["methods"][mode]
                require(
                    archived["dt"] == dt
                    and archived["update_matrices"] is update
                    and archived["dynamics_queries"] == count + 1,
                    "Exact integration mode and budget",
                )
                require(
                    np.isfinite(archived["compute_seconds"]) and archived["compute_seconds"] >= 0,
                    "Recorded finite elapsed time",
                )
                compare("held_goal", goal, archived["goal"])
                q, velocity = q0.copy(), v0.copy()
                q_history, v_history, requested_history, applied_history = [], [], [], []
                clipped_history, mass_change, jacobian_change = [], [], []
                for inner in range(count):
                    state = start if inner == 0 else query(q, velocity)
                    mass = state["mass"] if update else start["mass"]
                    jacobian = state["jacobian"] if update else start["jacobian"]
                    _, lambda_pos, lambda_ori, null = functions["opspace_matrices"](
                        mass, jacobian, jacobian[:3], jacobian[3:]
                    )
                    rotation_error = functions["orientation_error"](goal[:3, :3], state["pose"][:3, :3])
                    force = kp[:3] * (goal[:3, 3] - state["pose"][:3, 3]) - kd[:3] * (jacobian[:3] @ velocity)
                    moment = kp[3:] * rotation_error - kd[3:] * (jacobian[3:] @ velocity)
                    task = jacobian.T @ np.r_[lambda_pos @ force, lambda_ori @ moment]
                    null_torque = functions["nullspace_torques"](mass, null, initial, q, velocity, joint_kp=10)
                    requested = task + null_torque + state["bias"]
                    applied = np.minimum(torque_limits[:, 1], np.maximum(torque_limits[:, 0], requested))
                    rhs = mass @ velocity + dt * (applied - state["bias"] + state["passive_extra"])
                    velocity = np.linalg.solve(mass + np.diag(dt * state["damping"]), rhs)
                    q = q + dt * velocity
                    q_history.append(q.copy())
                    v_history.append(velocity.copy())
                    requested_history.append(requested)
                    applied_history.append(applied)
                    clipped_history.append(requested != applied)
                    mass_change.append(float(np.max(abs(state["mass"] - start["mass"]))))
                    jacobian_change.append(float(np.max(abs(state["jacobian"] - start["jacobian"]))))
                checks = dict(
                    joint_position=compare("numerical_q", q_history, archived["joint_position"]),
                    joint_velocity=compare("numerical_velocity", v_history, archived["joint_velocity"]),
                    requested_torque=compare(
                        "requested_torque", requested_history, archived["requested_torque"], TORQUE_TOLERANCE
                    ),
                    applied_torque=compare(
                        "applied_torque", applied_history, archived["applied_torque"], TORQUE_TOLERANCE
                    ),
                )
                np.testing.assert_array_equal(clipped_history, archived["clipped"])
                compare("matrix_mass_change", mass_change, archived["mass_change"])
                compare("matrix_jacobian_change", jacobian_change, archived["jacobian_change"])
                endpoint = query(q, velocity)
                compare("endpoint_site_pose", endpoint["pose"], archived["endpoint_pose"])
                compare("endpoint_site_twist", endpoint["jacobian"] @ velocity, archived["endpoint_twist"])
                result["numerical_inner_steps"] += count
                method_checks[mode] = dict(
                    steps=count, max_abs_difference=checks, clipped_coordinates=int(np.count_nonzero(clipped_history))
                )

            actual_q = np.asarray([item["q"] for item in row["actual_inner"]])
            actual_v = np.asarray([item["qvel"] for item in row["actual_inner"]])
            require(actual_q.shape == actual_v.shape == (25, 7), "All actual inner labels")
            times = np.asarray([item["time"] for item in row["actual_inner"]])
            compare("actual_inner_time", times, states[step, 0] + 0.002 * np.arange(1, 26))
            np.testing.assert_array_equal(actual_q, entry["actual_q"])
            np.testing.assert_array_equal(actual_v, entry["actual_velocity"])
            np.testing.assert_array_equal(actual_q[-1], states[step + 1, 1 + qids])
            np.testing.assert_array_equal(actual_v[-1], states[step + 1, 1 + model.nq + dofs])
            actual_tcp = np.asarray(
                [query(q, v)["pose"] @ site_from_tcp for q, v in zip(actual_q, actual_v, strict=True)]
            )
            compare("actual_tcp_fk", actual_tcp, entry["actual_tcp"])
            actual_margin = np.minimum(actual_q - limits[:, 0], limits[:, 1] - actual_q)
            for mode, archived in entry["methods"].items():
                stride = 1 if mode == "analytic" else int(round(0.002 / archived["dt"]))
                pq = np.asarray(archived["joint_position"])[stride - 1 :: stride]
                pv = np.asarray(archived["joint_velocity"])[stride - 1 :: stride]
                require(pq.shape == pv.shape == (25, 7), "Predictions aligned to all25 native timestamps")
                tcp = np.asarray([query(q, v)["pose"] @ site_from_tcp for q, v in zip(pq, pv, strict=True)])
                translation = actual_tcp[:, :3, 3] - tcp[:, :3, 3]
                rotation = Rotation.from_matrix(actual_tcp[:, :3, :3] @ tcp[:, :3, :3].transpose(0, 2, 1)).as_rotvec()
                error = np.c_[translation, rotation]
                margin = np.minimum(pq - limits[:, 0], limits[:, 1] - pq)
                metrics = dict(
                    q_error=pq - actual_q,
                    velocity_error=pv - actual_v,
                    tcp_error=error,
                    margin_overestimate=margin - actual_margin,
                    max_joint_error=float(np.max(abs(pq - actual_q))),
                    max_velocity_error=float(np.max(abs(pv - actual_v))),
                    endpoint_position_error=float(np.linalg.norm(error[-1, :3])),
                    endpoint_rotation_error=float(np.linalg.norm(error[-1, 3:])),
                    max_margin_overestimate=float(np.max(margin - actual_margin)),
                )
                require(set(metrics) == set(archived["metrics"]), "Complete metric roster")
                for name, value in metrics.items():
                    compare(f"metric_{name}", value, archived["metrics"][name])
                if mode == "analytic":
                    method_checks[mode] = dict(
                        steps=25, archived_q_checked=True, independent_analytical_derivation=False
                    )
                method_checks[mode]["all_metrics_checked"] = True
            result["actual_inner_labels"] += 25
            result["entries"].append(
                dict(
                    step=step,
                    methods=method_checks,
                    actual_inner_labels=25,
                    native_float32_goal=native_goal,
                    published_double_exp_goal=goal,
                    native_goal_precision_max_difference=goal_precision_difference,
                )
            )
        require(
            len(result["entries"]) == 9
            and result["analytic_inner_steps"] == 225
            and result["numerical_inner_steps"] == 1350
            and result["actual_inner_labels"] == 225,
            "Complete9/225/1350 verification budget",
        )
        for ref in references.values():
            checked(ref)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            source_refs_unchanged=list(references.values()),
            case_count=9,
            numerical_mode_count=36,
            evidence="Independent numerical reconstruction only",
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        publish(ROOT / "verified.json", result)


if __name__ == "__main__":
    main()
