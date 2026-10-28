"""Independently reconstruct the registered fixed saved-direction backtracking batch.

Only evidence loading is shared with production. Request construction, numerical
recurrence, constraint arithmetic and scores do not import production helpers.
The native recurrence is copied from the frozen H13 independent checker; its
HORIZON name is bound per case to twelve or thirteen in a fresh closure.
"""

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
from check_tracking_v5_prefix_preserving_trajectory_control import (
    finite_max_abs_difference,
    require_absolute_tolerance,
    require_finite_array_equal,
)
from check_tracking_v5_state_updated_response import native_functions
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.saved_direction_backtracking_evidence import (
    CHECK_ROOT,
    CPU_ENVIRONMENT,
    DIAGNOSTIC_REF,
    NATIVE_PYTHON,
    PLAN,
    ROOT,
    SOURCES,
    checked,
    closed_owner,
    finish,
    load_prerequisites,
    read,
    reference,
    require,
)
from s2a.saved_direction_backtracking_evidence import (
    remember as bind_reference,
)
from s2a.terminal_development_execution import CONTROL_REF

NUMERIC_TOLERANCE = 1e-10
TORQUE_TOLERANCE = 1e-8
ALPHAS = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
CASE_COUNTS = {"prefix": (12, 10, 9), "early": (13, 5, 10), "late": (13, 1, 10)}
FROZEN_CHECKER = REPO / "scripts/check_tracking_v5_paired_reference_phase.py"
NATIVE_FUNCTIONS = (
    "query",
    "reconstruct",
    "compare_response",
    "reconstruct_trajectory",
    "compare_rollout",
    "command_groups",
    "classify",
    "check_audit",
)


def native_math_proof():
    """Compare full ASTs, normalizing only three nonmathematical error messages."""
    messages = {
        "Exactly thirteen autonomous predicted periods": "Registered autonomous predicted periods",
        "All thirteen full period responses retained": "All registered full period responses retained",
        "H13 seven-joint audit": "Registered seven-joint audit",
    }

    class NormalizeMessages(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in messages:
                return ast.copy_location(ast.Constant(value=messages[node.value]), node)
            return node

    def functions(path):
        tree = NormalizeMessages().visit(ast.parse(Path(path).read_text()))
        return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    old, new = functions(FROZEN_CHECKER), functions(__file__)
    proof = {}
    for name in NATIVE_FUNCTIONS:
        proof[name] = ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)
        require(proof[name], f"Frozen independent numerical AST unchanged: {name}")
    return dict(
        source_ref=reference(FROZEN_CHECKER),
        functions=proof,
        horizon_binding="The unchanged HORIZON name is a per-case closure binding: prefix12, early13, late13",
        normalized_error_messages=messages,
    )


def normalized_errors(errors):
    errors = np.asarray(errors, dtype=float)
    require(errors.ndim == 2 and errors.shape[1] == 6 and np.all(np.isfinite(errors)), "Finite archived TCP errors")
    result = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    require(np.all(np.isfinite(result)), "Finite squared pose scores")
    return result


def rebuild_requests(diagnostic, producers):
    """Build all96 requests independently, preserving both exact endpoint byte arrays."""
    requests, unique, endpoint_count = [], set(), 0
    require(set(diagnostic["cases"]) == set(producers) == set(CASE_COUNTS), "Exactly the three preregistered cases")
    every_old_key = {
        np.asarray(row["commands"], dtype=float).tobytes()
        for source in producers.values()
        for row in source["answer"]["evaluated_candidates"]
    }
    for name, (horizon, direction_count, target) in CASE_COUNTS.items():
        case = diagnostic["cases"][name]
        source = producers[name]["answer"]
        trace = source["evaluated_candidates"]
        require(
            case["horizon"] == source["horizon"] == horizon and case["target_index"] == target,
            "Original cursor9 mapping",
        )
        old_keys = {np.asarray(row["commands"], dtype=float).tobytes() for row in trace}
        require(len(old_keys) == len(trace), "Original exact-byte trace is unique")
        require(all(type(pair["main_subset"]) is bool for pair in case["pairs"]), "Explicit saved main-subset labels")
        chosen = [(index, pair) for index, pair in enumerate(case["pairs"]) if pair["main_subset"] is True]
        require(len(chosen) == direction_count, "Every registered main-subset direction, without additions")
        for pair_index, pair in chosen:
            base_index, trial_index = pair["base_index"], pair["trial_index"]
            require(
                type(base_index) is int and type(trial_index) is int and 0 <= base_index < trial_index < len(trace),
                "Existing ordered direction endpoints",
            )
            base = np.asarray(trace[base_index]["commands"], dtype=float)
            trial = np.asarray(trace[trial_index]["commands"], dtype=float)
            require(
                base.shape == trial.shape == (horizon, 6) and np.all(np.isfinite(base)) and np.all(np.isfinite(trial)),
                "Complete finite source directions",
            )
            scores = normalized_errors(trace[base_index]["tcp_errors"])
            require_finite_array_equal(scores, pair["qbase"], name="registered source base scores")
            base_target_max = float(np.max(scores[target]))
            require(
                pair["phase_target_cursor"] == target and pair["prefix_before"] == target,
                "Registered failing cursor9 stage",
            )
            require(
                trace[base_index]["joint_feasible"] is True
                and trace[base_index]["safety_min"] >= 1e-8
                and np.all(scores[:target] <= 1),
                "Original admissible direction base",
            )
            require_absolute_tolerance(
                abs(base_target_max - pair["target"]["base_max"]), 1e-10, name="saved base target"
            )
            for alpha_index, alpha in enumerate(ALPHAS):
                commands = base.copy() if alpha == 0 else trial.copy() if alpha == 1 else base + alpha * (trial - base)
                key = commands.tobytes()
                require(key not in unique, "No within-batch or cross-case duplicate response")
                unique.add(key)
                source_index = base_index if alpha == 0 else trial_index if alpha == 1 else None
                require(
                    (key in every_old_key) == (source_index is not None), "Only the32 endpoints occur in any old trace"
                )
                endpoint_count += int(source_index is not None)
                requests.append(
                    dict(
                        case_name=name,
                        pair_index=pair_index,
                        base_index=base_index,
                        trial_index=trial_index,
                        alpha_index=alpha_index,
                        alpha=alpha,
                        prefix_before=target,
                        target_index=target,
                        horizon=horizon,
                        commands=commands,
                        source_trace_index=source_index,
                        base_target_max=base_target_max,
                    )
                )
    require(len(requests) == len(unique) == 96 and endpoint_count == 32, "Fixed96 responses and64 new commands")
    return requests


def backtracking_classification(metrics, request):
    """Derive diagnostic labels from independently reconstructed full constraints."""
    target = request["target_index"]
    target_max = float(np.max(metrics["normalized_squared_errors"][target]))
    improvement = float(request["base_target_max"] - target_max)
    prefix_preserved = bool(metrics["prefix_length"] >= request["prefix_before"])
    buffer_satisfied = bool(metrics["safety_min"] >= 1e-8)
    improved = bool(improvement > 1e-12)
    return dict(
        tcp_errors=metrics["tcp_errors"],
        normalized_squared_errors=metrics["normalized_squared_errors"],
        safety_min=metrics["safety_min"],
        prefix_length=metrics["prefix_length"],
        joint_feasible=metrics["joint_feasible"],
        prefix_preserved=prefix_preserved,
        buffer_satisfied=buffer_satisfied,
        target_max=target_max,
        target_improvement=improvement,
        target_improved=improved,
        target_passed=bool(np.all(metrics["normalized_squared_errors"][target] <= 1)),
        feasible=metrics["feasible"],
        feasible_descent=bool(metrics["joint_feasible"] and prefix_preserved and buffer_satisfied and improved),
    )


def endpoint_evidence(request, metrics, source, compare):
    """Old trace comparisons deliberately exclude unavailable full dynamics vectors."""
    index = request["source_trace_index"]
    if index is None:
        return None
    archived = source["answer"]["evaluated_candidates"][index]
    require(
        np.asarray(archived["commands"], dtype=float).tobytes() == request["commands"].tobytes(),
        "Exact old endpoint bytes",
    )
    tcp_difference = compare("old_endpoint_tcp_errors", metrics["tcp_errors"], archived["tcp_errors"])
    safety_difference = compare("old_endpoint_safety_min", metrics["safety_min"], archived["safety_min"])
    require(
        metrics["prefix_length"] == archived["prefix_length"]
        and metrics["joint_feasible"] is archived["joint_feasible"]
        and metrics["feasible"] is archived["feasible"],
        "Strict old endpoint classifications remain unchanged",
    )
    return dict(
        source_trace_index=index,
        fields_checked=["commands", "tcp_errors", "safety_min", "prefix_length", "joint_feasible", "feasible"],
        commands_exact_bytes=True,
        tcp_errors_max_abs_difference=tcp_difference,
        safety_min_abs_difference=safety_difference,
        strict_labels_equal=True,
        old_full_rollout_available=False,
    )


def score_phase_contracts(site_poses, case, site_from_tcp):
    """Recompute original timed, fixed retimed and scheduled original-node errors."""
    tcp = np.asarray(site_poses) @ np.asarray(site_from_tcp)
    require(tcp.shape == (13, 4, 4) and np.all(np.isfinite(tcp)), "All thirteen finite predicted TCP poses")

    def metric(desired, actual):
        rotation = Rotation.from_matrix(desired[:, :3, :3] @ actual[:, :3, :3].swapaxes(-1, -2)).as_rotvec()
        errors = np.c_[desired[:, :3, 3] - actual[:, :3, 3], rotation]
        normalized = np.column_stack(((errors[:, :3] ** 2).sum(1) / 0.01**2, (errors[:, 3:] ** 2).sum(1) / 0.05**2))
        require(np.all(np.isfinite(normalized)), "Finite dual-contract normalized pose scores")
        passed = np.all(normalized <= 1, axis=1)
        failures = np.flatnonzero(~passed)
        prefix = int(failures[0]) if len(failures) else len(passed)
        return dict(
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            pose_cap_pass=passed,
            all_pose_caps_satisfied=bool(np.all(passed)),
            prefix_length=prefix,
            maximum_normalized_squared_error=float(np.max(normalized)),
            maximum_pose_cap_violation=float(max(0, np.max(np.sqrt(normalized)) - 1)),
            pose_loss=float(np.sum(normalized)),
        )

    original = metric(np.asarray(case["original_timed_tcp"]), tcp)
    retimed = metric(np.asarray(case["desired_tcp"]), tcp)
    indices = np.asarray(case["original_node_endpoint_indices"], dtype=int)
    nodes = metric(np.asarray(case["original_timed_tcp"])[:12], tcp[indices])
    return dict(
        predicted_tcp=tcp,
        original_timed=original,
        retimed_path=retimed,
        original_nodes=nodes,
        predicted_qualified_original_node_prefix=nodes["prefix_length"],
        predicted_final_qualified_original_cursor=nodes["prefix_length"] - 1,
        actual_consumed_reference_count=0,
    )


def compare_contract(actual, saved, *, compare=None, name="contract"):
    """Compare every contract field, preserving strict integer/boolean labels."""
    if compare is None:

        def compare(label, left, right):
            difference = finite_max_abs_difference(left, right, name=label)
            require_absolute_tolerance(difference, NUMERIC_TOLERANCE, name=label)

    if isinstance(actual, dict):
        require(isinstance(saved, dict) and set(actual) == set(saved), f"Complete {name} fields")
        for key in actual:
            compare_contract(actual[key], saved[key], compare=compare, name=name + "." + key)
    elif isinstance(actual, (list, tuple)) and any(isinstance(item, dict) for item in actual):
        require(isinstance(saved, (list, tuple)) and len(saved) == len(actual), f"Complete {name} sequence")
        for index, (item, archived) in enumerate(zip(actual, saved, strict=True)):
            compare_contract(item, archived, compare=compare, name=f"{name}[{index}]")
    elif actual is None or isinstance(actual, (str, bool, int)):
        require(type(saved) is type(actual) and saved == actual, f"Exact {name} label")
    else:
        array, archived = np.asarray(actual), np.asarray(saved)
        if array.dtype.kind in "biu":
            kind_ok = archived.dtype.kind == "b" if array.dtype.kind == "b" else archived.dtype.kind in "iu"
            require(
                kind_ok and array.shape == archived.shape and np.array_equal(array, archived), f"Exact {name} labels"
            )
        else:
            compare(name, actual, saved)


def case_engine(name, original, source_reg, prefix_check_reg, result, refs, compare):
    """Create one independent native-query closure and reset every candidate to actual368."""
    HORIZON = CASE_COUNTS[name][0]

    def remember(ref):
        return bind_reference(ref, refs)

    def load(ref):
        return read(remember(ref))

    reg, producer = source_reg, original
    early = load(source_reg["early_numeric_ref"])
    entries = [entry for entry in early["entries"] if entry["step"] == 368 and entry["horizon"] == 12]
    require(len(entries) == 1, "Exactly one original actual368 source entry")
    entry = entries[0]
    with np.load(checked(remember(source_reg["states_ref"])), allow_pickle=False) as archive:
        state = archive["sim_states"][368].copy()
    with checked(remember(source_reg["original_rows_ref"])).open() as stream:
        for index, line in enumerate(stream):
            if index == 367:
                previous_row = json.loads(line)
                break
        else:
            raise ValueError("Missing completed actual367 controller state")
    previous = np.asarray(previous_row["command"], dtype=float)
    previous_site = np.asarray(previous_row["qualified_forward_input"]["x_site_pose"])[0]
    latch = np.asarray(previous_row["controller_goal_after"])
    control_record, tool = load(source_reg["control_ref"]), load(source_reg["tool_ref"])
    control = control_record["contract"]
    require(control["control_freq"] == 20 and control["model_timestep"] == 0.002, "Fixed native timing")
    site_from_tcp = np.linalg.solve(np.asarray(control_record["source_from_site"]), np.asarray(tool["source_from_tcp"]))
    initial = np.asarray(load(source_reg["initial_ref"])["initial_joint"])
    kp, kd = np.asarray(control["kp"]), np.asarray(control["kd"])
    input_min, input_max = np.asarray(control["input_min"]), np.asarray(control["input_max"])
    output_min, output_max = np.asarray(control["output_min"]), np.asarray(control["output_max"])
    previous_delta = output_min + (np.clip(previous[:6], input_min, input_max) - input_min) * (
        (output_max - output_min) / (input_max - input_min)
    )
    require(np.any(previous_delta[3:] != 0), "Completed367 reset determines the inherited rotation latch")
    profile, normals = np.zeros((25, 7)), np.eye(7)[6:]
    desired = np.asarray(original["desired_tcp"] if name == "prefix" else original["case_contract"]["desired_tcp"])
    require(desired.shape == (HORIZON, 4, 4), "Original complete case reference")
    _, stop_check = closed_owner(source_reg["prior_stop_check_owner_ref"], refs)
    stop_check_reg = load(stop_check["registration_ref"])
    control_refs = [ref for ref in source_reg["native_source_refs"] if ref["path"].endswith("/utils/control_utils.py")]
    require(len(control_refs) == 1, "Single pinned native OSC source")
    functions, control_locations = native_functions(
        remember(control_refs[0]), ("orientation_error", "opspace_matrices", "nullspace_torques")
    )
    transform_ref = remember(source_reg["native_transform_ref"])
    require(transform_ref == stop_check_reg["native_transform_source_ref"], "Same independently verified goal source")
    transformations, transform_locations = native_functions(
        transform_ref, ("axisangle2quat", "quat2mat"), include_eps=True
    )
    manifest = load(source_reg["manifest_ref"])
    for ref in [*source_reg["native_source_refs"], transform_ref]:
        remember(ref)
        require(
            manifest["execution_sha256"][ref["path"]] == ref["sha256"], "Original native execution source unchanged"
        )
    helper_ref = remember(reference(REPO / "scripts/check_tracking_v5_state_updated_response.py"))
    require(helper_ref == prefix_check_reg["ast_helper_ref"], "Same independently audited AST extraction source")
    result["native_function_sources"] = dict(
        control_ref=control_refs[0],
        transform_ref=transform_ref,
        ast_helper_ref=helper_ref,
        control_locations=control_locations,
        transform_locations=transform_locations,
    )
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
    require(result["native_runtime"] == stop_check["native_runtime"], "Same verified native numerical runtime")
    joints = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)])
    require(np.all(joints >= 0) and np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE), "Seven arm joints")
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
    quaternion_blocks = []
    for joint in range(model.njnt):
        kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            quaternion_blocks.append(np.arange(address + 3, address + 7))
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            quaternion_blocks.append(np.arange(address, address + 4))
    quaternion_ids = np.concatenate(quaternion_blocks) if quaternion_blocks else np.empty(0, dtype=int)
    scalar_ids = np.setdiff1d(np.arange(model.nq), quaternion_ids)

    def query(q, velocity, fixed):
        mujoco.mj_resetData(model, data)
        for name, (qi, vi) in fingers.items():
            data.qpos[qi], data.qvel[vi] = fixed[name]
        data.qpos[qids], data.qvel[dofs] = q, velocity
        before_q, before_v = data.qpos.copy(), data.qvel.copy()
        mujoco.mj_forward(model, data)
        result["mj_forward_queries"] += 1
        require_finite_array_equal(data.qpos[scalar_ids], before_q[scalar_ids], name="scalar qpos")
        for indices in quaternion_blocks:
            compare(
                "quaternion_normalization",
                data.qpos[indices],
                before_q[indices] / np.linalg.norm(before_q[indices]),
            )
        require_finite_array_equal(data.qvel, before_v, name="qvel")
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
            result["max_jit_vs_python_goal_difference"] = max(result["max_jit_vs_python_goal_difference"], difference)
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
            _, lambda_pos, lambda_ori, null = functions["opspace_matrices"](mass, jacobian, jacobian[:3], jacobian[3:])
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
        require(archived["update_matrices"] is True and archived["dynamics_queries"] == 26, "25x2ms updated response")
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

    def reconstruct_trajectory(commands):
        q, velocity, inherited_goal = q0.copy(), v0.copy(), latch.copy()
        periods, differences = [], []
        for command in commands:
            response, _, difference = reconstruct(q, velocity, fixed, command, inherited_goal)
            periods.append(response)
            differences.append(difference)
            q = response["joint_position"][-1].copy()
            velocity = response["joint_velocity"][-1].copy()
            inherited_goal = response["goal"][:3, :3].copy()
        require(len(periods) == HORIZON, "Registered autonomous predicted periods")
        return dict(
            inner_joint_position=np.concatenate([v["joint_position"] for v in periods]),
            inner_joint_velocity=np.concatenate([v["joint_velocity"] for v in periods]),
            joint_position=np.asarray([v["joint_position"][-1] for v in periods]),
            joint_velocity=np.asarray([v["joint_velocity"][-1] for v in periods]),
            pose=np.asarray([v["endpoint_pose"] for v in periods]),
            twist=np.asarray([v["endpoint_twist"] for v in periods]),
            goals=np.asarray([v["goal"] for v in periods]),
            periods=periods,
        ), differences

    def compare_rollout(actual, archived):
        require(set(actual) == set(archived), "Every aggregated trajectory field is present")
        for name in actual:
            if name != "periods":
                compare("trajectory_" + name, actual[name], archived[name])
        require(len(archived["periods"]) == HORIZON, "All registered full period responses retained")
        for response, saved in zip(actual["periods"], archived["periods"], strict=True):
            compare_response(response, saved)

    def command_groups(commands):
        change = np.diff(np.vstack((previous[:6], commands)), axis=0)
        return dict(
            box_lower=(commands - input_min, 1e-12),
            box_upper=(input_max - commands, 1e-12),
            rate_lower=(0.25 + change, 1e-12),
            rate_upper=(0.25 - change, 1e-12),
        )

    def classify(commands, rollout):
        inner_q = rollout["inner_joint_position"]
        braking = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
        groups = dict(
            **command_groups(commands),
            inner_lower=(inner_q - lower - margin, 1e-10),
            inner_upper=(upper - margin - inner_q, 1e-10),
            braking_lower=(braking - lower - 0.05, 1e-10),
            braking_upper=(upper - 0.05 - braking, 1e-10),
            terminal=(-normals @ rollout["joint_velocity"][-1], 1e-10),
        )
        require(all(np.all(np.isfinite(v)) for v, _ in groups.values()), "Finite independent constraint slacks")
        safe = all(np.all(slack >= -tol) for slack, tol in groups.values())
        safety = np.concatenate([groups[name][0].ravel() for name in safety_names])
        tcp = rollout["pose"] @ site_from_tcp
        rotation = Rotation.from_matrix(desired[:, :3, :3] @ np.swapaxes(tcp[:, :3, :3], -1, -2)).as_rotvec()
        errors = np.concatenate((desired[:, :3, 3] - tcp[:, :3, 3], rotation), axis=1)
        normalized = np.stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
        )
        require(np.all(np.isfinite(normalized)), "Finite independently reconstructed normalized pose errors")
        failed = np.flatnonzero(np.any(normalized > 1.0, axis=1))
        prefix = int(failed[0]) if len(failed) else HORIZON
        return dict(
            groups=groups,
            safety_slack=safety,
            safety_min=float(np.min(safety)),
            all_constraint_min=float(min(np.min(v) for v, _ in groups.values())),
            joint_feasible=bool(safe),
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            prefix_length=prefix,
            maximum_normalized_squared_error=float(np.max(normalized)),
            maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
            pose_loss=float(np.sum(normalized)),
            feasible=bool(safe and np.all(normalized <= 1.0)),
        )

    def check_audit(archived, metrics):
        require(archived["horizon"] == HORIZON and archived["joint_count"] == 7, "Registered seven-joint audit")
        require(archived["activity_tolerance"] == 1e-8, "Original activity threshold")
        compare("audit_margin", margin, archived["position_margin_rad"])
        require(set(archived["groups"]) == set(metrics["groups"]), "All nine audit constraint groups")
        nonterminal = set(metrics["groups"]) - {"terminal"}
        require(
            set(archived["first_period_minima"]) == set(archived["future_period_minima"]) == nonterminal,
            "Complete first/future grouped minima",
        )
        for name, (slack, tolerance) in metrics["groups"].items():
            group = archived["groups"][name]
            saved = np.asarray(group["slack"])
            compare("audit_" + name, slack, saved)
            require(group["tolerance"] == tolerance, "Original feasibility tolerance")
            compare("audit_minimum_" + name, np.min(slack), group["minimum"])
            # Exact activity/violation labels refer to the saved arithmetic;
            # the slacks and actual eligibility are independently recomputed.
            for field, indices in dict(
                active_indices=np.argwhere(saved <= 1e-8), violated_indices=np.argwhere(saved < -tolerance)
            ).items():
                np.testing.assert_array_equal(np.asarray(group[field], dtype=int).reshape(-1, saved.ndim), indices)
            if name != "terminal":
                count = 25 if name.startswith("inner_") else 1
                compare("audit_first_" + name, np.min(slack[:count]), archived["first_period_minima"][name])
                compare("audit_future_" + name, np.min(slack[count:]), archived["future_period_minima"][name])
        require(archived["feasible"] is metrics["joint_feasible"], "Independent full trajectory joint eligibility")

    require(state.shape == (1 + model.nq + model.nv,), "Complete actual368 state vector")
    q0, v0 = state[1 + qids].copy(), state[1 + model.nq + dofs].copy()
    fixed = {name: [state[1 + qi], state[1 + model.nq + vi]] for name, (qi, vi) in fingers.items()}
    require_finite_array_equal(q0, entry["current_q"], name="actual368 arm position")
    require_finite_array_equal(v0, entry["current_velocity"], name="actual368 arm velocity")
    compare_contract(fixed, producer["fixed_gripper_state"], compare=compare, name="actual368 fingers")
    current = query(q0, v0, fixed)
    compare("earlier_current_site", current["pose"], entry["current_site"])
    compare_contract(current, producer["current_quantities"], compare=compare, name="current_quantities")
    for name, value in dict(
        current_q=q0,
        current_velocity=v0,
        initial_joint=initial,
        kp=kp,
        kd=kd,
        torque_limits=torque_limits,
        joint_limits=limits,
        site_from_tcp=site_from_tcp,
        previous_command=previous,
        previous_site=previous_site,
        previous_physical_delta=previous_delta,
        latched_rotation_goal=latch,
        past_inner_error_rad=profile,
        terminal_normals=normals,
    ).items():
        compare("input_" + name, value, producer[name])
    prior_goal = quat2mat_jit(transformations["axisangle2quat"](previous_delta[3:])) @ previous_site[:3, :3]
    compare("actual367_latched_goal", prior_goal, latch)
    margin = np.tile(0.05 + profile.max(axis=0), (25 * HORIZON, 1))
    margin[:25] = 0.05 + profile
    safety_names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
    return reconstruct_trajectory, compare_rollout, classify, check_audit, current, quat2mat_jit


def main(dry=False):
    strict_native_environment()
    require(
        {key: os.environ.get(key) for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT,
        "Registered CPU and single-thread environment",
    )
    diagnostic, originals, registrations, refs = load_prerequisites()
    proof = native_math_proof()
    bind_reference(proof["source_ref"], refs)
    requests = rebuild_requests(diagnostic, originals)
    require(sum(request["horizon"] * 25 for request in requests) == 29700, "Fixed independent inner-step budget")
    require(3 + sum(request["horizon"] * 26 for request in requests) == 30891, "Fixed independent forward-query budget")
    finish(refs)
    if dry:
        print(
            json.dumps(
                dict(
                    ready=True,
                    dry=True,
                    requests=len(requests),
                    source_refs=len(refs),
                    new_responses=0,
                    new_queries=0,
                    numerical_inner_steps_if_executed=29700,
                    query_count_if_executed=30891,
                    numerical_math_proof=proof,
                )
            ),
            flush=True,
        )
        return
    require(not CHECK_ROOT.exists(), "New independent output directory; no overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_saved_direction_backtracking_execution_20261028/result.json")
    owner, producer = closed_owner(owner_ref, refs)
    producer_ref = reference(ROOT / "verified.json")
    require(owner["result_ref"] == producer_ref, "Owner binds this complete registered producer")
    launch = read(owner["owned_command"]["launch"])
    require(
        launch["command"]
        == [str(NATIVE_PYTHON), str(REPO / "scripts/probe_tracking_v5_saved_direction_backtracking.py")],
        "Only the unique registered producer command",
    )
    producer_reg = read(bind_reference(producer["registration_ref"], refs))
    require(
        producer_reg["diagnostic_ref"] == DIAGNOSTIC_REF
        and producer_reg["plan_ref"] == reference(PLAN)
        and producer_reg["seed"] == 42,
        "Fixed diagnosis, plan and seed",
    )
    diagnostic_reg = read(diagnostic["registration_ref"])
    require(producer_reg["producer_refs"] == diagnostic_reg["producer_refs"], "Same three original source producers")
    require(set(producer["cases"]) == set(CASE_COUNTS), "Exactly the three fixed cases")
    require(
        producer["response_evaluations"] == 96
        and producer["numerical_inner_steps"] == 29700
        and producer["query_count"] == 30891
        and producer["new_unique_commands"] == 64,
        "Complete fixed producer accounting",
    )
    require(
        producer["control_qualified"] is False
        and all(producer[key] == 0 for key in ("physics_steps", "world_model_calls", "fresh_cases")),
        "Numerical evidence cannot qualify or dispatch physical control",
    )
    for ref in producer_reg["source_refs"] + producer_reg["input_refs"]:
        bind_reference(ref, refs)
    prefix_check = read(
        reference(REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_check_20261028/verified.json")
    )
    prefix_check_reg = read(bind_reference(prefix_check["registration_ref"], refs))
    registration = publish(
        CHECK_ROOT / "registration.json",
        dict(
            producer_ref=producer_ref,
            producer_owner_ref=owner_ref,
            producer_registration_ref=producer["registration_ref"],
            diagnostic_ref=DIAGNOSTIC_REF,
            plan_ref=reference(PLAN),
            seed=42,
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            numerical_math_proof=proof,
            expected_responses=96,
            expected_inner_steps=29700,
            expected_queries=30891,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            independence="Independent96-command construction, frozen native AST/JIT query/recurrence, nine raw constraint groups and strict scores; metadata-only provenance is shared",
            endpoint_scope="Old endpoints compare only saved commands, TCP errors, scalar minimum and strict labels; full new dynamics vectors compare against this producer only",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        cases={},
        response_evaluations=0,
        numerical_inner_steps=0,
        query_count=0,
        new_unique_commands=64,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        retiming_qualified=False,
        task_qualified=False,
        boundary_classification_disagreements=[],
    )
    try:
        guard = read(bind_reference(CONTROL_REF, refs))["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        for ref in [guard["exact_probe_ref"], *guard["runtime_source_refs"]]:
            bind_reference(ref, refs)
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        for name, (horizon, direction_count, _) in CASE_COUNTS.items():
            saved_case = producer["cases"][name]
            case_requests = [request for request in requests if request["case_name"] == name]
            entries = saved_case["entries"]
            require(
                len(entries) == len(case_requests) == direction_count * 6,
                "Every fixed case request has one complete response",
            )
            report = dict(
                entries=[],
                response_evaluations=0,
                numerical_inner_steps=0,
                mj_forward_queries=0,
                max_abs_difference={},
                max_jit_vs_python_goal_difference=0.0,
                jit_python_goal_differences=0,
            )
            result["cases"][name] = report

            def compare(label, actual, expected, tolerance=NUMERIC_TOLERANCE):
                difference = finite_max_abs_difference(actual, expected, name=label)
                report["max_abs_difference"][label] = max(report["max_abs_difference"].get(label, 0.0), difference)
                require_absolute_tolerance(difference, tolerance, name=label)
                return difference

            reconstruct, compare_rollout, classify, check_audit, current, compiled_goal = case_engine(
                name,
                originals[name],
                registrations["prefix"],
                prefix_check_reg,
                report,
                refs,
                compare,
            )
            compare_contract(current, saved_case["current_quantities"], compare=compare, name="new_current_quantities")
            for request_index, (request, saved) in enumerate(zip(case_requests, entries, strict=True)):
                compare_contract(request, saved["request"], compare=compare, name="registered_request")
                require(
                    np.asarray(saved["request"]["commands"], dtype=float).tobytes() == request["commands"].tobytes(),
                    "All new command bytes use the fixed interpolation",
                )
                before_inner, before_queries = report["numerical_inner_steps"], report["mj_forward_queries"]
                rollout, differences = reconstruct(request["commands"])
                require(
                    report["numerical_inner_steps"] - before_inner == horizon * 25
                    and report["mj_forward_queries"] - before_queries == horizon * 26,
                    "Each alpha independently resets and reconstructs the complete horizon",
                )
                compare_rollout(rollout, saved["rollout"])
                metrics = classify(request["commands"], rollout)
                expected = backtracking_classification(metrics, request)
                archived = saved["classification"]
                require(
                    set(archived) == set(expected) | {"audit"}, "Every original classification field, without omissions"
                )
                mismatch = {
                    key: dict(independent=value, producer=archived[key])
                    for key, value in expected.items()
                    if isinstance(value, (bool, int))
                    and (type(archived[key]) is not type(value) or archived[key] != value)
                }
                if mismatch:
                    result["boundary_classification_disagreements"].append(
                        dict(
                            case_name=name,
                            request_index=request_index,
                            mismatch=mismatch,
                            independent_normalized=metrics["normalized_squared_errors"],
                            producer_normalized=archived["normalized_squared_errors"],
                            evidence="Strict numerical classification disagreement, not a physical-limit or search-success inference",
                        )
                    )
                require(not mismatch, "Strict independent backtracking labels agree without relaxed caps")
                compare_contract(
                    expected,
                    {key: archived[key] for key in expected},
                    compare=compare,
                    name="backtracking_classification",
                )
                check_audit(archived["audit"], metrics)
                endpoint = endpoint_evidence(request, metrics, originals[name], compare)
                compare_contract(endpoint, saved["endpoint_match"], compare=compare, name="endpoint_match")
                dual = (
                    None
                    if name == "prefix"
                    else score_phase_contracts(
                        rollout["pose"], originals[name]["case_contract"], np.asarray(originals[name]["site_from_tcp"])
                    )
                )
                compare_contract(dual, saved["dual_metrics"], compare=compare, name="dual_metrics")
                report["entries"].append(
                    dict(
                        request=request,
                        classification=expected,
                        endpoint_match=endpoint,
                        dual_metrics=dual,
                        maximum_native_jit_vs_python_goal_difference=max(differences),
                    )
                )
                report["response_evaluations"] += 1
                print(json.dumps(dict(case_name=name, checked=request_index + 1, total=len(entries))), flush=True)
            report["query_count"] = report["mj_forward_queries"]
            require(
                report["response_evaluations"] == saved_case["response_evaluations"] == direction_count * 6
                and report["numerical_inner_steps"]
                == saved_case["numerical_inner_steps"]
                == direction_count * 6 * horizon * 25
                and report["query_count"] == saved_case["query_count"] == direction_count * 6 * horizon * 26 + 1,
                "Exact complete per-case response and query accounting",
            )
            require(bool(compiled_goal.nopython_signatures), "Native quaternion goal actually compiled")
            report["native_quaternion_jit_signatures"] = [str(value) for value in compiled_goal.nopython_signatures]
            result["response_evaluations"] += report["response_evaluations"]
            result["numerical_inner_steps"] += report["numerical_inner_steps"]
            result["query_count"] += report["query_count"]
        require(
            result["response_evaluations"] == 96
            and result["numerical_inner_steps"] == 29700
            and result["query_count"] == 30891,
            "Entire96-response independent budget",
        )
        for ref in producer["source_refs_unchanged"]:
            bind_reference(ref, refs)
        finish(refs)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            source_refs_unchanged=list(refs.values()),
            endpoint_count=32,
            evidence="Independent full reconstruction of the fixed96 numerical responses only; no actual control or task qualification",
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        publish(CHECK_ROOT / "verified.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Verify only closed evidence, fixed requests and source math; create no output and issue no queries",
    )
    main(parser.parse_args().dry)
