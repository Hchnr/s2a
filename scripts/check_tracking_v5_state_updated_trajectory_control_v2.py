"""Reconstruct the same complete trajectory audit with cheaper successful checks.

The original aborted checker remains immutable. Only the successful comparison
and query-integrity assertion paths change; all dynamic equations, trajectory
recurrences, constraints, selection rules and numerical tolerances are retained.
The original attempt must have actual terminal wait/reap and an explicit stop
record before this new version can start. It is not an automatic retry.
"""

import json
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

ROOT = REPO / "outputs/tracking_v5_state_updated_trajectory_control_v2_check_20261028"
PRODUCER = REPO / "outputs/tracking_v5_state_updated_trajectory_control_20261028/verified.json"
NUMERIC_TOLERANCE = 1e-10
TORQUE_TOLERANCE = 1e-8
HORIZON = 12


def finite_max_abs_difference(actual, expected, *, name="array"):
    """Return max absolute error for finite, same-shaped arrays (empty gives0)."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape and np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)),
        f"Finite matching shape: {name}",
    )
    return float(np.max(abs(actual - expected))) if actual.size else 0.0


def require_absolute_tolerance(difference, tolerance, *, name="array"):
    """Apply the original finite absolute comparison, with zero relative tolerance."""
    require(
        np.isfinite(difference) and np.isfinite(tolerance) and tolerance >= 0 and 0 <= difference <= tolerance,
        f"{name}: maximum absolute difference {difference!r} exceeds tolerance {tolerance!r}",
    )


def require_finite_array_equal(actual, expected, *, name="array"):
    """Require every finite same-shaped coordinate to be exactly unchanged."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and np.all(np.isfinite(actual))
        and np.all(np.isfinite(expected))
        and np.array_equal(actual, expected),
        f"Every quantity query preserves all finite {name} coordinates exactly",
    )


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
        launch = load(owner["owned_command"]["launch"])
        exited = load(owner["owned_command"]["exit"])
        released = load(owner["owned_command"]["release"])
        require(
            exited["actual_wait"] is True
            and exited["returncode"] == 0
            and released["reaped"] is True
            and released["returncode"] == 0
            and released["signal_sent"] is False
            and released["original_error"] is None,
            "Prerequisite actual wait0/reap0 without signal or original error",
        )
        require(
            released["pid"] == launch["child"]["pid"] and released["start_ticks"] == launch["child"]["start_ticks"],
            "Same owned process birth through release",
        )
        require(
            launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] == ""
            and launch["selected_environment"]["LD_LIBRARY_PATH"] == ""
            and launch["automatic_retry"] is False
            and exited["automatic_retry"] is False,
            "CPU-only prerequisite without automatic retry",
        )
        if expected_result is not None:
            require(owner["result_ref"] == expected_result, "Owner binds exact registered result")
        value = load(owner["result_ref"])
        require(value["complete"] is value["passed"] is True, "Prerequisite result passed")
        return owner, value

    producer_ref = remember(reference(PRODUCER))
    producer_owner_ref = remember(
        reference(REPO / "outputs/tracking_v5_state_updated_trajectory_control_execution_20261028/result.json")
    )
    producer_owner, producer = closed_owner(producer_owner_ref, producer_ref)
    owner_reg = load(producer_owner["registration_ref"])
    for path, sha256 in owner_reg["source_sha256"].items():
        remember(dict(path=path, sha256=sha256))
    producer_launch = load(producer_owner["owned_command"]["launch"])
    require(
        producer_launch["command"][-1] == str(REPO / "scripts/probe_tracking_v5_state_updated_trajectory_control.py"),
        "Owner actually ran the new trajectory producer",
    )
    require(
        producer["physics_steps"] == producer["world_model_calls"] == producer["fresh_cases"] == 0
        and producer["control_qualified"] is False,
        "Numerical search only, no action dispatch or physical qualification",
    )
    reg = load(producer["registration_ref"])
    require(
        reg["seed"] == 42
        and reg["endpoint"] == 368
        and reg["cursor"] == 0
        and reg["horizon"] == HORIZON
        and reg["target_cursors"] == list(range(HORIZON))
        and reg["initial_history_completed_periods"] == 0
        and reg["automatic_retry"] is False,
        "Single registered actual368 H12 roster and empty causal history",
    )
    require(
        reg["search"]["total_iterations"] == 60
        and reg["search"]["safety_iterations"] == 20
        and reg["search"]["response_budget"] == 4000
        and reg["search"]["difference_step"] == 1e-4
        and reg["search"]["optimization_safety_buffer"] == 1e-8
        and reg["search"]["ftol"] == 1e-12
        and reg["search"]["starts"] == 1,
        "Fixed single-start trajectory search budget and difference rule",
    )
    for name, value in dict(
        rate=0.25, joint_margin=0.05, braking_time=0.1, position_cap=0.01, rotation_cap=0.05, pose_caps_horizon=HORIZON
    ).items():
        require(reg["constraints"][name] == value, "Original trajectory conditions")
    normals = np.eye(7)[6:]
    np.testing.assert_array_equal(reg["constraints"]["terminal_normals"], normals)
    for ref in reg["source_refs"] + reg["native_source_refs"]:
        remember(ref)
    _, stop_check = closed_owner(reg["prior_stop_check_owner_ref"])
    stop_check_reg = load(stop_check["registration_ref"])
    require(stop_check_reg["producer_ref"] == reg["prior_stop_ref"], "Same closed stopped-state attempt")
    _, stop = closed_owner(stop_check_reg["producer_owner_ref"], reg["prior_stop_ref"])
    require(
        stop["answer"]["witness_found"] is False and stop_check["witness_found"] is False,
        "Transition after the registered failed stopped-state search",
    )
    stop_reg = load(stop["registration_ref"])
    early = load(reg["early_numeric_ref"])
    require(early["complete"] is early["passed"] is True, "Completed prior early-state warm source")
    early_reg = load(early["registration_ref"])
    _, early_check = closed_owner(reg["early_independent_owner_ref"])
    require(
        load(early_check["registration_ref"])["producer_ref"] == reg["early_numeric_ref"],
        "Independent checker binds exactly the earlier H12 warm",
    )
    require(reg["original_producer_ref"] == early_reg["producer_ref"], "Original physical branch source")
    original = load(reg["original_producer_ref"])
    require(original["complete"] is original["passed"] is True, "Original actual branch complete")
    original_reg = load(original["registration_ref"])
    require(
        reg["physical_independent_owner_ref"] == early_reg["independent_owner_ref"],
        "Same independent original physical replay owner",
    )
    _, physical_check = closed_owner(reg["physical_independent_owner_ref"])
    require(
        load(physical_check["registration_ref"])["producer_ref"] == reg["original_producer_ref"],
        "Exact original physical replay",
    )
    branches = [b for b in original["branches"] if b["branch"] == "causal_error"]
    require(len(branches) == 1, "Exactly one original causal_error branch")
    require(
        reg["states_ref"] == early_reg["states_ref"] == branches[0]["states_ref"],
        "State368 from the independently audited causal_error branch",
    )
    for name in ("xml_ref", "control_ref", "tool_ref", "chunks_ref"):
        require(reg[name] == early_reg[name], "Unchanged early-state geometry/control/available reference")
    require(
        reg["manifest_ref"] == original_reg["manifest_ref"]
        and reg["original_rows_ref"] == original_reg["rows_ref"]
        and reg["initial_ref"] == original_reg["braking_numeric_ref"],
        "Original controller history, execution and nullspace target sources",
    )
    require(reg["chunks_ref"] == original_reg["chunks_ref"], "Reference generated by this same original run")
    require(
        reg["native_source_refs"] == stop_reg["native_source_refs"]
        and reg["native_transform_ref"] == stop_reg["native_transform_ref"],
        "Same pinned native controller and JIT quaternion sources",
    )
    entries = [e for e in early["entries"] if e["step"] == 368 and e["horizon"] == HORIZON]
    require(len(entries) == 1, "Unique prior H12 actual368 warm trajectory")
    entry = entries[0]
    require(entry["cursor"] == 0 and entry["target_cursors"] == list(range(HORIZON)), "Same warm target roster")
    warm = np.asarray(entry["answer"]["planned_commands"], dtype=float)
    require(warm.shape == (HORIZON, 6), "Exact prior H12 planned_commands, no other start")
    np.testing.assert_array_equal(entry["answer"]["terminal_normals"], normals)
    profile = np.zeros((25, 7))
    np.testing.assert_array_equal(entry["causal_history"], profile)
    with np.load(checked(remember(reg["states_ref"])), allow_pickle=False) as archive:
        state = archive["sim_states"][368].copy()
    with checked(remember(reg["original_rows_ref"])).open() as stream:
        for index, line in enumerate(stream):
            if index == 367:
                previous_row = json.loads(line)
                break
        else:
            raise ValueError("Missing completed actual367 controller state")
    previous = np.asarray(previous_row["command"], dtype=float)
    np.testing.assert_array_equal(previous, entry["previous_command"])
    previous_site = np.asarray(previous_row["qualified_forward_input"]["x_site_pose"])[0]
    latch = np.asarray(previous_row["controller_goal_after"])
    require(
        previous.shape == (7,) and previous_site.shape == (4, 4) and latch.shape == (3, 3),
        "Actual previous command, SITE and native orientation goal",
    )
    chunk = load(reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Chunk46 is already generated at current368")
    desired = np.asarray(chunk["effective_targets_tcp"])[0:HORIZON].copy()
    require(desired.shape == (HORIZON, 4, 4), "All twelve available desired TCP poses")
    control_record, tool = load(reg["control_ref"]), load(reg["tool_ref"])
    control = control_record["contract"]
    require(control["control_freq"] == 20 and control["model_timestep"] == 0.002, "Original control timing")
    site_from_tcp = np.linalg.solve(np.asarray(control_record["source_from_site"]), np.asarray(tool["source_from_tcp"]))
    initial = np.asarray(load(reg["initial_ref"])["initial_joint"])
    kp, kd = np.asarray(control["kp"]), np.asarray(control["kd"])
    input_min, input_max = np.asarray(control["input_min"]), np.asarray(control["input_max"])
    output_min, output_max = np.asarray(control["output_min"]), np.asarray(control["output_max"])
    previous_delta = output_min + (np.clip(previous[:6], input_min, input_max) - input_min) * (
        (output_max - output_min) / (input_max - input_min)
    )
    require(np.any(previous_delta[3:] != 0), "Nonzero367 reset makes historical latch independently identifiable")
    manifest = load(reg["manifest_ref"])
    control_refs = [r for r in reg["native_source_refs"] if r["path"].endswith("/utils/control_utils.py")]
    require(len(control_refs) == 1, "Single original native control utility source")
    functions, control_locations = native_functions(
        control_refs[0], ("orientation_error", "opspace_matrices", "nullspace_torques")
    )
    transform_ref = remember(reg["native_transform_ref"])
    require(transform_ref == stop_check_reg["native_transform_source_ref"], "Previously checked native JIT source")
    transformations, transform_locations = native_functions(
        transform_ref, ("axisangle2quat", "quat2mat"), include_eps=True
    )
    for ref in [*reg["native_source_refs"], transform_ref]:
        require(manifest["execution_sha256"][ref["path"]] == ref["sha256"], "Actual original native execution source")
    helper_ref = remember(reference(REPO / "scripts/check_tracking_v5_state_updated_response.py"))
    require(helper_ref == stop_check_reg["ast_helper_ref"], "Unchanged independently executed AST helper")
    checker_ref = remember(reference(Path(__file__)))
    answer, trace = producer["answer"], producer["answer"]["evaluated_candidates"]
    require(1 <= len(trace) == answer["response_evaluations"] <= 4000, "All evaluated trajectories, within budget")
    # This version follows an explicitly terminated audit, not a successful old
    # audit or an untracked retry. Pin all actual terminal evidence before output.
    old_owner_root = REPO / "outputs/tracking_v5_state_updated_trajectory_control_check_execution_20261028"
    old_output_root = REPO / "outputs/tracking_v5_state_updated_trajectory_control_check_20261028"
    old_owner_ref = remember(reference(old_owner_root / "result.json"))
    old_owner = load(old_owner_ref)
    require(
        old_owner["complete"] is False and old_owner["passed"] is False,
        "Original audit remains an unsuccessful, incomplete attempt",
    )
    old_owner_reg = load(old_owner["registration_ref"])
    require(
        old_owner_reg["stage"] == "check" and old_owner_reg["automatic_retry"] is False,
        "Original checker had no automatic retry",
    )
    old_check_registration_ref = remember(reference(old_output_root / "registration.json"))
    old_check_reg = load(old_check_registration_ref)
    require(
        old_check_reg["producer_ref"] == producer_ref and old_check_reg["expected_trace_count"] == 4000,
        "New audit uses exactly the same full4000 producer trace",
    )
    old_checker_ref = remember(old_check_reg["source_ref"])
    require(
        old_owner_reg["source_sha256"][old_checker_ref["path"]] == old_checker_ref["sha256"],
        "Original executed checker source preserved",
    )
    require(not (old_output_root / "verified.json").exists(), "Aborted old audit has no complete numerical result")
    stop_decision_ref = remember(
        reference(REPO / "outputs/tracking_v5_state_updated_trajectory_control_check_stop_20261028/decision.json")
    )
    decision = load(stop_decision_ref)
    old_launch_ref = remember(reference(old_owner_root / "capture/launch.json"))
    old_exit_ref = remember(reference(old_owner_root / "capture/exit.json"))
    old_release_ref = remember(reference(old_owner_root / "capture/ownership/release.json"))
    old_log_ref = remember(reference(old_owner_root / "capture/command.log"))
    old_launch, old_exit, old_release = load(old_launch_ref), load(old_exit_ref), load(old_release_ref)
    require(
        decision["complete"] is True
        and decision["signal"] == "SIGTERM"
        and decision["automatic_retry"] is False
        and decision["launch_ref"] == old_launch_ref
        and decision["new_physics_steps"] == decision["new_world_model_calls"] == 0,
        "Explicit CPU-only single-child termination decision",
    )
    require(
        decision["reason"]
        == "Observed all-query assertion overhead cannot complete registered4000 trajectories within1200s. New version must preserve all numerical conditions and full scope.",
        "Termination was registered for assertion overhead, not numerical outcome selection",
    )
    require(
        old_exit["actual_wait"] is True
        and old_exit["returncode"] == -15
        and old_exit["automatic_retry"] is False
        and old_release["reaped"] is True
        and old_release["returncode"] == -15
        and old_release["signal_sent"] is False
        and old_release["original_error"] == old_owner["error"],
        "Original owner actually waited and reaped externally SIGTERM-terminated child",
    )
    require(
        old_launch["command"][-1] == old_checker_ref["path"]
        and decision["process_cmdline"].split() == old_launch["command"]
        and old_launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] == ""
        and old_launch["selected_environment"]["LD_LIBRARY_PATH"] == "",
        "Original signal targeted the exact CPU checker command",
    )
    for key in ("pid", "start_ticks"):
        require(
            decision["child"][key] == old_launch["child"][key] == old_release[key],
            "Same original child birth in stop decision, launch and actual release",
        )
    require(decision["child"]["ppid"] == old_launch["child"]["ppid"], "Same original owned parent")
    progress = []
    for line in checked(old_log_ref).read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "checked_trajectories" in item:
            progress.append(item)
    require(
        bool(progress)
        and progress[-1]
        == decision["last_progress"]
        == dict(checked_trajectories=500, total_trajectories=4000, numerical_inner_steps=150000),
        "Exactly recorded partial old audit remains incomplete",
    )
    version_plan_ref = remember(reference(REPO / "plans/v0.0.1/state_updated_trajectory_check_v2_20261028.md"))
    require(producer["numerical_inner_steps"] == len(trace) * HORIZON * 25, "Producer trajectory/inner-step accounting")
    registration = publish(
        ROOT / "registration.json",
        dict(
            producer_ref=producer_ref,
            producer_registration_ref=producer["registration_ref"],
            producer_owner_ref=producer_owner_ref,
            prior_stop_check_owner_ref=reg["prior_stop_check_owner_ref"],
            early_independent_owner_ref=reg["early_independent_owner_ref"],
            physical_independent_owner_ref=reg["physical_independent_owner_ref"],
            source_ref=checker_ref,
            ast_helper_ref=helper_ref,
            native_control_source_ref=control_refs[0],
            native_transform_source_ref=transform_ref,
            native_control_function_lines=control_locations,
            native_transform_function_lines=transform_locations,
            seed=42,
            endpoint=368,
            cursor=0,
            horizon=HORIZON,
            target_cursors=list(range(HORIZON)),
            expected_trace_count=len(trace),
            prior_aborted_owner_ref=old_owner_ref,
            prior_aborted_check_registration_ref=old_check_registration_ref,
            prior_aborted_source_ref=old_checker_ref,
            prior_aborted_launch_ref=old_launch_ref,
            prior_aborted_exit_ref=old_exit_ref,
            prior_aborted_release_ref=old_release_ref,
            prior_aborted_command_log_ref=old_log_ref,
            explicit_stop_decision_ref=stop_decision_ref,
            version_plan_ref=version_plan_ref,
            comparison_policy="Same finite/shape/absolute1e-10 or torque1e-8 checks; compare max absolute difference directly, no success-path testing assertion overhead",
            query_integrity_policy="Every query still checks all scalar qpos/qvel exactly and every normalized quaternion within the original tolerance",
            initial_history_completed_periods=0,
            numeric_tolerance=NUMERIC_TOLERANCE,
            torque_tolerance=TORQUE_TOLERANCE,
            dynamics="Independent resetData/current fixed fingers/predicted arm q-v/mj_forward; no mj_step",
            recurrence="Independent native AST OSC and implicit-damping Euler, carry predicted q/v and JIT goal through12x25 steps",
            independence="No import of new trajectory, response, goal, query, audit, history or optimizer implementations",
            history="Only current368 state and completed367 goal; zero own past-error profile, no future observations or labels",
            trace="Recompute every trajectory including difference probes; box checked, rate violations explicitly classified",
            constraints="All12 TCP caps, all300 inner margins, all12 braking endpoints, final +e7 terminal; original tolerances",
            qualification="Numerical reconstruction only; passed does not authorize physical action or certify control/task/fresh",
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
        trace_checks=[],
        numerical_inner_steps=0,
        mj_forward_queries=0,
        max_abs_difference={},
        max_jit_vs_python_goal_difference=0.0,
        jit_python_goal_differences=0,
    )

    def compare(name, actual, expected, tolerance=NUMERIC_TOLERANCE):
        difference = finite_max_abs_difference(actual, expected, name=name)
        result["max_abs_difference"][name] = max(result["max_abs_difference"].get(name, 0.0), difference)
        require_absolute_tolerance(difference, tolerance, name=name)
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
        require(result["native_runtime"] == stop_check["native_runtime"], "Same verified native numerical runtime")
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

        require(state.shape == (1 + model.nq + model.nv,), "Complete actual368 state vector")
        q0, v0 = state[1 + qids].copy(), state[1 + model.nq + dofs].copy()
        fixed = {name: [state[1 + qi], state[1 + model.nq + vi]] for name, (qi, vi) in fingers.items()}
        np.testing.assert_array_equal(q0, entry["current_q"])
        np.testing.assert_array_equal(v0, entry["current_velocity"])
        require(set(fixed) == set(producer["fixed_gripper_state"]), "Exactly two current fixed finger states")
        for name in fixed:
            np.testing.assert_array_equal(fixed[name], producer["fixed_gripper_state"][name])
        current = query(q0, v0, fixed)
        compare("earlier_current_site", current["pose"], entry["current_site"])
        require(set(current) == set(producer["current_quantities"]), "All current robot quantities")
        for name, value in current.items():
            compare("current_" + name, value, producer["current_quantities"][name])
        for name, value in dict(
            current_q=q0,
            current_velocity=v0,
            initial_joint=initial,
            kp=kp,
            kd=kd,
            torque_limits=torque_limits,
            joint_limits=limits,
            site_from_tcp=site_from_tcp,
            desired_tcp=desired,
            previous_command=previous,
            previous_site=previous_site,
            previous_physical_delta=previous_delta,
            latched_rotation_goal=latch,
            warm=warm,
            past_inner_error_rad=profile,
            terminal_normals=normals,
        ).items():
            compare("input_" + name, value, producer[name])
        prior_goal = quat2mat_jit(transformations["axisangle2quat"](previous_delta[3:])) @ previous_site[:3, :3]
        compare("actual367_latched_goal", prior_goal, latch)
        history = producer["history_before_search"]
        require(
            history["completed_periods"] == 0
            and history["last_completed_step"] is None
            and history["future_guarantee"] is False,
            "No native-goal response observations exist before this branch starts at368",
        )
        np.testing.assert_array_equal(history["inner_position_error_rad"], profile)
        np.testing.assert_array_equal(answer["past_inner_error_rad"], profile)
        np.testing.assert_array_equal(answer["terminal_normals"], normals)
        require(
            answer["horizon"] == answer["pose_caps_horizon"] == HORIZON and answer["all_pose_caps_enforced"] is True,
            "All twelve TCP caps and original final +e7 terminal remain enforced",
        )
        lo, hi = np.tile(input_min, (HORIZON, 1)), np.tile(input_max, (HORIZON, 1))
        lo[0], hi[0] = np.maximum(lo[0], previous[:6] - 0.25), np.minimum(hi[0], previous[:6] + 0.25)
        require(np.all(hi - lo > 1e-4), "Every normalized-command difference probe has room inside box")
        require(
            np.all(warm >= lo - 1e-12)
            and np.all(warm <= hi + 1e-12)
            and np.all(abs(np.diff(np.vstack((previous[:6], warm)), axis=0)) <= 0.25 + 1e-12),
            "Registered warm satisfies original box/rate before any search",
        )
        start = np.clip(warm, lo, hi)
        np.testing.assert_array_equal(answer["supplied_warm"], warm)
        np.testing.assert_array_equal(answer["used_start"], start)
        np.testing.assert_array_equal(answer["start_roundoff_clip"], start - warm)
        require(np.max(abs(start - warm)) <= 1e-12, "Only admitted warm bound roundoff clipped")
        compare("command_lower", lo, answer["lower_command"])
        compare("command_upper", hi, answer["upper_command"])

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
            require(len(periods) == HORIZON, "Exactly twelve autonomous predicted periods")
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
            require(len(archived["periods"]) == HORIZON, "All twelve full period responses retained")
            for response, saved in zip(actual["periods"], archived["periods"], strict=True):
                compare_response(response, saved)

        # Written independently from the shared production constraint audit.
        margin = np.tile(0.05 + profile.max(axis=0), (25 * HORIZON, 1))
        margin[:25] = 0.05 + profile
        safety_names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")

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
            return dict(
                groups=groups,
                safety_slack=safety,
                safety_min=float(np.min(safety)),
                all_constraint_min=float(min(np.min(v) for v, _ in groups.values())),
                joint_feasible=bool(safe),
                tcp_errors=errors,
                normalized_squared_errors=normalized,
                maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
                pose_loss=float(np.sum(normalized)),
                feasible=bool(safe and np.all(normalized <= 1.0)),
            )

        def check_audit(archived, metrics):
            require(archived["horizon"] == HORIZON and archived["joint_count"] == 7, "H12 seven-joint audit")
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

        full_entries = {
            name: answer[name] for name in ("initial", "selected", "closest_safety_attempt") if answer[name] is not None
        }
        require(
            "initial" in full_entries and "closest_safety_attempt" in full_entries, "Initial and closest fully retained"
        )
        wanted = {np.asarray(v["commands"], dtype=float).tobytes() for v in full_entries.values()}
        retained, trace_keys, trace_key_set = {}, [], set()
        classified, archived_rank, archived_safety_rank = [], [], []
        result["trace_rate_violations"] = 0
        for index, item in enumerate(trace):
            commands = np.asarray(item["commands"], dtype=float)
            require(
                commands.shape == (HORIZON, 6)
                and np.all(np.isfinite(commands))
                and np.all(commands >= lo - 1e-12)
                and np.all(commands <= hi + 1e-12),
                "Every trace command inside registered box and first-period bounds",
            )
            key = commands.tobytes()
            require(key not in trace_key_set, "Unique exact-byte trajectory accounting")
            trace_keys.append(key)
            trace_key_set.add(key)
            rollout, differences = reconstruct_trajectory(commands)
            metrics = classify(commands, rollout)
            compare("trace_tcp_errors", metrics["tcp_errors"], item["tcp_errors"])
            compare("trace_safety_minimum", metrics["safety_min"], item["safety_min"])
            require(
                item["joint_feasible"] is metrics["joint_feasible"] and item["feasible"] is metrics["feasible"],
                "Every evaluated trajectory eligibility independently reconstructed",
            )
            rates_ok = all(np.all(metrics["groups"][name][0] >= -1e-12) for name in ("rate_lower", "rate_upper"))
            result["trace_rate_violations"] += int(not rates_ok)
            saved_errors = np.asarray(item["tcp_errors"])
            normalized = np.stack(
                (
                    np.sum(saved_errors[:, :3] ** 2, axis=1) / 0.01**2,
                    np.sum(saved_errors[:, 3:] ** 2, axis=1) / 0.05**2,
                ),
                axis=1,
            )
            archived_rank.append((float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)), float(np.sum(normalized))))
            # The production closest rank includes box/rate as well as nonlinear
            # safety. Reconstruct it from checked saved arithmetic to preserve ties.
            saved_worst = min(
                float(item["safety_min"]), *(float(np.min(v)) for v, _ in command_groups(commands).values())
            )
            archived_safety_rank.append((max(0.0, -saved_worst), archived_rank[-1][1]))
            classified.append(
                {
                    k: metrics[k]
                    for k in (
                        "joint_feasible",
                        "feasible",
                        "safety_min",
                        "normalized_squared_errors",
                        "all_constraint_min",
                    )
                }
            )
            if key in wanted:
                retained[key] = (rollout, metrics)
            result["trace_checks"].append(
                dict(
                    index=index,
                    joint_feasible=metrics["joint_feasible"],
                    feasible=metrics["feasible"],
                    safety_min=metrics["safety_min"],
                    all_constraint_min=metrics["all_constraint_min"],
                    rate_feasible=bool(rates_ok),
                    normalized_squared_errors=metrics["normalized_squared_errors"],
                    native_jit_vs_python_goal_difference=max(differences),
                )
            )
            if (index + 1) % 100 == 0 or index + 1 == len(trace):
                print(
                    json.dumps(
                        dict(
                            checked_trajectories=index + 1,
                            total_trajectories=len(trace),
                            numerical_inner_steps=result["numerical_inner_steps"],
                        )
                    ),
                    flush=True,
                )
        require(wanted <= set(retained), "All fully retained candidates occur in the actual response trace")
        np.testing.assert_array_equal(trace[0]["commands"], start)
        for name, saved in full_entries.items():
            key = np.asarray(saved["commands"], dtype=float).tobytes()
            rollout, metrics = retained[key]
            compare_rollout(rollout, saved["rollout"])
            check_audit(saved["audit"], metrics)
            for field in (
                "tcp_errors",
                "normalized_squared_errors",
                "safety_slack",
                "maximum_pose_cap_violation",
                "pose_loss",
            ):
                compare("full_" + field, metrics[field], saved[field])
            require(saved["feasible"] is metrics["feasible"], "Full candidate eligibility agrees")
            if name == "initial":
                require(key == trace_keys[0], "Initial is exactly the one registered warm start")
        safe = [i for i, item in enumerate(trace) if item["joint_feasible"]]
        witnesses = [i for i in safe if trace[i]["feasible"]]
        require(answer["witness_found"] is bool(witnesses), "Witness requires every original action/joint/pose gate")
        require(
            answer["status"] == ("witness" if witnesses else "finite_search_without_witness"),
            "Truthful bounded-search status",
        )
        if safe:
            selected = min(witnesses or safe, key=lambda i: archived_rank[i])
            require(answer["selected"] is not None, "Best safe trajectory retained")
            np.testing.assert_array_equal(answer["selected"]["commands"], trace[selected]["commands"])
            result["selected_trace_index"] = selected
        else:
            require(answer["selected"] is None, "No safe trajectory cannot yield selected commands")
            result["selected_trace_index"] = None
        closest = min(range(len(trace)), key=lambda i: archived_safety_rank[i])
        np.testing.assert_array_equal(answer["closest_safety_attempt"]["commands"], trace[closest]["commands"])
        result["closest_safety_trace_index"] = closest
        require(
            answer["max_evaluations"] == 4000
            and answer["max_iterations"] == 60
            and answer["finite_difference_step"] == 1e-4
            and answer["optimization_safety_buffer"] == 1e-8,
            "Published fixed trajectory budget and difference arithmetic",
        )
        phases = answer["phases"]
        require(
            [p["phase"] for p in phases] in ([], ["safety"], ["pose"], ["safety", "pose"]),
            "At most one safety restoration and one pose phase",
        )
        needs_safety = trace[0]["safety_min"] < 1e-8
        require(
            bool(phases and phases[0]["phase"] == "safety") is needs_safety,
            "Safety restoration exactly when initial nonlinear safety needs it",
        )
        used, previous_end = 0, 1
        for phase in phases:
            require(
                type(phase["iterations"]) is int and 0 <= phase["iterations"] <= phase["max_iterations"],
                "Each phase stays inside its iteration budget",
            )
            require(
                phase["max_iterations"] == (20 if phase["phase"] == "safety" else 60 - used),
                "Registered safety/remaining-pose iteration allocation",
            )
            require(
                1 <= phase["evaluations_before"] <= phase["evaluations_after"] <= len(trace)
                and phase["evaluations_before"] == previous_end,
                "Every phase shares the one response ledger",
            )
            phase_key = np.asarray(phase["initial_commands"], dtype=float).tobytes()
            require(
                phase_key in trace_keys[: phase["evaluations_before"]],
                "Phase starts from an already evaluated trajectory",
            )
            before_index = trace_keys.index(phase_key)
            if phase["phase"] == "safety":
                require(before_index == 0, "Safety phase uses the unique declared warm start")
            else:
                prior_safe = [i for i in range(phase["evaluations_before"]) if trace[i]["joint_feasible"]]
                require(
                    bool(prior_safe) and before_index == min(prior_safe, key=lambda i: archived_rank[i]),
                    "Pose phase uses deterministic best already safe trajectory",
                )
            compare("phase_initial_safety", classified[before_index]["safety_min"], phase["initial_safety_min"])
            initial_epigraph = (
                max(0.0, 1e-8 - trace[before_index]["safety_min"])
                if phase["phase"] == "safety"
                else max(0.0, float(np.max(classified[before_index]["normalized_squared_errors"])) - 1.0)
            ) + 1e-6
            compare("phase_initial_epigraph", initial_epigraph, phase["initial_epigraph"])
            # Optimizer success is recorded by the producer but never certifies a witness.
            used += phase["iterations"]
            previous_end = phase["evaluations_after"]
        require(previous_end == len(trace), "All evaluations accounted for in declared phases or initial start")
        require(used == answer["total_iterations"] <= 60, "Shared total iteration budget respected")
        require(
            result["numerical_inner_steps"] == len(trace) * HORIZON * 25 <= 1200000,
            "Every unique complete trajectory was independently reconstructed",
        )
        require(
            result["mj_forward_queries"] == 1 + len(trace) * HORIZON * 26,
            "Independent current query plus every period's25 inner and endpoint queries",
        )
        require(bool(quat2mat_jit.nopython_signatures), "Native quaternion path really compiled in nopython JIT")
        for ref in references.values():
            checked(ref)
        strict_native_environment()
        result.update(
            complete=True,
            passed=True,
            endpoint=368,
            horizon=HORIZON,
            target_cursors=list(range(HORIZON)),
            initial_history_completed_periods=0,
            actual_history_labels=0,
            trace_count=len(trace),
            predicted_control_periods=len(trace) * HORIZON,
            full_candidates_checked=list(full_entries),
            unique_full_candidates=len(wanted),
            joint_feasible_trajectories=len(safe),
            witness_trajectories=len(witnesses),
            native_quaternion_jit_signatures=[str(v) for v in quat2mat_jit.nopython_signatures],
            witness_found=bool(witnesses),
            total_iterations=used,
            past_inner_error_rad=profile,
            terminal_normals=normals,
            source_refs_unchanged=list(references.values()),
            evidence="Every full causal trajectory independently reconstructed; no physical control/task/fresh qualification",
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        publish(ROOT / "verified.json", result)


if __name__ == "__main__":
    main()
