"""Compare frozen and state-updated robot predictions on nine archived periods."""

import os
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse, pose_error
from s2a.joint_limit_response import build_joint_response
from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.nonlinear_joint_response import rollout_nonlinear_joint_response
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.robot_site_kinematics import RobotSiteKinematics
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/tracking_v5_state_updated_response_20261028"
MODES = (
    ("fixed_002", False, 0.002, 25),
    ("updated_002", True, 0.002, 25),
    ("fixed_001", False, 0.001, 50),
    ("updated_001", True, 0.001, 50),
)


def main():
    strict_native_environment()
    require(not ROOT.exists(), "No overwrite or automatic retry")
    native_ref = reference(REPO / "outputs/tracking_v5_fk_first_long_h8_native_20261028/verified.json")
    owner_ref = reference(REPO / "outputs/tracking_v5_fk_first_long_h8_native_check_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Independent H8 replay closed")
    exited, released = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exited["actual_wait"] is True
        and exited["returncode"] == 0
        and released["reaped"] is True
        and released["returncode"] == 0,
        "Actual wait0/reap0",
    )
    independent = read(owner["result_ref"])
    require(independent["complete"] is independent["passed"] is True, "Independent H8 numerical replay passed")
    require(read(independent["registration_ref"])["producer_ref"] == native_ref, "Exact native source")
    native = read(native_ref)
    require(
        native["complete"] is native["passed"] is True and native["continuation_steps"] == 9,
        "Nine completed native periods",
    )
    nreg = read(native["registration_ref"])
    inputs = read(read(nreg["numeric_ref"])["registration_ref"])
    original = read(read(inputs["producer_ref"])["registration_ref"])
    initial_ref = original["braking_numeric_ref"]
    initial = np.asarray(read(initial_ref)["initial_joint"])
    contract_record = read(inputs["control_ref"])
    contract = contract_record["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    site_from_tcp = inverse(np.asarray(contract_record["source_from_site"])) @ np.asarray(
        read(inputs["tool_ref"])["source_from_tcp"]
    )
    rows = read(native["periods_ref"])
    require([r["step"] for r in rows] == list(range(368, 377)), "Exact fixed nine-period roster")
    with np.load(checked(native["states_ref"]), allow_pickle=False) as archive:
        states = archive["sim_states"].copy()
    require(len(states) == 378, "All archived endpoints")
    source_refs = [
        reference(Path(__file__)),
        *[
            reference(REPO / p)
            for p in (
                "src/s2a/nonlinear_joint_response.py",
                "src/s2a/mujoco_robot_dynamics.py",
                "src/s2a/joint_limit_response.py",
                "src/s2a/robot_site_kinematics.py",
                "plans/v0.0.1/state_updated_joint_response_20261028.md",
                "reports/state_updated_joint_response_source_audit_20261028.md",
            )
        ],
    ]
    manifest = read(nreg["manifest_ref"])
    package = REPO / "outputs/libero_env311/lib/python3.11/site-packages"
    native_sources = []
    for name in (
        "controllers/osc.py",
        "controllers/base_controller.py",
        "utils/control_utils.py",
        "robots/single_arm.py",
        "environments/base.py",
        "utils/binding_utils.py",
    ):
        ref = reference(package / "robosuite" / name)
        require(manifest["execution_sha256"][ref["path"]] == ref["sha256"], "Actual archived native source")
        native_sources.append(ref)
    registration = publish(
        ROOT / "registration.json",
        dict(
            native_ref=native_ref,
            independent_owner_ref=owner_ref,
            independent_ref=owner["result_ref"],
            states_ref=native["states_ref"],
            periods_ref=native["periods_ref"],
            initial_ref=initial_ref,
            xml_ref=inputs["xml_ref"],
            control_ref=inputs["control_ref"],
            tool_ref=inputs["tool_ref"],
            source_refs=source_refs,
            native_source_refs=native_sources,
            seed=42,
            steps=list(range(368, 377)),
            modes=[list(v) for v in MODES],
            initialization="One measured robot/gripper state at the start of each independent period; held actual issued command",
            recurrence="Predicted q/v only; same nonlinear FK, sine-angle OSC, bias, passive forces and clipping in both numerical modes; only control and plant M/J selection differs",
            integrator="Semi-implicit Euler with implicit diagonal damping; 2ms primary, 1ms numerical/feedback-resolution sensitivity (not native controller reconfiguration)",
            omitted="Contacts, joint limits, frictionloss constraints and gripper acceleration coupling; current finger q/v held fixed; no object state input",
            budget=dict(
                periods=9,
                analytic_inner_steps=225,
                numerical_inner_steps=1350,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
            automatic_retry=False,
            scope="Single-development-initial-state robot response diagnosis; no control/task qualification",
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
    )
    try:
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        import mujoco

        xml = checked(inputs["xml_ref"])
        names = [f"robot0_joint{k}" for k in range(1, 8)]
        shape = MujocoRobotDynamics(xml, contract["controller_site"], names, {})
        model, qids, dofs = shape.model, shape.qids, shape.dofs
        require(
            model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and model.opt.timestep == 0.002,
            "Archived native Euler2ms",
        )
        actuator_ids = []
        for jid in shape.joints:
            ids = np.flatnonzero((model.actuator_trnid[:, 0] == jid) & (model.actuator_trntype == 0))
            require(len(ids) == 1, "One direct joint motor per arm joint")
            actuator_ids.append(ids[0])
        aids = np.asarray(actuator_ids)
        np.testing.assert_array_equal(model.actuator_gear[aids], np.tile([1, 0, 0, 0, 0, 0], (7, 1)))
        require(
            np.all(model.actuator_gaintype[aids] == 0)
            and np.all(model.actuator_gainprm[aids, 0] == 1)
            and np.all(model.actuator_biastype[aids] == 0)
            and np.all(model.actuator_dyntype[aids] == 0)
            and np.all(model.actuator_ctrllimited[aids])
            and not np.any(model.actuator_forcelimited[aids]),
            "Fixed unit direct torque contract",
        )
        torque_limits, limits = model.actuator_ctrlrange[aids].copy(), model.jnt_range[shape.joints].copy()
        fk = RobotSiteKinematics(xml.read_text(), contract["controller_site"], names)
        result.update(
            initial_joint=initial,
            site_from_tcp=site_from_tcp,
            torque_limits=torque_limits,
            joint_limits=limits,
            native_mujoco_version=mujoco.__version__,
        )
        for row in rows:
            step, current = row["step"], row["current_dynamics"]
            q0, v0 = np.asarray(current["robot_joint_position"]), np.asarray(current["robot_joint_velocity"])
            state = states[step]
            np.testing.assert_array_equal(state[1 + qids], q0)
            np.testing.assert_array_equal(state[1 + model.nq + dofs], v0)
            gripper = {}
            for k in (1, 2):
                name = f"gripper0_finger_joint{k}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                require(jid >= 0, "Named current finger")
                gripper[name] = [state[1 + model.jnt_qposadr[jid]], state[1 + model.nq + model.jnt_dofadr[jid]]]
            dynamics = MujocoRobotDynamics(xml, contract["controller_site"], names, gripper)
            start = dynamics(q0, v0)
            for key in ("mass", "jacobian"):
                np.testing.assert_allclose(start[key], current[key], atol=1e-10, rtol=0)
            np.testing.assert_allclose(start["pose"], fk(q0)[0], atol=1e-10, rtol=0)
            require(
                current["uncoupling"] is True and current["use_delta"] is True and current["impedance_mode"] == "fixed",
                "Fixed native OSC contract",
            )
            delta = interface.decode(row["command"][:6])
            require(np.any(delta[3:] != 0), "Nonzero native orientation reset for this issued command")
            kp, kd = np.asarray(current["kp"]), np.asarray(current["kd"])
            local = build_joint_response(start["mass"], start["jacobian"], kp, kd, q0, initial, null_kp=10.0, dt=0.002)
            z = np.r_[np.zeros(7), v0]
            analytic = []
            for _ in range(25):
                z = local["transition"] @ z + local["input"] @ delta + local["affine"]
                analytic.append(np.r_[q0 + z[:7], z[7:]])
            analytic = np.asarray(analytic)
            np.testing.assert_allclose(analytic[:, :7], row["predicted_inner"], atol=1e-10, rtol=0)
            methods = dict(analytic=dict(joint_position=analytic[:, :7], joint_velocity=analytic[:, 7:], dt=0.002))
            # Predictions are complete before any inner feedback/endpoint labels are read below.
            for mode, update, dt, count in MODES:
                started = time.monotonic()
                pred = rollout_nonlinear_joint_response(
                    q0,
                    v0,
                    delta,
                    initial,
                    kp,
                    kd,
                    dynamics,
                    torque_limits[:, 0],
                    torque_limits[:, 1],
                    dt=dt,
                    substeps=count,
                    update_matrices=update,
                )
                methods[mode] = dict(**pred, dt=dt, compute_seconds=time.monotonic() - started)
            actual_q = np.asarray([v["q"] for v in row["actual_inner"]])
            actual_v = np.asarray([v["qvel"] for v in row["actual_inner"]])
            require(actual_q.shape == actual_v.shape == (25, 7), "Exactly25 actual inner labels")
            np.testing.assert_array_equal(actual_q[-1], states[step + 1][1 + qids])
            np.testing.assert_array_equal(actual_v[-1], states[step + 1][1 + model.nq + dofs])
            actual_tcp = np.asarray([fk(q)[0] @ site_from_tcp for q in actual_q])
            actual_margin = np.minimum(actual_q - limits[:, 0], limits[:, 1] - actual_q)
            for mode, pred in methods.items():
                stride = round(0.002 / pred["dt"])
                pq, pv = pred["joint_position"][stride - 1 :: stride], pred["joint_velocity"][stride - 1 :: stride]
                tcp = np.asarray([fk(q)[0] @ site_from_tcp for q in pq])
                error = np.asarray([pose_error(a, p) for a, p in zip(actual_tcp, tcp, strict=True)])
                margin = np.minimum(pq - limits[:, 0], limits[:, 1] - pq)
                pred["metrics"] = dict(
                    q_error=pq - actual_q,
                    velocity_error=pv - actual_v,
                    tcp_error=error,
                    margin_overestimate=margin - actual_margin,
                    max_joint_error=float(np.max(np.abs(pq - actual_q))),
                    max_velocity_error=float(np.max(np.abs(pv - actual_v))),
                    endpoint_position_error=float(np.linalg.norm(error[-1, :3])),
                    endpoint_rotation_error=float(np.linalg.norm(error[-1, 3:])),
                    max_margin_overestimate=float(np.max(margin - actual_margin)),
                )
            result["entries"].append(
                dict(
                    step=step,
                    q0=q0,
                    v0=v0,
                    command=row["command"],
                    delta=delta,
                    kp=kp,
                    kd=kd,
                    fixed_gripper_state=gripper,
                    initial_quantities=start,
                    methods=methods,
                    actual_q=actual_q,
                    actual_velocity=actual_v,
                    actual_tcp=actual_tcp,
                )
            )
        for ref in source_refs + native_sources:
            checked(ref)
        result.update(complete=True, passed=True)
    finally:
        publish(ROOT / "verified.json", result)


if __name__ == "__main__":
    main()
