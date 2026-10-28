"""Audit actual377 constraints and two one-period FK feasibility diagnostics."""

import json
import os
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.fk_consistent_joint_preview import fk_consistent_rollout
from s2a.fk_pose_cap_scope_joint_plan import solve_interior_plan
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse, pose_error, transform
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.robot_site_kinematics import RobotSiteKinematics
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_fk_stop_constraint_audit_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_fk_first_long_h8_native_check_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed independent actual H8 replay")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0/reap0",
    )
    native_ref = read(read(owner["result_ref"])["registration_ref"])["producer_ref"]
    native = read(native_ref)
    require(
        native["complete"] is native["passed"] is True
        and native["continuation_steps"] == 9
        and native["stopped"]["step"] == 377
        and native["stopped"]["action_dispatched"] is False,
        "Exact default-H8 actual stopped endpoint377",
    )
    nreg = read(native["registration_ref"])
    numeric_ref = nreg["numeric_ref"]
    inputs = read(read(numeric_ref)["registration_ref"])
    original_reg = read(read(inputs["producer_ref"])["registration_ref"])
    initial = np.asarray(read(original_reg["braking_numeric_ref"])["initial_joint"])
    control = read(inputs["control_ref"])
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(
        read(inputs["tool_ref"])["source_from_tcp"]
    )
    rows = read(native["periods_ref"])
    require(rows[-1]["step"] == 376, "Only completed own branch history")
    profile, previous = np.asarray(rows[-1]["history_after"]), np.asarray(rows[-1]["command"])
    stop_answer = native["stopped"]["answer"]
    np.testing.assert_array_equal(profile, native["stopped"]["history_before"])
    np.testing.assert_array_equal(profile, stop_answer["past_inner_error_rad"])
    np.testing.assert_array_equal(np.tile(previous[:6], (8, 1)), stop_answer["seed_commands"])
    closest = stop_answer["closest_linear_feasible_attempt"]
    require(closest is not None, "Saved closest checked H8 candidate")
    original_commands = np.asarray(closest["commands"])
    require(original_commands.shape == (8, 6), "Complete archived H8 plan")
    normals = np.asarray(stop_answer["terminal_normals"]).reshape(-1, 7)
    warm = original_commands[:1].copy()
    with np.load(checked(native["states_ref"]), allow_pickle=False) as archive:
        require(len(archive["sim_states"]) == 378, "All endpoints through377")
        state = archive["sim_states"][-1].copy()
    chunk = read(nreg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Already generated chunk46")
    targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
    sources = [
        reference(Path(__file__)),
        *[
            reference(REPO / "src/s2a" / name)
            for name in (
                "joint_preview_constraint_audit.py",
                "fk_pose_cap_scope_joint_plan.py",
                "fk_consistent_joint_preview.py",
                "robot_site_kinematics.py",
                "interior_joint_preview.py",
                "joint_limit_preview_objective.py",
            )
        ],
    ]
    registration = publish(
        root / "registration.json",
        dict(
            independent_owner_ref=owner_ref,
            prior_native_ref=native_ref,
            numeric_ref=numeric_ref,
            xml_ref=inputs["xml_ref"],
            control_ref=inputs["control_ref"],
            tool_ref=inputs["tool_ref"],
            chunks_ref=nreg["chunks_ref"],
            source_refs=sources,
            seed=42,
            endpoint=377,
            cursor=9,
            modes=["early_terminal", "no_terminal"],
            horizon=1,
            joint_count=7,
            supplied_warm="First command of archived closest H8 linear-feasible candidate, unchanged between both modes",
            seed_policy="Last actually executed command repeated once",
            terminal_policy="Original H8 direction normals applied at50ms (previously400ms), versus no terminal directions; first-period box/rate/inner/braking unchanged",
            start_policy="Same supplied warm; retain solver-reported LP replacement and actual start policy differences",
            per_search_response_budget=160,
            total_response_budget=320,
            per_search_iteration_budget=60,
            original_pose_caps=[0.01, 0.05],
            original_inner_margin_rad=0.05,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Numerical constraint attribution only; H1 also drops future costs and constraints; no action dispatch or qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(checked(inputs["xml_ref"])))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
    require(all(v >= 0 for v in joints), "Actual robot joints")
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    require(sid >= 0, "Actual controller SITE")
    data.time = state[0]
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq :]
    mujoco.mj_forward(model, data)
    q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    mass, jac = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
    site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
    fixed = rows[-1]["current_dynamics"]
    local = build_joint_preview(mass, jac, fixed["kp"], fixed["kd"], q, initial, null_kp=10.0, dt=0.002)
    limits = model.jnt_range[joints]
    desired = targets[9:10]
    twists = np.asarray([reference_world_twist(targets, 9, anchor, 0.05)["twist"]])

    def new_fk():
        return RobotSiteKinematics(
            checked(inputs["xml_ref"]).read_text(),
            contract["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
        )

    def candidate_record(commands, direction_normals, expected_errors):
        commands = np.asarray(commands)
        raw = rollout_joint_preview(
            local,
            velocity,
            np.asarray([interface.decode(c) for c in commands]),
            site,
            limits[:, 0],
            limits[:, 1],
            enforce_limits=False,
        )
        rollout = fk_consistent_rollout(raw, new_fk())
        errors = np.asarray(
            [pose_error(targets[9 + k], pose @ site_from_tcp) for k, pose in enumerate(rollout["pose"])]
        )
        np.testing.assert_allclose(errors, expected_errors, atol=1e-10, rtol=0)
        audit = audit_joint_preview_constraints(
            rollout,
            commands,
            previous[:6],
            interface,
            limits[:, 0],
            limits[:, 1],
            past_inner_error_rad=profile,
            terminal_normals=direction_normals,
        )
        require(audit["feasible"] is True, "Independently rebuilt original joint/action constraints")
        return dict(
            commands=commands,
            all_tcp_errors=errors,
            audit=audit,
            rollout={
                k: rollout[k] for k in ("joint_position", "joint_velocity", "inner_joint_position", "pose", "twist")
            },
        )

    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        entries=[],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        current_q=q,
        current_velocity=velocity,
        current_site=site,
        current_mass=mass,
        current_jacobian=jac,
        initial_joint=initial,
        kp=fixed["kp"],
        kd=fixed["kd"],
        lower=limits[:, 0],
        upper=limits[:, 1],
        history_before=profile,
        previous_command=previous,
    )
    try:
        result["original_candidate"] = candidate_record(original_commands, normals, closest["all_tcp_errors"])
        np.testing.assert_array_equal(
            result["original_candidate"]["audit"]["position_margin_rad"], stop_answer["requested_position_margin_rad"]
        )
        for mode, directions in (("early_terminal", normals), ("no_terminal", np.empty((0, 7)))):
            started = time.monotonic()
            answer = solve_interior_plan(
                local,
                velocity,
                previous[:6][None, :],
                previous[:6],
                interface,
                site,
                site_from_tcp,
                desired,
                twists,
                limits[:, 0],
                limits[:, 1],
                fk_jacobian=new_fk(),
                pose_cap_periods=1,
                initial_guess=warm,
                past_inner_error_rad=profile,
                terminal_normals=directions,
            )
            elapsed = time.monotonic() - started
            candidate = answer if answer["witness_found"] else answer.get("closest_linear_feasible_attempt")
            saved = None
            if candidate is not None:
                commands = candidate["planned_commands"] if answer["witness_found"] else candidate["commands"]
                saved = candidate_record(commands, directions, candidate["all_tcp_errors"])
                if answer["witness_found"]:
                    error = saved["all_tcp_errors"][0]
                    require(
                        np.linalg.norm(error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(error[3:]) <= 0.05 + 1e-10,
                        "First endpoint original pose caps",
                    )
            result["entries"].append(dict(mode=mode, compute_seconds=elapsed, answer=answer, candidate=saved))
        for ref in sources:
            checked(ref)
        require(float(data.time) == state[0], "No physics time advance")
        result.update(
            complete=True,
            passed=True,
            scope="Saved H8 active-constraint audit and two numerical H1 policies; no control/physical/task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
