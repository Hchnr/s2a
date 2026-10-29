"""Compare first/full FK pose caps at independently replayed actual endpoint370."""

import json
import os
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.fk_pose_cap_scope_joint_plan import solve_interior_plan
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse, pose_error, transform
from s2a.interior_joint_preview import build_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.robot_site_kinematics import RobotSiteKinematics
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_fk_cap_scope_numeric_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    check_owner_ref = reference(REPO / "outputs/tracking_v5_fk_feedback_native_check_execution_20261028/result.json")
    owner = read(check_owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed independent physical replay")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0/reap0",
    )
    native_ref = reference(REPO / "outputs/tracking_v5_fk_feedback_native_20261028/verified.json")
    native = read(native_ref)
    require(
        native["complete"] is native["passed"] is True
        and native["continuation_steps"] == 2
        and native["stopped"]["step"] == 370
        and native["stopped"]["action_dispatched"] is False,
        "Exact stopped endpoint370",
    )
    nreg = read(native["registration_ref"])
    numeric = read(nreg["numeric_ref"])
    inputs = read(numeric["registration_ref"])
    original = read(inputs["producer_ref"])
    oreg = read(original["registration_ref"])
    initial = np.asarray(read(oreg["braking_numeric_ref"])["initial_joint"])
    control = read(inputs["control_ref"])
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(
        read(inputs["tool_ref"])["source_from_tcp"]
    )
    rows = read(native["periods_ref"])
    require(rows[-1]["step"] == 369, "Only this branch completed368/369")
    previous = np.asarray(rows[-1]["command"])
    profile = np.asarray(rows[-1]["history_after"])
    np.testing.assert_array_equal(profile, native["stopped"]["answer"]["past_inner_error_rad"])
    old_plan = np.asarray(rows[-1]["answer"]["planned_commands"])
    warm = np.vstack((old_plan[1:], old_plan[-1]))
    with np.load(checked(native["states_ref"]), allow_pickle=False) as archive:
        require(len(archive["sim_states"]) == 371, "Every actual endpoint through370")
        state = archive["sim_states"][-1].copy()
    chunk = read(nreg["chunks_ref"])[46]
    targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
    sources = [
        reference(Path(__file__)),
        *[
            reference(REPO / "src/s2a" / name)
            for name in (
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
            independent_owner_ref=check_owner_ref,
            native_ref=native_ref,
            current_states_ref=native["states_ref"],
            xml_ref=inputs["xml_ref"],
            control_ref=inputs["control_ref"],
            tool_ref=inputs["tool_ref"],
            source_refs=sources,
            seed=42,
            endpoint=370,
            cursor=2,
            horizon=8,
            modes=["all_endpoints", "first_only"],
            per_search_response_budget=160,
            total_response_budget=320,
            per_search_iteration_budget=60,
            pose_caps=[0.01, 0.05],
            actual_joint_margin_rad=0.05,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Matched current state/history and unchanged full future pose/twist objective; compare extra future precision conditions only; no actual threshold relaxation",
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
    seed = np.tile(previous[:6], (8, 1))
    desired = targets[2:10]
    twists = np.asarray([reference_world_twist(targets, k, anchor, 0.05)["twist"] for k in range(2, 10)])
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        entries=[],
    )
    try:
        for mode, cap_count in (("all_endpoints", None), ("first_only", 1)):
            fk = RobotSiteKinematics(
                checked(inputs["xml_ref"]).read_text(),
                contract["controller_site"],
                [f"robot0_joint{k}" for k in range(1, 8)],
            )
            started = time.monotonic()
            answer = solve_interior_plan(
                local,
                velocity,
                seed,
                previous[:6],
                interface,
                site,
                site_from_tcp,
                desired,
                twists,
                limits[:, 0],
                limits[:, 1],
                fk_jacobian=fk,
                pose_cap_periods=cap_count,
                initial_guess=warm,
                past_inner_error_rad=profile,
            )
            elapsed = time.monotonic() - started
            checks = []
            if answer["witness_found"]:
                for period in range(8):
                    data.qpos[qids] = answer["rollout"]["joint_position"][period]
                    data.qvel[dofs] = answer["rollout"]["joint_velocity"][period]
                    mujoco.mj_forward(model, data)
                    pose = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                    error = pose_error(desired[period], pose @ site_from_tcp)
                    np.testing.assert_allclose(pose, answer["rollout"]["pose"][period], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(error, answer["all_tcp_errors"][period], atol=1e-10, rtol=0)
                    okay = bool(np.linalg.norm(error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(error[3:]) <= 0.05 + 1e-10)
                    require(okay or (mode == "first_only" and period > 0), "Declared pose cap scope retained")
                    checks.append(dict(period=period, native_tcp_error=error, native_caps_satisfied=okay))
            result["entries"].append(dict(mode=mode, answer=answer, compute_seconds=elapsed, native_checks=checks))
        for ref in sources:
            checked(ref)
        result.update(
            complete=True,
            passed=True,
            current_q=q,
            current_velocity=velocity,
            current_site=site,
            current_mass=mass,
            current_jacobian=jac,
            initial_joint=initial,
            kp=fixed["kp"],
            kd=fixed["kd"],
            history_before=profile,
            previous_command=previous,
            warm_start=warm,
            scope="Only same actual370 conditional model search and independent FK; no new physical action or task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
