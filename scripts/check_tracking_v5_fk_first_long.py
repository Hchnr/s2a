"""MuJoCo geometry cross-check of fixed FK-consistent preview candidates; no search."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse, pose_error, transform
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.robot_site_kinematics import RobotSiteKinematics
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_fk_first_long_check_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_fk_first_long_numeric_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed fixed-horizon producer")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0/reap0",
    )
    producer_ref = owner["result_ref"]
    producer = read(producer_ref)
    reg = read(producer["registration_ref"])
    original = read(reg["producer_ref"])
    original_reg = read(original["registration_ref"])
    branch = next(v for v in original["branches"] if v["branch"] == "causal_error")
    d = read(branch["steps_ref"])[0]["dynamics"]
    limits = np.asarray(read(branch["inner_ref"])["joint_limits"])
    initial = np.asarray(read(original_reg["braking_numeric_ref"])["initial_joint"])
    control = read(reg["control_ref"])
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(
        read(reg["tool_ref"])["source_from_tcp"]
    )
    with np.load(checked(reg["states_ref"]), allow_pickle=False) as archive:
        state = archive["sim_states"][368].copy()
    targets = np.asarray(read(reg["chunks_ref"])[46]["effective_targets_tcp"])
    robot_fk = RobotSiteKinematics(
        checked(reg["xml_ref"]).read_text(), contract["controller_site"], [f"robot0_joint{k}" for k in range(1, 8)]
    )
    registration = publish(
        root / "registration.json",
        dict(
            producer_owner_ref=owner_ref,
            producer_ref=producer_ref,
            xml_ref=reg["xml_ref"],
            source_ref=reference(Path(__file__)),
            seed=42,
            horizons=[8, 12, 16],
            selection="Accepted witness else saved closest checked linear-feasible attempt, one per horizon; no search/restart",
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="Independent MuJoCo FK/J and geometric target-error audit of predicted q; not executed actions or physical control qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(checked(reg["xml_ref"])))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
    require(all(v >= 0 for v in joints), "Native robot joints")
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    require(sid >= 0, "Native controller SITE")
    data.time = state[0]
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq :]
    np.testing.assert_array_equal(limits, model.jnt_range[joints])
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
        for entry in producer["entries"]:
            horizon = entry["horizon"]
            answer = entry["answer"]
            require(
                answer["pose_caps_horizon"] == 1 and answer["all_pose_caps_enforced"] is False,
                "Explicit first-period predicted pose caps",
            )
            candidate = answer if answer["witness_found"] else answer["closest_linear_feasible_attempt"]
            if candidate is None:
                require(answer["witness_found"] is False, "No candidate cannot be a witness")
                result["entries"].append(dict(horizon=horizon, model_witness=False, candidate_available=False, rows=[]))
                continue
            commands = np.asarray(candidate["planned_commands"] if answer["witness_found"] else candidate["commands"])
            require(commands.shape == (horizon, 6) and np.isfinite(commands).all(), "Complete finite candidate")
            require(
                np.all(commands >= interface.input_min - 1e-12) and np.all(commands <= interface.input_max + 1e-12),
                "Original command box",
            )
            require(
                np.max(abs(np.diff(np.vstack((np.asarray(entry["previous_command"])[:6], commands)), axis=0)))
                <= 0.25 + 1e-12,
                "Original command rate",
            )
            local = build_joint_preview(
                d["mass"], d["jacobian"], d["kp"], d["kd"], entry["current_q"], initial, null_kp=10.0, dt=0.002
            )
            raw = rollout_joint_preview(
                local,
                entry["current_velocity"],
                np.asarray([interface.decode(v) for v in commands]),
                entry["current_site"],
                limits[:, 0],
                limits[:, 1],
                enforce_limits=False,
            )
            rows = []
            for period in range(horizon):
                q, velocity = raw["joint_position"][period], raw["joint_velocity"][period]
                data.qpos[qids] = q
                data.qvel[dofs] = velocity
                mujoco.mj_forward(model, data)
                site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                jac = np.vstack((jp, jr))[:, dofs]
                xml_pose, xml_jac = robot_fk(q)
                np.testing.assert_allclose(xml_pose, site, atol=1e-10, rtol=0)
                np.testing.assert_allclose(xml_jac, jac, atol=1e-10, rtol=0)
                error = pose_error(targets[period], site @ site_from_tcp)
                np.testing.assert_allclose(error, candidate["all_tcp_errors"][period], atol=1e-10, rtol=0)
                if answer["witness_found"]:
                    np.testing.assert_allclose(site, answer["rollout"]["pose"][period], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(jac @ velocity, answer["rollout"]["twist"][period], atol=1e-10, rtol=0)
                caps_satisfied = bool(
                    np.linalg.norm(error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(error[3:]) <= 0.05 + 1e-10
                )
                require(
                    not answer["witness_found"] or period > 0 or caps_satisfied,
                    "Accepted first-period witness retains native geometric accuracy",
                )
                rows.append(
                    dict(
                        native_caps_satisfied=caps_satisfied,
                        period=period,
                        native_site=site,
                        native_jacobian=jac,
                        native_tcp_error=error,
                        pose_difference=float(np.max(abs(xml_pose - site))),
                        jacobian_difference=float(np.max(abs(xml_jac - jac))),
                    )
                )
            margin = np.asarray(answer["requested_position_margin_rad"])
            q = raw["inner_joint_position"]
            slack = float(np.minimum(q - limits[:, 0] - margin, limits[:, 1] - margin - q).min())
            require(slack >= -1e-10, "Recorded candidate satisfies original tightened inner bounds")
            braking = raw["joint_position"] + 0.1 * raw["joint_velocity"]
            braking_slack = float(np.minimum(braking - limits[:, 0] - 0.05, limits[:, 1] - 0.05 - braking).min())
            require(braking_slack >= -1e-10, "Original braking constraints")
            require(
                np.all(np.asarray(answer["terminal_normals"]) @ raw["joint_velocity"][-1] <= 1e-10),
                "Original terminal constraints",
            )
            data.qvel[:] = state[1 + model.nq :]
            require(float(data.time) == state[0], "No time advance")
            result["entries"].append(
                dict(
                    horizon=horizon,
                    model_witness=answer["witness_found"],
                    rows=rows,
                    minimum_tightened_inner_slack_rad=slack,
                    minimum_braking_slack_rad=braking_slack,
                    native_caps_satisfied_count=sum(v["native_caps_satisfied"] for v in rows),
                    maximum_pose_difference=max(v["pose_difference"] for v in rows),
                    maximum_jacobian_difference=max(v["jacobian_difference"] for v in rows),
                )
            )
        result.update(
            complete=True,
            passed=True,
            scope="MuJoCo/XML geometry agreement only; frozen joint dynamics and actual action response remain unqualified",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
