"""Measure frozen joint prediction errors against independently replayed inner states."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse
from s2a.interior_joint_preview import build_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.quality_capped_inner_joint_plan import solve_interior_plan
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_reference_resume_numeric_20261028"
    require(not root.exists(), "No overwrite/retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_check_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed independent full-inner replay")
    check_ref = owner["result_ref"]
    require(read(check_ref)["complete"] is read(check_ref)["passed"] is True, "Verified all-inner states")
    producer_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_20261028/result.json")
    producer = read(producer_ref)
    reg = read(producer["registration_ref"])
    worker = read(reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    control_ref = reference(folder / "controller.json")
    contract = read(control_ref)["contract"]
    numeric_ref = reference(REPO / "outputs/tracking_v5_velocity_braking_numeric_20261028/result.json")
    initial_joint = np.asarray(read(numeric_ref)["initial_joint"])
    sources = {
        ref["path"]: ref["sha256"]
        for ref in [
            reference(Path(__file__)),
            *[
                reference(REPO / "src/s2a" / name)
                for name in (
                    "interior_joint_preview.py",
                    "joint_limit_response.py",
                    "quality_capped_inner_joint_plan.py",
                )
            ],
        ]
    }
    registration = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            independent_owner_ref=owner_ref,
            independent_ref=check_ref,
            producer_ref=producer_ref,
            numeric_ref=numeric_ref,
            control_ref=control_ref,
            xml_ref=reference(folder / "model.xml"),
            seed=42,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            rendering_policy="XML-only CPU mj_forward; no renderer/context; OSMesa forcing unset before MuJoCo import",
            envelope="Per-joint running maximum of earlier completed-period absolute free-model residual; first period has no bound",
            scope="All actually executed periods; independent native-state causal-history/plan audit; mj_forward only, no task qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    # XML-only CPU dynamics: no OpenGL context or renderer is constructed.
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
    qids, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limits = model.jnt_range[joints]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    require(sid >= 0 and all(i >= 0 for i in joints), "Actual robot joints/SITE")
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        branches=[],
    )
    try:
        hold_check_owner_ref = reference(
            REPO / "outputs/tracking_v5_reference_hold_native_check_execution_20261028/result.json"
        )
        check_owner = read(hold_check_owner_ref)
        require(
            check_owner["complete"] is check_owner["passed"] is True, "Closed independent held-state physical replay"
        )
        hold_ref = reference(REPO / "outputs/tracking_v5_reference_hold_native_20261028/result.json")
        hold = read(hold_ref)
        held = next(v for v in hold["branches"] if v["branch"] == "held")
        history_ref = reference(REPO / "outputs/tracking_v5_causal_inner_history_audit_20261028/verified.json")
        history = read(history_ref)
        require(history["complete"] is history["passed"] is True, "Independent prior causal history")
        hist_branch = next(v for v in history["branches"] if v["branch"] == "causal_error")
        records = read(hist_branch["periods_ref"])
        prior = next(v for v in records if v["step"] == 376)
        before = np.asarray(prior["causal_inner_profile_after"])
        hold_numeric_ref = reference(REPO / "outputs/tracking_v5_reference_hold_numeric_20261028/verified.json")
        searches = read(
            next(v for v in read(hold_numeric_ref)["branches"] if v["branch"] == "causal_error")["searches_ref"]
        )
        prediction = next(v for v in searches if v["step"] == 377)["held_quality"]
        commands = np.asarray(read(held["commands_ref"]))
        previous = commands[-1, :6]
        np.testing.assert_array_equal(previous, np.asarray(prediction["planned_commands"])[0])
        inner = read(held["inner_ref"])
        actual = np.asarray([v["q"] for v in inner])
        predicted = np.asarray(prediction["rollout"]["inner_joint_position"])[:25]
        require(actual.shape == predicted.shape == (25, 7), "One actual completed held period")
        residual = abs(actual - predicted)
        profile = np.maximum.accumulate(np.maximum(before, residual), axis=0)
        update_ref = publish(
            root / "history_update.json",
            dict(
                prefix_completed_periods=9,
                held_completed_periods=1,
                last_completed_step=377,
                before_profile=before,
                held_actual_inner=actual,
                held_predicted_inner=predicted,
                held_residual=residual,
                after_profile=profile,
                future_guarantee=False,
            ),
        )
        with np.load(checked(held["states_ref"]), allow_pickle=False) as archive:
            states = archive["sim_states"].copy()
        require(len(states) == 379, "Actual held endpoint378")
        state = states[-1]
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.qpos[qids], actual[-1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(data.qvel[dofs], inner[-1]["qvel"], atol=1e-12, rtol=0)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
        source_branch = next(v for v in producer["branches"] if v["branch"] == "causal_error")
        d = read(source_branch["steps_ref"])[9]["dynamics"]
        local = build_joint_preview(
            mass, jacobian, d["kp"], d["kd"], data.qpos[qids], initial_joint, null_kp=10.0, dt=0.002
        )
        site = np.eye(4)
        site[:3, :3] = data.site_xmat[sid].reshape(3, 3)
        site[:3, 3] = data.site_xpos[sid]
        control = read(control_ref)
        manifest = read(reg["manifest_ref"])
        tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
        tool = inverse(np.asarray(control["source_from_site"])) @ np.asarray(read(tool_ref)["source_from_tcp"])
        chunk = read(reg["chunks_ref"])[46]
        poses, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
        seed = np.tile(previous, (8, 1))
        old_plan = np.asarray(prediction["planned_commands"])
        warm = np.vstack((old_plan[1:], old_plan[-1]))
        outcomes = []
        for mode in ("hold8", "resume9"):
            targets = np.tile(poses[8], (8, 1, 1)) if mode == "hold8" else poses[9:17]
            twists = (
                np.zeros((8, 6))
                if mode == "hold8"
                else np.asarray([reference_world_twist(poses, k, anchor, 0.05)["twist"] for k in range(9, 17)])
            )
            answer = solve_interior_plan(
                local,
                data.qvel[dofs],
                seed,
                previous,
                interface,
                site,
                tool,
                targets,
                twists,
                limits[:, 0],
                limits[:, 1],
                initial_guess=warm,
                past_inner_error_rad=profile,
            )
            if answer["witness_found"]:
                require(
                    np.linalg.norm(answer["first_tcp_error"][:3]) <= 0.01 + 1e-10
                    and np.linalg.norm(answer["first_tcp_error"][3:]) <= 0.05 + 1e-10,
                    "Original absolute accuracy retained",
                )
                require(answer["minimum_tightened_inner_slack_rad"] >= -1e-10, "All tightened inner points")
            outcomes.append(dict(mode=mode, first_target=targets[0], answer=answer))
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(
            complete=True,
            passed=True,
            hold_ref=hold_ref,
            hold_check_owner_ref=hold_check_owner_ref,
            history_ref=history_ref,
            hold_numeric_ref=hold_numeric_ref,
            held_states_ref=held["states_ref"],
            history_update_ref=update_ref,
            endpoint=378,
            previous_command=previous,
            joint_position=data.qpos[qids].copy(),
            joint_velocity=data.qvel[dofs].copy(),
            outcomes=outcomes,
            scope="Same actual post-hold state, own actual updated causal history; held or original next reference quality search; no physics, original-timing/task qualification or proof of global nonlinear infeasibility",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
