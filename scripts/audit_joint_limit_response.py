"""Fixed saved-state one-period prediction comparison, with optional all-case audit."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error, transform
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main(*, all_cases=False):
    strict_native_environment()
    name = "joint_limit_response_all_20261027" if all_cases else "joint_limit_response_audit_20261027"
    root = REPO / "outputs" / name
    root.mkdir(exist_ok=False)
    prep_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    prepared = read(prep_ref)
    indices = list(range(7)) if all_cases else [0, 1]
    publish(
        root / "registration.json",
        dict(
            cases=indices,
            predictions_per_case=40,
            prepared_ref=prep_ref,
            source_refs=[reference(Path(__file__)), reference(REPO / "src/s2a/joint_limit_response.py")],
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
            scope="one-period saved-state model diagnosis, not control qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment

    cases = []
    for i in indices:
        case = prepared["cases"][i]
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        worker = read(refs["worker_result.json"])
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
            first_state = archive["sim_states"][case["replay_periods"]].copy()
        manifest = read(case["manifest_ref"])
        rows_ref = reference(REPO / f"outputs/preview_chunk_branches_20261027/case{i:02d}_h40/steps.json")
        rows = read(rows_ref)
        require(len(rows) == 40, "Complete saved40-step branch required")
        env = None
        records = []
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{i:02d}", manifest["record"], 0)
            import mujoco

            model, data = env.sim.model._model, env.sim.data._data
            robot, controller = env.robots[0], env.robots[0].controller
            dofs, qids = np.asarray(robot._ref_joint_vel_indexes), np.asarray(robot._ref_joint_pos_indexes)
            joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
            limits = model.jnt_range[joints].copy()
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
            interface = OSCInterface(
                *(getattr(controller, key) for key in ("input_min", "input_max", "output_min", "output_max"))
            )
            for j, row in enumerate(rows):
                before_state = first_state if j == 0 else np.asarray(rows[j - 1]["sim_state"])
                env.sim.set_state_from_flattened(before_state)
                env.sim.forward()
                mass = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, mass, data.qM)
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                before = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
                local = build_joint_response(
                    mass[np.ix_(dofs, dofs)],
                    np.vstack((jp, jr))[:, dofs],
                    controller.kp,
                    controller.kd,
                    q,
                    controller.initial_joint,
                    null_kp=10.0,
                    dt=0.002,
                )
                delta = interface.decode(np.asarray(row["command"])[:6])
                predictions = [
                    predict_joint_response(
                        local, velocity, delta, limits[:, 0], limits[:, 1], substeps=25, enforce_limits=enabled
                    )
                    for enabled in (False, True)
                ]
                # Future endpoint is loaded only after both current-state predictions.
                env.sim.set_state_from_flattened(np.asarray(row["sim_state"]))
                env.sim.forward()
                after = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                actual = pose_error(after, before)
                error = [np.asarray(p["task_displacement"]) - actual for p in predictions]
                records.append(
                    dict(
                        index=j,
                        free_error=error[0].tolist(),
                        limited_error=error[1].tolist(),
                        projected_steps=predictions[1]["position_projected_steps"],
                        initial_penetration_rad=predictions[1]["initial_max_penetration_rad"],
                    )
                )
            metrics = {}
            for method in ("free", "limited"):
                errors = np.asarray([r[f"{method}_error"] for r in records])
                metrics[method] = dict(
                    position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                    rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                )
            cases.append(dict(case=case["case"], rows_ref=rows_ref, metrics=metrics, records=records))
        finally:
            if env is not None:
                env.close()
        print(json.dumps(dict(completed_case=i)), flush=True)
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    cases=cases,
                    prepared_ref=prep_ref,
                    scope="model computation completed; performance and control qualification not implied",
                    physics_steps=0,
                    model_calls=0,
                    fresh_cases=0,
                ),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Audit all seven preselected saved branches")
    main(all_cases=parser.parse_args().all)
