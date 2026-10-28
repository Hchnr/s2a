"""Fixed reference pose tubes: nearest sequential bounded configurations, geometry only."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.bounded_pose_tube import closest_pose_in_tube
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.full_coupled_pose_inverse import hat, right_inverse_jacobian
from s2a.geometry import pose_error, transform
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/preview_pose_tube_20261027"
    root.mkdir(exist_ok=False)
    prepared_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    prepared = read(prepared_ref)
    original_ref = reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json")
    sources = dict(read(original_ref)["source_sha256"])
    for path in (Path(__file__), REPO / "src/s2a/bounded_pose_ik.py", REPO / "src/s2a/bounded_pose_tube.py"):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    registration_ref = publish(
        root / "registration.json",
        dict(
            prepared_ref=prepared_ref,
            source_sha256=sources,
            case_indices=[0, 1],
            branches=["sequential_current", "sequential_controller_initial"],
            targets_per_branch=40,
            total_tube_queries=160,
            cold_ik_max_nfev=200,
            max_slsqp_iterations=60,
            random_seed=42,
            position_tube_m=0.01,
            rotation_tube_rad=0.05,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="closest bounded pose tube construction; no collision-free or timed controller guarantee",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment

    cases = []
    for index in (0, 1):
        case = prepared["cases"][index]
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        worker = read(refs["worker_result.json"])
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
            current_state = archive["sim_states"][case["replay_periods"]].copy()
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        targets = np.asarray(chunk["effective_targets_tcp"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        site_from_tcp = np.linalg.inv(control["source_from_site"]) @ h
        manifest = read(case["manifest_ref"])
        env = None
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            import mujoco

            model, data = env.sim.model._model, env.sim.data._data
            robot = env.robots[0]
            env.sim.set_state_from_flattened(current_state)
            env.sim.forward()
            dofs, qids = np.asarray(robot._ref_joint_vel_indexes), np.asarray(robot._ref_joint_pos_indexes)
            joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
            np.testing.assert_array_equal(model.jnt_qposadr[joints], qids)
            limits = model.jnt_range[joints].copy()
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.controller.eef_name)
            current, initial = data.qpos[qids].copy(), robot.controller.initial_joint.copy()
            frozen_time, frozen_velocity = float(data.time), data.qvel.copy()

            def fk(q):
                data.qpos[qids] = q
                env.sim.forward()
                site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                offset = site[:3, :3] @ site_from_tcp[:3, 3]
                jacobian = np.vstack((jp[:, dofs] - hat(offset) @ jr[:, dofs], jr[:, dofs]))
                return site @ site_from_tcp, jacobian

            probe = np.clip(current, limits[:, 0], limits[:, 1])
            pose, jacobian = fk(probe)
            error = pose_error(targets[0], pose)
            derivative = np.vstack((-jacobian[:3], -right_inverse_jacobian(error[3:]) @ jacobian[3:]))
            max_jacobian_error = 0.0
            for column in range(7):
                plus, minus = probe.copy(), probe.copy()
                plus[column] += 1e-6
                minus[column] -= 1e-6
                numeric = (pose_error(targets[0], fk(plus)[0]) - pose_error(targets[0], fk(minus)[0])) / 2e-6
                max_jacobian_error = max(max_jacobian_error, float(np.max(abs(numeric - derivative[:, column]))))
            require(max_jacobian_error < 1e-5, "Native TCP residual Jacobian mismatch")
            branches = []
            for name, start in (("sequential_current", current), ("sequential_controller_initial", initial)):
                q = start.copy()
                rows = []
                for j, target in enumerate(targets):
                    try:
                        solved = closest_pose_in_tube(fk, target, q, limits[:, 0], limits[:, 1], max_nfev=200)
                        actual_pose, _ = fk(solved["joints"])
                        np.testing.assert_allclose(pose_error(target, actual_pose), solved["error"], atol=1e-9, rtol=0)
                        contacts = []
                        for k in range(data.ncon):
                            contact = data.contact[k]
                            names = [
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{g}"
                                for g in (contact.geom1, contact.geom2)
                            ]
                            if any("robot0_" in n or "gripper0_" in n for n in names):
                                contacts.append(dict(geoms=names, distance_m=float(contact.dist)))
                        row = {
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in solved.items()
                        }
                        row.update(
                            index=j,
                            diagnostic_joint_change_rad=(solved["joints"] - q).tolist(),
                            static_current_scene_contacts=contacts,
                        )
                        q = solved["joints"].copy()
                    except Exception as error:
                        row = dict(index=j, within_tube=False, error_message=repr(error), no_retry=True)
                    rows.append(row)
                joint_path = np.asarray([r["joints"] for r in rows if "joints" in r])
                differences = [
                    np.asarray(rows[j]["joints"]) - np.asarray(rows[j - 1]["joints"])
                    for j in range(1, len(rows))
                    if "joints" in rows[j] and "joints" in rows[j - 1]
                ]
                max_difference = None if not differences else float(np.max(abs(np.asarray(differences))))
                branches.append(
                    dict(
                        name=name,
                        start_joints=start.tolist(),
                        rows=rows,
                        tube_configurations=sum(r["within_tube"] for r in rows),
                        max_abs_path_difference_rad=max_difference,
                        max_abs_path_difference_over_0p05_rad_s=None
                        if max_difference is None
                        else max_difference / 0.05,
                        first_solution_index=next((r["index"] for r in rows if "joints" in r), None),
                        first_solution_distance_from_actual_current_rad=None
                        if not len(joint_path)
                        else float(np.max(abs(joint_path[0] - current))),
                    )
                )
            require(float(data.time) == frozen_time, "FK audit advanced time")
            np.testing.assert_array_equal(data.qvel, frozen_velocity)
            cases.append(
                dict(
                    case=case["case"],
                    current_joints=current.tolist(),
                    joint_limits=limits.tolist(),
                    jacobian_max_abs_error=max_jacobian_error,
                    branches=branches,
                    contacts_scope="static current object poses and current gripper opening; not future collision acceptance",
                )
            )
        finally:
            if env is not None:
                env.close()
        print(json.dumps(dict(completed_case=index)), flush=True)
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    registration_ref=registration_ref,
                    cases=cases,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    control_qualified=False,
                    scope="pose tube construction and joint path descriptors; no controller or velocity feasibility claim",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
