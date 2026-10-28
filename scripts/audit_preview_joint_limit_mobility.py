"""Saved-state active limit geometry and ideal constrained robot mobility."""

import json

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/preview_joint_limit_mobility_20261027"
    root.mkdir(exist_ok=False)
    paired_ref = reference(REPO / "outputs/preview_chunk_diagnostic_verified_20261027/verified.json")
    paired = read(paired_ref)
    require(paired["passed"] is True, "Paired branch audit required")
    prepared = read(paired["prepared_ref"])
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment
    from s2a.native_training import state_sha256

    cases = []
    for i, case in enumerate(prepared["cases"]):
        manifest = read(case["manifest_ref"])
        rows_ref = reference(REPO / f"outputs/preview_chunk_branches_20261027/case{i:02d}_h40/steps.json")
        env = None
        records = []
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{i:02d}", manifest["record"], 0)
            import mujoco

            model, data = env.sim.model._model, env.sim.data._data
            dofs = np.asarray(env.robots[0]._ref_joint_vel_indexes)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, env.robots[0].controller.eef_name)
            for j, row in enumerate(read(rows_ref)):
                state = np.asarray(row["sim_state"])
                require(state_sha256(state) == row["state_sha256"], "Saved state hash")
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                np.testing.assert_allclose(env.sim.get_state().flatten(), state, atol=1e-14, rtol=0)
                mass = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, mass, data.qM)
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                jacobian = np.vstack((jp, jr))[:, dofs]
                robot_mass = mass[np.ix_(dofs, dofs)]
                inverse = np.linalg.solve(robot_mass, np.eye(len(dofs)))
                normal_rows, limits = [], []
                for k in range(data.nefc):
                    if int(data.efc_type[k]) != int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
                        continue
                    jid = int(data.efc_id[k])
                    dof = int(model.jnt_dofadr[jid])
                    if dof not in dofs:
                        continue
                    normal = np.zeros(len(dofs))
                    normal[int(np.flatnonzero(dofs == dof)[0])] = 1
                    normal_rows.append(normal)
                    limits.append(
                        dict(
                            joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
                            position_rad=float(data.qpos[model.jnt_qposadr[jid]]),
                            range_rad=model.jnt_range[jid].tolist(),
                            constraint_position=float(data.efc_pos[k]),
                        )
                    )
                constrained = inverse.copy()
                if normal_rows:
                    normals = np.asarray(normal_rows)
                    constrained -= (
                        inverse @ normals.T @ np.linalg.pinv(normals @ inverse @ normals.T) @ normals @ inverse
                    )
                    require(np.max(np.abs(normals @ constrained)) < 1e-9, "Projection does not fix active normals")
                free_task = jacobian @ inverse @ jacobian.T
                limit_task = jacobian @ constrained @ jacobian.T
                records.append(
                    dict(
                        index=j,
                        active_limit_geometry=limits,
                        free_task_inverse_inertia=free_task.tolist(),
                        ideal_fixed_limit_task_inverse_inertia=limit_task.tolist(),
                        relative_mobility_change=float(
                            np.linalg.norm(limit_task - free_task) / np.linalg.norm(free_task)
                        ),
                        free_condition_number=float(np.linalg.cond(free_task)),
                        fixed_limit_condition_number=float(np.linalg.cond(limit_task)),
                    )
                )
            cases.append(dict(case=case["case"], rows_ref=rows_ref, endpoints=records))
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
                    paired_ref=paired_ref,
                    source_ref=reference(__file__),
                    cases=cases,
                    physics_steps=0,
                    model_calls=0,
                    fresh_cases=0,
                    scope="recomputed limit geometry and ideal fixed-normal mobility; no original force or dynamic rollout evidence",
                    unilateral_soft_limit_is_not_bilateral_fixed_constraint=True,
                    contact_constraints_omitted=True,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
