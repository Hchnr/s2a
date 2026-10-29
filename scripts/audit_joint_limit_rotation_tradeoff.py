"""Compare saved executed refinement and seed using the same exact TCP SO(3) loss."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error, transform
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/joint_limit_rotation_tradeoff_20261027"
    root.mkdir(exist_ok=False)
    verified_ref = reference(REPO / "outputs/joint_limit_candidate_verified_20261027/verified.json")
    verified = read(verified_ref)
    require(verified["passed"] is True, "Physical comparison required")
    owner = read(verified["owner_ref"])
    registration = read(owner["registration_ref"])
    for path, digest in registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    prep_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    case = read(prep_ref)["cases"][0]
    refs = {Path(r["path"]).name: r for r in case["original_refs"]}
    worker = read(refs["worker_result.json"])
    with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
        first_state = archive["sim_states"][case["replay_periods"]].copy()
    chunks = read(refs["chunks.json"])
    generated = np.asarray(chunks[case["event"]["query_index"]]["effective_targets_tcp"])
    h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
    control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
    site_from_tcp = np.linalg.inv(control["source_from_site"]) @ h
    manifest = read(case["manifest_ref"])
    rows_ref = verified["cases"][0]["rows_ref"]
    rows = read(rows_ref)
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment

    env = None
    records = []
    try:
        env, _ = make_environment(manifest["runtime_config"], root, manifest["record"], 0)
        import mujoco

        robot, controller = env.robots[0], env.robots[0].controller
        model, data = env.sim.model._model, env.sim.data._data
        dofs, qids = np.asarray(robot._ref_joint_vel_indexes), np.asarray(robot._ref_joint_pos_indexes)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}") for k in range(1, 8)]
        limits = model.jnt_range[joints]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
        interface = OSCInterface(
            *(control["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max"))
        )
        for j, row in enumerate(rows):
            env.sim.set_state_from_flattened(first_state if j == 0 else np.asarray(rows[j - 1]["sim_state"]))
            env.sim.forward()
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
            local = build_joint_response(
                mass[np.ix_(dofs, dofs)],
                np.vstack((jp, jr))[:, dofs],
                controller.kp,
                controller.kd,
                data.qpos[qids],
                controller.initial_joint,
                null_kp=10.0,
                dt=0.002,
            )
            bias = np.asarray(row["refinement"]["history_before"])
            predictions, errors = [], []
            for command in (row["seed_command"], row["command"][:6]):
                predicted = predict_joint_response(
                    local,
                    data.qvel[dofs],
                    interface.decode(np.asarray(command)),
                    limits[:, 0],
                    limits[:, 1],
                    substeps=25,
                    enforce_limits=True,
                )
                response = np.r_[predicted["task_displacement"], predicted["task_velocity"]]
                predicted_site = transform(
                    site[:3, 3] + response[:3] + bias[:3],
                    Rotation.from_rotvec(bias[3:6]).as_matrix()
                    @ Rotation.from_rotvec(response[3:6]).as_matrix()
                    @ site[:3, :3],
                )
                errors.append(pose_error(generated[j], predicted_site @ site_from_tcp))
                predictions.append(response)
            np.testing.assert_allclose(predictions[1], row["refinement"]["nominal_response"], atol=1e-8, rtol=0)
            seed, chosen = errors
            predicted_site_rotation = Rotation.from_rotvec(predictions[1][3:6]).as_matrix() @ site[:3, :3]
            desired_site = generated[j] @ np.linalg.inv(site_from_tcp)
            exact_site_error = Rotation.from_matrix(
                desired_site[:3, :3] @ (Rotation.from_rotvec(bias[3:6]).as_matrix() @ predicted_site_rotation).T
            ).as_rotvec()
            additive = pose_error(desired_site, site)[3:] - bias[3:6] - predictions[1][3:6]
            records.append(
                dict(
                    index=j,
                    seed_tcp_error=seed.tolist(),
                    chosen_tcp_error=chosen.tolist(),
                    rotation_worsens_vs_seed=bool(np.linalg.norm(chosen[3:]) > np.linalg.norm(seed[3:]) + 1e-8),
                    position_worsens_vs_seed=bool(np.linalg.norm(chosen[:3]) > np.linalg.norm(seed[:3]) + 1e-8),
                    exact_vs_additive_rotation_error_difference_rad=float(np.linalg.norm(exact_site_error - additive)),
                    actual_rotation_tracking_error_rad=float(np.linalg.norm(row["tcp_error"][3:])),
                    corrected_predicted_rotation_tracking_error_rad=float(np.linalg.norm(chosen[3:])),
                )
            )
    finally:
        if env is not None:
            env.close()
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    verified_ref=verified_ref,
                    rows_ref=rows_ref,
                    prepared_ref=prep_ref,
                    source_ref=reference(__file__),
                    periods=len(records),
                    records=records,
                    rotation_worsening_periods=sum(r["rotation_worsens_vs_seed"] for r in records),
                    position_worsening_periods=sum(r["position_worsens_vs_seed"] for r in records),
                    scope="same-state seed/proposal model diagnostics; seed counterfactual not physically executed at these changed states",
                    physics_steps=0,
                    model_calls=0,
                    fresh_cases=0,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
