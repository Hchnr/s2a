"""Joint whole-reference pose-tube planning from two fixed complete seed paths."""

import json
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.bounded_joint_path import plan_joint_path
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.full_coupled_pose_inverse import hat
from s2a.geometry import transform
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/preview_joint_path_20261027"
    root.mkdir(exist_ok=False)
    seed_verified_ref = reference(REPO / "outputs/preview_pose_tube_verified_20261027/verified.json")
    seed_verified = read(seed_verified_ref)
    require(seed_verified["complete"] is seed_verified["passed"] is True, "Independent seed-path FK check required")
    seeds_ref = seed_verified["producer_ref"]
    seeds = read(seeds_ref)
    prepared_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    prepared = read(prepared_ref)
    old_ref = reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json")
    sources = dict(read(old_ref)["source_sha256"])
    for path in (Path(__file__), REPO / "scripts/verify_preview_joint_path.py", REPO / "src/s2a/bounded_joint_path.py"):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    registration_ref = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            seed_verified_ref=seed_verified_ref,
            seed_producer_ref=seeds_ref,
            prepared_ref=prepared_ref,
            cases=[0, 1],
            complete_seed_paths_per_case=2,
            horizon=40,
            position_tube_m=0.01,
            rotation_tube_rad=0.05,
            max_iterations=80,
            max_path_evaluations=160,
            interpolation_samples=[0.25, 0.5, 0.75],
            seed=42,
            actual_anchor_fixed=True,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="joint geometric path optimization; no dynamic or fixed-controller qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment

    cases = []
    for index in (0, 1):
        case = prepared["cases"][index]
        require(case["case"] == seeds["cases"][index]["case"], "Seed case identity differs")
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        worker = read(refs["worker_result.json"])
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
            snapshot = archive["sim_states"][case["replay_periods"]].copy()
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        targets = np.asarray(chunk["effective_targets_tcp"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        tool = np.linalg.inv(control["source_from_site"]) @ h
        manifest = read(case["manifest_ref"])
        env = None
        plans = []
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            import mujoco

            model, data = env.sim.model._model, env.sim.data._data
            robot = env.robots[0]
            env.sim.set_state_from_flattened(snapshot)
            env.sim.forward()
            qids, dofs = np.asarray(robot._ref_joint_pos_indexes), np.asarray(robot._ref_joint_vel_indexes)
            actual = data.qpos[qids].copy()
            frozen_time, frozen_velocity = float(data.time), data.qvel.copy()
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.controller.eef_name)

            def fk(q):
                data.qpos[qids] = q
                env.sim.forward()
                site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                offset = site[:3, :3] @ tool[:3, 3]
                return site @ tool, np.vstack((jp[:, dofs] - hat(offset) @ jr[:, dofs], jr[:, dofs]))

            for branch in seeds["cases"][index]["branches"]:
                require(len(branch["rows"]) == branch["tube_configurations"] == 40, "Whole feasible seed path required")
                seed = np.asarray([r["joints"] for r in branch["rows"]])
                started = time.perf_counter()
                solved = plan_joint_path(fk, targets, actual, seed, limits[:, 0], limits[:, 1])
                np.testing.assert_array_equal(solved["actual_anchor"], actual)
                record = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in solved.items()}
                record.update(
                    seed_name=branch["name"], seed_joints=seed.tolist(), compute_seconds=time.perf_counter() - started
                )
                plans.append(record)
                publish(root / f"case{index:02d}_{branch['name']}.json", record)
                print(
                    json.dumps(
                        dict(
                            case=index,
                            seed=branch["name"],
                            energy=solved["energy"],
                            seed_energy=solved["seed_energy"],
                            max_difference=solved["maximum_joint_difference_rad"],
                            success=solved["optimizer_success"],
                            seconds=record["compute_seconds"],
                        )
                    ),
                    flush=True,
                )
            require(float(data.time) == frozen_time, "Geometry planning advanced simulation time")
            np.testing.assert_array_equal(data.qvel, frozen_velocity)
        finally:
            if env is not None:
                env.close()
        cases.append(dict(case=case["case"], actual_anchor=actual.tolist(), joint_limits=limits.tolist(), plans=plans))
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
                    scope="bounded geometric endpoint planning with actual anchor; continuous or dynamic feasibility still unproven",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
