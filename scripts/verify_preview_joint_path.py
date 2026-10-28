"""Fresh FK endpoint and sampled-between-endpoint audit of whole joint paths."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error, transform
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/preview_joint_path_verified_20261027"
    require(not root.exists(), "No overwrite")
    producer_ref = reference(REPO / "outputs/preview_joint_path_20261027/verified.json")
    producer = read(producer_ref)
    require(
        producer["complete"] is producer["passed"] is True and len(producer["cases"]) == 2,
        "Complete fixed cases required",
    )
    registration = read(producer["registration_ref"])
    for path, digest in registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    prepared = read(registration["prepared_ref"])
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment

    cases = []
    for index, case_record in enumerate(producer["cases"]):
        case = prepared["cases"][index]
        require(case_record["case"] == case["case"] and len(case_record["plans"]) == 2, "Fixed complete paths required")
        refs = {r["path"].rsplit("/", 1)[-1]: r for r in case["original_refs"]}
        worker = read(refs["worker_result.json"])
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
            snapshot = archive["sim_states"][case["replay_periods"]].copy()
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        targets, anchor_tcp = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        tool = np.linalg.inv(control["source_from_site"]) @ h
        manifest = read(case["manifest_ref"])
        env = None
        plans = []
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            env.sim.set_state_from_flattened(snapshot)
            env.sim.forward()
            robot, data = env.robots[0], env.sim.data
            qids = robot._ref_joint_pos_indexes
            actual = np.asarray(data.qpos)[qids].copy()
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            np.testing.assert_array_equal(limits, case_record["joint_limits"])

            def pose(q):
                data.qpos[qids] = q
                env.sim.forward()
                return (
                    transform(
                        data.get_site_xpos(robot.controller.eef_name).copy(),
                        data.get_site_xmat(robot.controller.eef_name).copy(),
                    )
                    @ tool
                )

            for record in case_record["plans"]:
                np.testing.assert_array_equal(record["actual_anchor"], actual)
                path, seed = np.asarray(record["joints"]), np.asarray(record["seed_joints"])
                require(path.shape == seed.shape == (40, 7), "Complete40-step paths required")
                require(
                    np.isfinite(path).all()
                    and np.all(path >= limits[:, 0] - 1e-12)
                    and np.all(path <= limits[:, 1] + 1e-12),
                    "Finite bounded path",
                )
                difference = np.diff(np.vstack((actual, path)), axis=0)
                energy = 0.5 * float(np.sum(difference**2))
                seed_energy = 0.5 * float(np.sum(np.diff(np.vstack((actual, seed)), axis=0) ** 2))
                np.testing.assert_allclose(
                    [energy, seed_energy], [record["energy"], record["seed_energy"]], atol=1e-10, rtol=0
                )
                require(energy <= seed_energy + 1e-12, "Energy increase")
                require(
                    record["trajectory_evaluations"] <= 160 and record["optimizer_iterations"] <= 80, "Planner budget"
                )
                endpoint_errors = np.asarray(
                    [pose_error(target, pose(q)) for target, q in zip(targets, path, strict=True)]
                )
                np.testing.assert_allclose(endpoint_errors, record["errors"], atol=1e-9, rtol=0)
                require(
                    np.all(np.linalg.norm(endpoint_errors[:, :3], axis=1) <= 0.01 + 1e-10)
                    and np.all(np.linalg.norm(endpoint_errors[:, 3:], axis=1) <= 0.05 + 1e-10),
                    "Endpoint tube violation",
                )
                samples = []
                for k in range(40):
                    previous_q = actual if k == 0 else path[k - 1]
                    previous_target = anchor_tcp if k == 0 else targets[k - 1]
                    rotation_delta = Rotation.from_matrix(targets[k, :3, :3] @ previous_target[:3, :3].T).as_rotvec()
                    for alpha in registration["interpolation_samples"]:
                        q = previous_q + alpha * (path[k] - previous_q)
                        desired = transform(
                            previous_target[:3, 3] + alpha * (targets[k, :3, 3] - previous_target[:3, 3]),
                            Rotation.from_rotvec(alpha * rotation_delta).as_matrix() @ previous_target[:3, :3],
                        )
                        e = pose_error(desired, pose(q))
                        samples.append(
                            dict(
                                period=k,
                                alpha=alpha,
                                error=e.tolist(),
                                joint_bound_excess_rad=float(
                                    max(0, np.max(limits[:, 0] - q), np.max(q - limits[:, 1]))
                                ),
                            )
                        )
                errors = np.asarray([s["error"] for s in samples])
                plans.append(
                    dict(
                        seed_name=record["seed_name"],
                        energy=energy,
                        seed_energy=seed_energy,
                        first_difference_max_rad=float(np.max(abs(difference[0]))),
                        maximum_difference_rad=float(np.max(abs(difference))),
                        endpoint_position_max_m=float(np.max(np.linalg.norm(endpoint_errors[:, :3], axis=1))),
                        endpoint_rotation_max_rad=float(np.max(np.linalg.norm(endpoint_errors[:, 3:], axis=1))),
                        sample_position_max_m=float(np.max(np.linalg.norm(errors[:, :3], axis=1))),
                        sample_rotation_max_rad=float(np.max(np.linalg.norm(errors[:, 3:], axis=1))),
                        samples_inside_tube=int(
                            np.sum(
                                (np.linalg.norm(errors[:, :3], axis=1) <= 0.01 + 1e-10)
                                & (np.linalg.norm(errors[:, 3:], axis=1) <= 0.05 + 1e-10)
                            )
                        ),
                        samples=samples,
                    )
                )
        finally:
            if env is not None:
                env.close()
        cases.append(dict(case=case["case"], plans=plans))
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    producer_ref=producer_ref,
                    cases=cases,
                    source_ref=reference(__file__),
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    control_qualified=False,
                    scope="FK endpoints and three interpolation samples per interval; no continuous, collision or controller guarantee",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
