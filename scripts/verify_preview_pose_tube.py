"""Fresh-environment verification of each reported bounded pose tube configuration."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import pose_error, transform
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/preview_pose_tube_verified_20261027"
    require(not root.exists(), "No overwrite")
    producer_ref = reference(REPO / "outputs/preview_pose_tube_20261027/verified.json")
    producer = read(producer_ref)
    require(
        producer["passed"] is producer["complete"] is True and len(producer["cases"]) == 2, "Complete fixed diagnostic"
    )
    registration = read(producer["registration_ref"])
    for path, digest in registration["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    prepared = read(registration["prepared_ref"])
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment

    cases, count = [], 0
    for index, entry in enumerate(producer["cases"]):
        case = prepared["cases"][index]
        require(entry["case"] == case["case"], "Case changed")
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        targets = np.asarray(chunk["effective_targets_tcp"])
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        tool = np.linalg.inv(control["source_from_site"]) @ h
        manifest = read(case["manifest_ref"])
        env = None
        branches = []
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            robot, data = env.robots[0], env.sim.data
            qids = robot._ref_joint_pos_indexes
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            require([b["name"] for b in entry["branches"]] == registration["branches"], "Fixed branches changed")
            for branch in entry["branches"]:
                require([r["index"] for r in branch["rows"]] == list(range(40)), "Complete40 targets required")
                positives, solved = 0, 0
                worst_position = worst_rotation = 0.0
                for row in branch["rows"]:
                    count += 1
                    if "joints" not in row:
                        require(row["within_tube"] is False and "error_message" in row, "Unreported failed solve")
                        continue
                    solved += 1
                    q = np.asarray(row["joints"])
                    require(
                        q.shape == (7,)
                        and np.isfinite(q).all()
                        and row["cold_ik_nfev"] <= 200
                        and row["optimizer_iterations"] <= 60,
                        "Finite budgeted joints",
                    )
                    require(np.all(q >= limits[:, 0] - 1e-12) and np.all(q <= limits[:, 1] + 1e-12), "Joint bounds")
                    data.qpos[qids] = q
                    env.sim.forward()
                    site = transform(
                        data.get_site_xpos(robot.controller.eef_name).copy(),
                        data.get_site_xmat(robot.controller.eef_name).copy(),
                    )
                    error = pose_error(targets[row["index"]], site @ tool)
                    np.testing.assert_allclose(error, row["error"], atol=1e-9, rtol=0)
                    witness = bool(
                        np.linalg.norm(error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(error[3:]) <= 0.05 + 1e-10
                    )
                    require(witness is row["within_tube"], "Pose witness label differs")
                    positives += int(witness)
                    worst_position = max(worst_position, float(np.linalg.norm(error[:3])))
                    worst_rotation = max(worst_rotation, float(np.linalg.norm(error[3:])))
                require(positives == branch["tube_configurations"], "Branch count differs")
                branches.append(
                    dict(
                        name=branch["name"],
                        returned_solutions=solved,
                        verified_tube_configurations=positives,
                        max_position_residual_m=worst_position,
                        max_rotation_residual_rad=worst_rotation,
                    )
                )
        finally:
            if env is not None:
                env.close()
        cases.append(dict(case=entry["case"], branches=branches))
    require(count == registration["total_tube_queries"] == 160, "Fixed160 solve records required")
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
                    fresh_cases=0,
                    world_model_calls=0,
                    control_qualified=False,
                    scope="fresh FK and joint-range check only; no controller, collision-free or timed-path qualification",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
