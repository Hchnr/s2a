"""Causal earlier-query frozen-model limit forecasts on two old trajectories."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference
from s2a.geometry import transform
from s2a.long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/early_limit_risk_20261027"
    root.mkdir(exist_ok=False)
    prepared_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    prepared = read(prepared_ref)
    sources = dict(
        read(reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json"))["source_sha256"]
    )
    for path in (Path(__file__), REPO / "src/s2a/long_joint_preview.py", REPO / "src/s2a/joint_limit_response.py"):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    registration_ref = publish(
        root / "registration.json",
        dict(
            prepared_ref=prepared_ref,
            source_sha256=sources,
            cases=[0, 1],
            query_indices=list(range(1, 12)),
            horizons=[4, 8],
            seed=42,
            physics_steps=0,
            scope="saved actual states and already generated references at earlier queries; no new actions",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.coupled_pose_preview_runtime import solve_preview
    from s2a.libero_simulation import make_environment

    cases = []
    for index in (0, 1):
        case = prepared["cases"][index]
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        worker = read(refs["worker_result.json"])
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
            snapshots = archive["sim_states"].copy()
        rows = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        chunks = read(refs["chunks.json"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        tool = np.linalg.inv(control["source_from_site"]) @ np.asarray(read(case["tool_ref"])["source_from_tcp"])
        manifest = read(case["manifest_ref"])
        env = None
        records = []
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            import mujoco

            model, data = env.sim.model._model, env.sim.data._data
            robot = env.robots[0]
            controller = robot.controller
            qids, dofs = np.asarray(robot._ref_joint_pos_indexes), np.asarray(robot._ref_joint_vel_indexes)
            np.testing.assert_array_equal(qids, np.arange(7))
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
            interface = OSCInterface(
                *(control["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max"))
            )
            for query in range(1, 12):
                prefix = query * 8
                env.sim.set_state_from_flattened(snapshots[prefix])
                env.sim.forward()
                np.testing.assert_allclose(env.sim.get_state().flatten(), snapshots[prefix], atol=1e-14, rtol=0)
                mass = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, mass, data.qM)
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                mass = mass[np.ix_(dofs, dofs)]
                jacobian = np.vstack((jp, jr))[:, dofs]
                q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
                site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
                chunk = chunks[query]
                generated = np.asarray(chunk["effective_targets_tcp"])
                anchor = np.asarray(chunk["anchor_tcp"])
                previous = np.asarray(rows[prefix - 1]["command"])[:6]
                seed = solve_preview(
                    mass,
                    jacobian,
                    site,
                    jacobian @ velocity,
                    generated @ np.linalg.inv(tool),
                    anchor @ np.linalg.inv(tool),
                    0,
                    previous,
                    control["contract"],
                    innovation=np.asarray(rows[prefix - 1]["coupled_response_history"]["innovation_after"]),
                )
                commands = np.asarray(seed["preview"]["planned_commands"])
                np.testing.assert_allclose(commands[0], rows[prefix]["command"][:6], atol=1e-8, rtol=0)
                local = build_joint_preview(
                    mass, jacobian, controller.kp, controller.kd, q, controller.initial_joint, null_kp=10.0, dt=0.002
                )
                forecasts = []
                for horizon in (4, 8):
                    planned = np.vstack((commands, np.tile(commands[-1], (horizon - len(commands), 1))))
                    deltas = np.asarray([interface.decode(c) for c in planned])
                    if np.any(np.all(deltas[:, 3:] == 0, axis=1)):
                        forecasts.append(dict(horizon=horizon, status="unsupported_future_rotation_latch"))
                        continue
                    free = rollout_joint_preview(
                        local, velocity, deltas, site, limits[:, 0], limits[:, 1], enforce_limits=False
                    )
                    margin = np.minimum(free["joint_position"] - limits[:, 0], limits[:, 1] - free["joint_position"])
                    violation = np.flatnonzero(np.any(margin < 0, axis=1))
                    forecasts.append(
                        dict(
                            horizon=horizon,
                            status="computed",
                            minimum_margin_rad_by_joint=margin.min(axis=0).tolist(),
                            first_predicted_violation_period=None if not len(violation) else int(violation[0]) + 1,
                        )
                    )
                records.append(
                    dict(
                        query=query,
                        step=prefix,
                        q=q.tolist(),
                        current_margin_rad_by_joint=np.minimum(q - limits[:, 0], limits[:, 1] - q).tolist(),
                        forecasts=forecasts,
                    )
                )
            actual_q = snapshots[:, 1:8]
            margin = np.minimum(actual_q - limits[:, 0], limits[:, 1] - actual_q)
            first = {}
            for threshold in (0.05, 0.02, 0.01, 0.001, 0.0):
                crossing = np.flatnonzero(margin[:, 6] <= threshold)
                first[str(threshold)] = None if not len(crossing) else int(crossing[0])
        finally:
            if env is not None:
                env.close()
        cases.append(
            dict(case=case["case"], arrays_ref=worker["arrays_ref"], first_q7_threshold_steps=first, records=records)
        )
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
                    scope="earlier frozen free forecasts; no limit-force evidence, interventions or global feasibility proof",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
