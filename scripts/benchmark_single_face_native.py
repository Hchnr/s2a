"""Probe full-plan feasibility restoration on fixed current native snapshots."""

import json
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference
from s2a.fast_long_joint_preview import rollout_joint_preview as fast_rollout
from s2a.geometry import transform
from s2a.long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/single_face_native_benchmark_20261027"
    root.mkdir(exist_ok=False)
    prepared_ref = reference(REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json")
    prepared = read(prepared_ref)
    old_ref = reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json")
    sources = dict(read(old_ref)["source_sha256"])
    for path in [
        Path(__file__),
        *[
            REPO / f"src/s2a/{n}.py"
            for n in (
                "long_joint_preview",
                "joint_limit_preview_objective",
                "long_joint_inverse",
                "fast_long_joint_preview",
                "single_face_projection",
                "joint_limit_response",
                "nullspace_convex_qp",
            )
        ],
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    registration_ref = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            case_indices=[0, 1],
            benchmark_repetitions=5,
            horizons=[4, 8, 16],
            seed_extension="hold final available original H4 planned command",
            pose_guard=False,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
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
            state = archive["sim_states"][case["replay_periods"]].copy()
        original = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        before = original[case["replay_periods"] - 1]
        previous = np.asarray(before["command"])[:6]
        chunk = read(refs["chunks.json"])[case["event"]["query_index"]]
        generated, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
        control = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        site_from_tcp = np.linalg.inv(control["source_from_site"]) @ h
        targets_site, anchor_site = generated @ np.linalg.inv(site_from_tcp), anchor @ np.linalg.inv(site_from_tcp)
        manifest = read(case["manifest_ref"])
        env = None
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            import mujoco

            model, data = env.sim.model._model, env.sim.data._data
            robot, controller = env.robots[0], env.robots[0].controller
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            dofs, qids = np.asarray(robot._ref_joint_vel_indexes), np.asarray(robot._ref_joint_pos_indexes)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            mass, jacobian = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
            q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
            site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
            solved = solve_preview(
                mass,
                jacobian,
                site,
                jacobian @ velocity,
                targets_site,
                anchor_site,
                0,
                previous,
                control["contract"],
                innovation=np.asarray(before["coupled_response_history"]["innovation_after"]),
            )
            commands = np.asarray(solved["preview"]["planned_commands"])
            np.testing.assert_allclose(commands[0], original[case["replay_periods"]]["command"][:6], atol=1e-8, rtol=0)
            interface = OSCInterface(
                *(control["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max"))
            )
            local = build_joint_preview(
                mass, jacobian, controller.kp, controller.kd, q, controller.initial_joint, null_kp=10.0, dt=0.002
            )
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            results = []
            for horizon in (4, 8, 16):
                seed = np.vstack((commands, np.tile(commands[-1], (horizon - len(commands), 1))))
                deltas = np.asarray([interface.decode(c) for c in seed])
                timings, outputs = {}, {}
                for name, function in (("general", rollout_joint_preview), ("single_face", fast_rollout)):
                    elapsed = []
                    for _ in range(5):
                        started = time.perf_counter()
                        outputs[name] = function(
                            local, velocity, deltas, site, limits[:, 0], limits[:, 1], enforce_limits=True
                        )
                        elapsed.append(time.perf_counter() - started)
                    timings[name] = dict(seconds=elapsed, median_seconds=float(np.median(elapsed)))
                max_difference = {}
                for key in (
                    "pose",
                    "twist",
                    "joint_position",
                    "joint_velocity",
                    "joint_position_jacobian",
                    "joint_velocity_jacobian",
                    "position_jacobian",
                    "rotation_jacobian",
                    "twist_jacobian",
                ):
                    np.testing.assert_allclose(outputs["general"][key], outputs["single_face"][key], atol=1e-10, rtol=0)
                    max_difference[key] = float(np.max(abs(outputs["general"][key] - outputs["single_face"][key])))
                np.testing.assert_array_equal(
                    outputs["general"]["weak_projection_steps"], outputs["single_face"]["weak_projection_steps"]
                )
                record = dict(horizon=horizon, timings=timings, maximum_differences=max_difference)
                results.append(record)
                print(json.dumps(dict(case=index, **record)), flush=True)
            cases.append(dict(case=case["case"], results=results))
        finally:
            if env is not None:
                env.close()
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
                    scope="native-snapshot frozen rollout equivalence and microbenchmark; no new physical execution",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
