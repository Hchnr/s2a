"""Compare geometric first motion with the frozen fixed-OSC autonomous null mode."""

import inspect
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.frozen_null_mode import forecast_null_modes
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/joint_path_null_mode_20261027"
    root.mkdir(exist_ok=False)
    path_verified_ref = reference(REPO / "outputs/preview_joint_path_verified_20261027/verified.json")
    verified = read(path_verified_ref)
    require(verified["passed"] is verified["complete"] is True, "Independent geometric path check required")
    producer = read(verified["producer_ref"])
    path_registration = read(producer["registration_ref"])
    prepared = read(path_registration["prepared_ref"])
    old_ref = reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json")
    sources = dict(read(old_ref)["source_sha256"])
    for path in (
        Path(__file__),
        REPO / "src/s2a/frozen_null_mode.py",
        REPO / "src/s2a/joint_limit_response.py",
        REPO / "src/s2a/nullspace_convex_qp.py",
        REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/controllers/osc.py",
        REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/control_utils.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    probe_names = ["original_seed", "previous", "feasible_lower_corner", "feasible_upper_corner"]
    registration_ref = publish(
        root / "registration.json",
        dict(
            path_verified_ref=path_verified_ref,
            source_sha256=sources,
            cases=[0, 1],
            geometric_seed_name="sequential_current",
            null_kp=10.0,
            dt=0.05,
            local_model_probes=probe_names,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            seed=42,
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
            snapshot = archive["sim_states"][case["replay_periods"]].copy()
        rows = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        previous = np.asarray(rows[case["replay_periods"] - 1]["command"])[:6]
        original_seed = np.asarray(rows[case["replay_periods"]]["command"])[:6]
        plan = next(p for p in producer["cases"][index]["plans"] if p["seed_name"] == "sequential_current")
        manifest = read(case["manifest_ref"])
        env = None
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{index:02d}", manifest["record"], 0)
            import mujoco
            from robosuite.utils.control_utils import nullspace_torques

            require(
                inspect.signature(nullspace_torques).parameters["joint_kp"].default == 10, "Native null gain changed"
            )
            robot, controller = env.robots[0], env.robots[0].controller
            require(
                controller.uncoupling and controller.use_delta and controller.impedance_mode == "fixed",
                "Fixed OSC required",
            )
            model, data = env.sim.model._model, env.sim.data._data
            env.sim.set_state_from_flattened(snapshot)
            env.sim.forward()
            qids, dofs = np.asarray(robot._ref_joint_pos_indexes), np.asarray(robot._ref_joint_vel_indexes)
            q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
            np.testing.assert_array_equal(q, plan["actual_anchor"])
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            mass, j = mass[np.ix_(dofs, dofs)], np.vstack((jp, jr))[:, dofs]
            mode = forecast_null_modes(mass, j, q, velocity, controller.initial_joint)
            require(mode["rows"].shape == (1, 7), "One redundant Panda mode expected")
            np.testing.assert_allclose(mode["task_forcing_residual"], np.zeros((1, 6)), atol=1e-10, rtol=0)
            interface = OSCInterface(
                *(getattr(controller, k) for k in ("input_min", "input_max", "output_min", "output_max"))
            )
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            local = build_joint_response(
                mass, j, controller.kp, controller.kd, q, controller.initial_joint, null_kp=10.0, dt=0.002
            )
            probes = []
            commands = [
                original_seed,
                previous,
                np.maximum(interface.input_min, previous - 0.25),
                np.minimum(interface.input_max, previous + 0.25),
            ]
            for name, command in zip(probe_names, commands, strict=True):
                predictions = []
                for limited in (False, True):
                    result = predict_joint_response(
                        local,
                        velocity,
                        interface.decode(command),
                        limits[:, 0],
                        limits[:, 1],
                        substeps=25,
                        enforce_limits=limited,
                    )
                    predicted_mode = mode["rows"] @ result["joint_position"]
                    if not limited:
                        np.testing.assert_allclose(predicted_mode, mode["next"], atol=1e-10, rtol=0)
                    predictions.append(
                        dict(limited=limited, null_displacement=(predicted_mode - mode["current"]).tolist())
                    )
                probes.append(dict(name=name, command=command.tolist(), predictions=predictions))
            observations = []
            for name, folder in (
                ("original", "preview_chunk_branches_20261027"),
                ("four_period", "joint_limit_preview_branches_20261027"),
            ):
                row_ref = reference(REPO / f"outputs/{folder}/case{index:02d}_h40/steps.json")
                after = np.asarray(read(row_ref)[0]["sim_state"])
                env.sim.set_state_from_flattened(after)
                env.sim.forward()
                observations.append(
                    dict(
                        name=name,
                        rows_ref=row_ref,
                        measured_null_displacement=(mode["rows"] @ (data.qpos[qids] - q)).tolist(),
                    )
                )
            desired_delta = np.asarray(plan["joints"])[0] - q
            cases.append(
                dict(
                    case=case["case"],
                    actual_q=q.tolist(),
                    actual_velocity=velocity.tolist(),
                    mode_rows=mode["rows"].tolist(),
                    planned_first_joint_difference=desired_delta.tolist(),
                    planned_null_displacement=(mode["rows"] @ desired_delta).tolist(),
                    free_model_null_displacement=mode["displacement"].tolist(),
                    probes=probes,
                    observations=observations,
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
                    scope="frozen free null-mode identity, ideal-limit probes and saved measured first steps; no global nonlinear impossibility proof",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
