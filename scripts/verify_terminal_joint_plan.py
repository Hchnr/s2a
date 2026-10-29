"""Probe full-plan feasibility restoration on fixed current native snapshots."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import transform
from s2a.joint_limit_preview_objective import preview_objective
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.long_joint_preview import build_joint_preview
from s2a.long_joint_preview import rollout_joint_preview as original_rollout
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/terminal_joint_plan_verified_20261027"
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
                "guarded_long_joint_inverse",
                "fast_guarded_long_joint_inverse",
                "preventive_joint_inverse",
                "terminal_joint_plan",
                "interior_joint_preview",
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
            benchmark_repetitions=1,
            intervention_step=72,
            first_pose_tube=[0.01, 0.05],
            joint_margin_rad=0.05,
            terminal_velocity_rule="non-outward along original free near-boundary normals",
            horizons=[8],
            seed_extension="hold final available original H4 planned command",
            pose_guard="original first-pose caps enforced",
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.coupled_pose_preview_runtime import solve_preview
    from s2a.libero_simulation import make_environment

    producer_ref = reference(REPO / "outputs/terminal_joint_plan_probe_20261027/verified.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Complete producer")
    for path, digest in read(producer["registration_ref"])["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    cases = []
    for index in (0, 1):
        case = prepared["cases"][index]
        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        worker = read(refs["worker_result.json"])
        with np.load(checked(worker["arrays_ref"]), allow_pickle=False) as archive:
            state = archive["sim_states"][72].copy()
        original = [json.loads(line) for line in checked(refs["steps.jsonl"]).read_text().splitlines()]
        before = original[72 - 1]
        previous = np.asarray(before["command"])[:6]
        chunk = read(refs["chunks.json"])[9]
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
            np.testing.assert_allclose(commands[0], original[72]["command"][:6], atol=1e-8, rtol=0)
            interface = OSCInterface(
                *(control["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max"))
            )
            local = build_joint_preview(
                mass, jacobian, controller.kp, controller.kd, q, controller.initial_joint, null_kp=10.0, dt=0.002
            )
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            record = producer["cases"][index]["record"]
            require(record["witness_found"] and record["status"] == "model_witness", "Expected current model witnesses")
            commands = np.asarray(record["planned_commands"])
            deltas = np.asarray([interface.decode(c) for c in commands])
            inner = np.asarray(record["rollout"]["inner_joint_position"])
            seed_q, seed_velocity = q.copy(), velocity.copy()
            seed_inner = []
            for command in np.asarray(record["seed_commands"]):
                delta = interface.decode(command)
                rebuilt = build_joint_response(
                    mass,
                    jacobian,
                    controller.kp,
                    controller.kd,
                    seed_q,
                    controller.initial_joint,
                    null_kp=10.0,
                    dt=0.002,
                )
                for k in range(1, 26):
                    endpoint = predict_joint_response(
                        rebuilt, seed_velocity, delta, limits[:, 0], limits[:, 1], substeps=k, enforce_limits=False
                    )
                    seed_inner.append(endpoint["joint_position"].copy())
                seed_q, seed_velocity = endpoint["joint_position"], endpoint["joint_velocity"]
            seed_inner = np.asarray(seed_inner)
            identity = np.eye(len(q))
            normals = np.vstack(
                (
                    identity[np.any(seed_inner >= limits[:, 1] - 0.05, axis=0)],
                    -identity[np.any(seed_inner <= limits[:, 0] + 0.05, axis=0)],
                )
            )
            np.testing.assert_array_equal(normals, np.asarray(record["terminal_normals"]).reshape(-1, len(q)))
            actual_q, actual_velocity = q.copy(), velocity.copy()
            maximum = 0.0
            for period, delta in enumerate(deltas):
                independent = build_joint_response(
                    mass,
                    jacobian,
                    controller.kp,
                    controller.kd,
                    actual_q,
                    controller.initial_joint,
                    null_kp=10.0,
                    dt=0.002,
                )
                for inner_index in range(1, 26):
                    result = predict_joint_response(
                        independent,
                        actual_velocity,
                        delta,
                        limits[:, 0],
                        limits[:, 1],
                        substeps=inner_index,
                        enforce_limits=False,
                    )
                    maximum = max(
                        maximum, float(np.max(abs(result["joint_position"] - inner[25 * period + inner_index - 1])))
                    )
                    np.testing.assert_allclose(
                        result["joint_position"], inner[25 * period + inner_index - 1], atol=1e-10, rtol=0
                    )
                actual_q, actual_velocity = result["joint_position"], result["joint_velocity"]
            signed = normals @ actual_velocity
            np.testing.assert_allclose(signed, record["terminal_signed_velocities_rad_s"], atol=1e-10, rtol=0)
            require(np.all(signed <= 1e-10), "Independent terminal velocity")
            require(
                np.all(inner >= limits[:, 0] + 0.05 - 1e-10) and np.all(inner <= limits[:, 1] - 0.05 + 1e-10),
                "All inner endpoints",
            )
            hard = original_rollout(local, velocity, deltas, site, limits[:, 0], limits[:, 1], enforce_limits=True)
            for key in ("pose", "twist", "joint_position", "joint_velocity"):
                np.testing.assert_allclose(hard[key], record["rollout"][key], atol=1e-10, rtol=0)
            twists = np.asarray([reference_world_twist(generated, j, anchor, 0.05)["twist"] for j in range(8)])
            score = preview_objective(hard, site_from_tcp, generated[:8], twists)
            np.testing.assert_allclose(score["objective"], record["objective"], atol=1e-9, rtol=0)
            first = score["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
            caps = np.asarray(record["first_pose_caps"])
            require(
                np.linalg.norm(first[:3]) <= caps[0] + 1e-10 and np.linalg.norm(first[3:]) <= caps[1] + 1e-10,
                "First TCP caps",
            )
            require(np.max(abs(np.diff(np.vstack((previous, commands)), axis=0))) <= 0.25 + 1e-12, "Rate")
            require(
                np.all(commands >= interface.input_min - 1e-12) and np.all(commands <= interface.input_max + 1e-12),
                "Box",
            )
            cases.append(
                dict(
                    case=case["case"],
                    maximum_inner_joint_difference_rad=maximum,
                    minimum_inner_margin_rad=float(np.min(np.minimum(inner - limits[:, 0], limits[:, 1] - inner))),
                    terminal_normals=normals.tolist(),
                    terminal_signed_velocities_rad_s=signed.tolist(),
                    objective=score["objective"],
                    first_tcp_error=first.tolist(),
                )
            )
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
                    producer_ref=producer_ref,
                    cases=cases,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    control_qualified=False,
                    scope="independent per-period rebuild of 2ms positions and terminal velocities; no physical execution or stability proof",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
