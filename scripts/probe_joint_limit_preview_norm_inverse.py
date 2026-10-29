"""Observe equivalent norm-constraint inverse on the same fixed current snapshots."""

import json
from pathlib import Path

import numpy as np

import s2a.joint_limit_preview_norm_inverse as inverse_module
from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import transform
from s2a.joint_limit_preview import build_joint_preview, rollout_joint_preview
from s2a.joint_limit_preview_norm_inverse import refine_joint_plan
from s2a.joint_limit_preview_objective import preview_objective
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/joint_limit_preview_norm_probe_20261027"
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
                "joint_limit_preview",
                "joint_limit_preview_objective",
                "joint_limit_preview_norm_inverse",
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
            inverse_max_response_evaluations=80,
            inverse_max_iterations=20,
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
            deltas = np.asarray([interface.decode(c) for c in commands])
            local = build_joint_preview(
                mass, jacobian, controller.kp, controller.kd, q, controller.initial_joint, null_kp=10.0, dt=0.002
            )
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            twists = np.asarray([reference_world_twist(generated, j, anchor, 0.05)["twist"] for j in range(4)])

            def evaluate(d):
                result = rollout_joint_preview(
                    local, velocity, d, site, limits[:, 0], limits[:, 1], enforce_limits=True
                )
                return result, preview_objective(result, site_from_tcp, generated[:4], twists)

            rollout, objective = evaluate(deltas)
            captured = []
            original_minimize = inverse_module.minimize

            def observed_minimize(*args, **kwargs):
                result = original_minimize(*args, **kwargs)
                captured.append(result)
                return result

            inverse_module.minimize = observed_minimize
            try:
                inverse = refine_joint_plan(
                    local,
                    velocity,
                    commands,
                    previous,
                    interface,
                    site,
                    site_from_tcp,
                    generated[:4],
                    twists,
                    limits[:, 0],
                    limits[:, 1],
                )
            finally:
                inverse_module.minimize = original_minimize
            column_scales = np.tile(
                (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min), 4
            )
            singular = np.linalg.svd(objective["jacobian"] * column_scales[None, :], compute_uv=False)
            termination = dict(
                normalized_jacobian_singular_values=singular.tolist(),
                jacobian_condition=None if singular[-1] == 0 else float(singular[0] / singular[-1]),
            )
            if captured:
                proposed_commands = captured[0].x.reshape(4, 6)
                _, proposed_score = evaluate(np.asarray([interface.decode(c) for c in proposed_commands]))
                proposed_errors = proposed_score["residual"].reshape(4, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)
                seed_errors_for_diagnostic = objective["residual"].reshape(4, 12)[:, :6] / np.array(
                    [100.0] * 3 + [20.0] * 3
                )
                termination.update(
                    proposal_objective=proposed_score["objective"],
                    max_position_guard_excess_m=float(
                        np.max(
                            np.linalg.norm(proposed_errors[:, :3], axis=1)
                            - np.linalg.norm(seed_errors_for_diagnostic[:, :3], axis=1)
                        )
                    ),
                    max_rotation_guard_excess_rad=float(
                        np.max(
                            np.linalg.norm(proposed_errors[:, 3:], axis=1)
                            - np.linalg.norm(seed_errors_for_diagnostic[:, 3:], axis=1)
                        )
                    ),
                    proposed_commands=proposed_commands.tolist(),
                )
            actual_objective = preview_objective(inverse["rollout"], site_from_tcp, generated[:4], twists)
            np.testing.assert_allclose(actual_objective["objective"], inverse["objective"], atol=1e-10, rtol=0)
            require(inverse["objective"] <= inverse["seed_objective"], "Returned model score worsened")
            seed_errors = objective["residual"].reshape(4, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)
            actual_errors = actual_objective["residual"].reshape(4, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)
            for start, end in ((0, 3), (3, 6)):
                require(
                    np.all(
                        np.linalg.norm(actual_errors[:, start:end], axis=1)
                        <= np.linalg.norm(seed_errors[:, start:end], axis=1) + 1e-10
                    ),
                    "Returned pose guard failed",
                )
            returned = inverse["planned_commands"]
            require(
                np.all(returned >= interface.input_min - 1e-12) and np.all(returned <= interface.input_max + 1e-12),
                "Returned plan box",
            )
            require(np.max(abs(np.diff(np.vstack((previous, returned)), axis=0))) <= 0.25 + 1e-12, "Returned plan rate")
            optimization = {
                key: value for key, value in inverse.items() if key not in ("rollout", "planned_commands", "command")
            }
            cases.append(
                dict(
                    case=case["case"],
                    planned_commands=commands.tolist(),
                    weak_projection_steps=rollout["weak_projection_steps"].tolist(),
                    initial_penetration_rad=rollout["initial_max_penetration_rad"],
                    optimization=optimization,
                    termination=termination,
                    returned_commands=returned.tolist(),
                    returned_tcp_errors=actual_errors.tolist(),
                    seed_tcp_errors=seed_errors.tolist(),
                    objective=objective["objective"],
                    predicted_joint_positions=rollout["joint_position"].tolist(),
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
                    scope="equivalent norm constraints with unchanged score/iteration budgets; model-only diagnostics",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
