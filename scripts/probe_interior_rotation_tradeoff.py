"""Reconstruct the captured causal step85 seed and compare numerical starts."""

import json
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import transform
from s2a.interior_joint_preview import build_joint_preview
from s2a.interior_rotation_tradeoff import solve_interior_plan as tradeoff_solve
from s2a.joint_limit_preview_objective import preview_objective
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/interior_rotation_tradeoff_20261027"
    root.mkdir(exist_ok=False)
    capture_ref = reference(REPO / "outputs/interior_joint_feedback_verified_20261027/budget_failure_capture.json")
    capture = read(capture_ref)
    verified = read(capture["verified_ref"])
    sources = dict(read(read(verified["owner_ref"])["registration_ref"])["source_sha256"])
    for path in (Path(__file__), REPO / "src/s2a/interior_rotation_tradeoff.py"):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    registration_ref = publish(
        root / "registration.json",
        dict(
            capture_ref=capture_ref,
            source_sha256=sources,
            starts=["original_seed", "linear_witness"],
            objective="half squared first SO3 error",
            rotation_cap_imposed=False,
            original_cap_still_reported=True,
            horizon=8,
            seed=42,
            max_iterations_per_start=60,
            max_responses_per_start=160,
            joint_margin_rad=0.05,
            first_pose_caps="max(original current free seed error,.01m/.05rad)",
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.coupled_pose_preview_runtime import solve_preview
    from s2a.coupled_response_history import CoupledResponseHistory
    from s2a.current_coupled_response import build_response
    from s2a.generated_reference_twist import reference_world_twist
    from s2a.libero_simulation import make_environment, measured_twist

    rows = read(capture["rows_ref"])
    before, completed, next_row = rows[11], rows[12], rows[13]
    manifest = read(capture["manifest_ref"])
    chunks = read(capture["chunks_ref"])
    chunk = chunks[10]
    generated = np.asarray(chunk["effective_targets_tcp"])
    anchor = np.asarray(chunk["anchor_tcp"])
    control = read(reference(Path(capture["chunks_ref"]["path"]).parent / "controller.json"))
    tool = np.linalg.inv(control["source_from_site"]) @ np.asarray(read(capture["tool_ref"])["source_from_tcp"])
    interface = OSCInterface(*(control["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max")))
    env = None
    results = {}
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        import mujoco

        model, data = env.sim.model._model, env.sim.data._data
        robot = env.robots[0]
        controller = robot.controller
        qids, dofs = np.asarray(robot._ref_joint_pos_indexes), np.asarray(robot._ref_joint_vel_indexes)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )

        def restore(row):
            state = np.asarray(row["sim_state"])
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            np.testing.assert_allclose(env.sim.get_state().flatten(), state, atol=1e-14, rtol=0)
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            return (
                mass[np.ix_(dofs, dofs)],
                np.vstack((jp, jr))[:, dofs],
                transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()),
                measured_twist(env),
                data.qpos[qids].copy(),
                data.qvel[dofs].copy(),
            )

        mass, j, site, twist, _, _ = restore(before)
        relative = np.asarray(completed["refinement"]["expected_goal_ori"]) @ site[:3, :3].T
        history = CoupledResponseHistory()
        history.commit_executed(
            site,
            twist,
            build_response(mass, j, controller.kp, controller.kd, relative),
            np.asarray(completed["command"])[:3],
            control["contract"],
        )
        mass, j, site, twist, q, velocity = restore(completed)
        history.observe_completed(site, twist)
        previous = np.asarray(completed["command"])[:6]
        solved = solve_preview(
            mass,
            j,
            site,
            twist,
            generated @ np.linalg.inv(tool),
            anchor @ np.linalg.inv(tool),
            5,
            previous,
            control["contract"],
            innovation=history.current(site, twist),
        )
        np.testing.assert_allclose(solved["command"], next_row["seed_command"], atol=1e-8, rtol=0)
        original = np.asarray(solved["preview"]["planned_commands"])
        seed = np.vstack((original, np.tile(original[-1], (8 - len(original), 1))))
        local = build_joint_preview(
            mass, j, controller.kp, controller.kd, q, controller.initial_joint, null_kp=10.0, dt=0.002
        )
        twists = np.asarray([reference_world_twist(generated, k, anchor, 0.05)["twist"] for k in range(5, 13)])
        args = (
            local,
            velocity,
            seed,
            previous,
            interface,
            site,
            tool,
            generated[5:13],
            twists,
            limits[:, 0],
            limits[:, 1],
        )
        replay_ref = reference(REPO / "outputs/interior_budget_replay_20261027/verified.json")
        replay = read(replay_ref)
        np.testing.assert_allclose(seed, replay["reconstructed_seed"], atol=1e-8, rtol=0)
        candidates = {
            "original_seed": seed,
            "linear_witness": np.asarray(replay["results"]["original_seed"]["linear_witness_commands"]),
        }
        for name, candidate in candidates.items():
            start = time.perf_counter()
            result = tradeoff_solve(*args, initial_guess=candidate)
            record = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items() if k != "rollout"}
            record["seconds"] = time.perf_counter() - start
            if result["witness_found"]:
                inner = result["rollout"]["inner_joint_position"]
                require(
                    np.all(inner >= limits[:, 0] + 0.05 - 1e-10) and np.all(inner <= limits[:, 1] - 0.05 + 1e-10),
                    "Joint interior",
                )
                commands = result["planned_commands"]
                require(
                    np.all(commands >= interface.input_min - 1e-12) and np.all(commands <= interface.input_max + 1e-12),
                    "Box",
                )
                require(np.max(abs(np.diff(np.vstack((previous, commands)), axis=0))) <= 0.25 + 1e-12, "Rate")
                score = preview_objective(result["rollout"], tool, generated[5:13], twists)
                error = score["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
                caps = np.asarray(replay["results"]["original_seed"]["first_pose_caps"])
                np.testing.assert_allclose(result["first_pose_caps"], caps, atol=1e-8, rtol=0)
                require(np.linalg.norm(error[:3]) <= caps[0] + 1e-10, "Original position cap")
                record.update(
                    first_tcp_error=error.tolist(),
                    original_full_constraints_satisfied=bool(np.linalg.norm(error[3:]) <= caps[1] + 1e-10),
                    rotation_excess_rad=float(np.linalg.norm(error[3:]) - caps[1]),
                )
            results[name] = record
            publish(root / f"{name}.json", record)
            print(
                json.dumps(
                    dict(
                        start=name,
                        status=record["status"],
                        rotation=record.get("first_rotation_error_rad"),
                        original_constraints=record.get("original_full_constraints_satisfied"),
                        seconds=record["seconds"],
                    )
                ),
                flush=True,
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
                    replay_ref=replay_ref,
                    reconstructed_seed=seed.tolist(),
                    results=results,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    control_qualified=False,
                    scope="exact first-SO3 minimization diagnostic with original joint/rate/position constraints; rotation cap only reported",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
