"""Reconstruct the captured causal step85 seed and compare numerical starts."""

import json
import time
from pathlib import Path

import numpy as np

from s2a.adaptive_joint_margin import maximize_margin
from s2a.adjustable_interior_joint_plan import solve_interior_plan as refine_plan
from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import transform
from s2a.interior_joint_preview import build_joint_preview
from s2a.joint_limit_preview_objective import preview_objective
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/adaptive_joint_margin_probe_20261027"
    root.mkdir(exist_ok=False)
    capture_ref = reference(REPO / "outputs/interior_joint_feedback_verified_20261027/budget_failure_capture.json")
    capture = read(capture_ref)
    verified = read(capture["verified_ref"])
    sources = dict(read(read(verified["owner_ref"])["registration_ref"])["source_sha256"])
    for path in (
        Path(__file__),
        REPO / "src/s2a/adaptive_joint_margin.py",
        REPO / "src/s2a/adjustable_interior_joint_plan.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    registration_ref = publish(
        root / "registration.json",
        dict(
            capture_ref=capture_ref,
            source_sha256=sources,
            starts=["original_seed", "shifted_previous_plan", "linear_witness"],
            horizon=8,
            seed=42,
            max_iterations_per_phase=60,
            phases=["maximize retained margin", "tracking refinement at retained margin minus 1e-9 numerical buffer"],
            max_responses_per_phase=160,
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
        old_plan = np.asarray(completed["refinement"]["planned_commands"])
        candidates = {
            "original_seed": seed,
            "shifted_previous_plan": np.vstack((old_plan[1:], old_plan[-1])),
            "linear_witness": np.asarray(replay["results"]["original_seed"]["linear_witness_commands"]),
        }
        for name, candidate in candidates.items():
            started = time.perf_counter()
            margin = maximize_margin(*args, initial_guess=candidate)
            record = {
                "margin_search": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in margin.items() if k != "rollout"
                }
            }
            record["margin_seconds"] = time.perf_counter() - started
            if margin["witness_found"]:
                requested = max(
                    0.0,
                    min(margin["retained_margin_rad"], margin["minimum_inner_margin_rad"], margin["current_margin_rad"])
                    - 1e-9,
                )
                started = time.perf_counter()
                refined = refine_plan(*args, initial_guess=margin["planned_commands"], joint_margin_rad=requested)
                record["refinement_seconds"] = time.perf_counter() - started
                record["refinement"] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in refined.items() if k != "rollout"
                }
                chosen = refined if refined["witness_found"] else margin
                record["chosen_phase"] = "tracking_refinement" if refined["witness_found"] else "margin_search"
                commands = chosen["planned_commands"]
                inner = chosen["rollout"]["inner_joint_position"]
                require(
                    np.all(inner >= limits[:, 0] + requested - 1e-10)
                    and np.all(inner <= limits[:, 1] - requested + 1e-10),
                    "Retained interior",
                )
                require(
                    np.all(commands >= interface.input_min - 1e-12) and np.all(commands <= interface.input_max + 1e-12),
                    "Box",
                )
                require(np.max(abs(np.diff(np.vstack((previous, commands)), axis=0))) <= 0.25 + 1e-12, "Rate")
                score = preview_objective(chosen["rollout"], tool, generated[5:13], twists)
                e = score["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
                caps = np.asarray(replay["results"]["original_seed"]["first_pose_caps"])
                require(
                    np.linalg.norm(e[:3]) <= caps[0] + 1e-10 and np.linalg.norm(e[3:]) <= caps[1] + 1e-10,
                    "Original TCP caps",
                )
                minimum = float(np.min(np.minimum(inner - limits[:, 0], limits[:, 1] - inner)))
                record.update(
                    planned_commands=commands.tolist(),
                    first_tcp_error=e.tolist(),
                    first_pose_caps=caps.tolist(),
                    retained_margin_rad=requested,
                    minimum_inner_margin_rad=minimum,
                    tracking_objective=score["objective"],
                    requested_margin_satisfied=bool(minimum >= 0.05 - 1e-10),
                    shortfall_from_requested_rad=max(0.0, 0.05 - minimum),
                )
            results[name] = record
            publish(root / f"{name}.json", record)
            print(
                json.dumps(
                    dict(
                        start=name,
                        witness=margin["witness_found"],
                        retained=record.get("retained_margin_rad"),
                        minimum=record.get("minimum_inner_margin_rad"),
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
                    scope="retained-margin model witnesses with original TCP caps; smaller margin does not pass original .05rad target",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
