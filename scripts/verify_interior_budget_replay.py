"""Independently check the captured seed trajectory and warm-start violations."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import transform
from s2a.joint_limit_preview_objective import preview_objective
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/interior_budget_replay_verified_20261027"
    root.mkdir(exist_ok=False)
    producer_ref = reference(REPO / "outputs/interior_budget_replay_20261027/verified.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Complete replay")
    reg = read(producer["registration_ref"])
    for path, digest in reg["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    capture = read(reg["capture_ref"])
    rows = read(capture["rows_ref"])
    manifest = read(capture["manifest_ref"])
    control = read(reference(Path(capture["chunks_ref"]["path"]).parent / "controller.json"))
    chunk = read(capture["chunks_ref"])[10]
    targets = np.asarray(chunk["effective_targets_tcp"])
    anchor = np.asarray(chunk["anchor_tcp"])
    tool = np.linalg.inv(control["source_from_site"]) @ np.asarray(read(capture["tool_ref"])["source_from_tcp"])
    interface = OSCInterface(*(control["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max")))
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.generated_reference_twist import reference_world_twist
    from s2a.libero_simulation import make_environment

    env = None
    checks = {}
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        import mujoco

        model, data = env.sim.model._model, env.sim.data._data
        robot = env.robots[0]
        controller = robot.controller
        state = np.asarray(rows[12]["sim_state"])
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        np.testing.assert_allclose(env.sim.get_state().flatten(), state, atol=1e-14, rtol=0)
        qids, dofs = np.asarray(robot._ref_joint_pos_indexes), np.asarray(robot._ref_joint_vel_indexes)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        mass = mass[np.ix_(dofs, dofs)]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        j = np.vstack((jp, jr))[:, dofs]
        q, velocity = data.qpos[qids].copy(), data.qvel[dofs].copy()
        site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )
        local = build_joint_preview(
            mass, j, controller.kp, controller.kd, q, controller.initial_joint, null_kp=10.0, dt=0.002
        )
        twists = np.asarray([reference_world_twist(targets, k, anchor, 0.05)["twist"] for k in range(5, 13)])
        seed = np.asarray(producer["reconstructed_seed"])
        delta = np.asarray([interface.decode(c) for c in seed])
        original = rollout_joint_preview(local, velocity, delta, site, limits[:, 0], limits[:, 1], enforce_limits=True)
        score = preview_objective(original, tool, targets[5:13], twists)
        errors = score["residual"].reshape(8, 12)[:, :6] / np.array([100.0] * 3 + [20.0] * 3)
        np.testing.assert_allclose(errors, rows[13]["refinement"]["plan_guard"]["seed_tcp_errors"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            score["objective"], rows[13]["refinement"]["optimization"]["seed_tracking_objective"], atol=1e-8, rtol=0
        )
        caps = np.asarray(producer["results"]["original_seed"]["first_pose_caps"])
        np.testing.assert_allclose(caps, capture["hard_diagnostic"]["first_pose_caps"], atol=1e-8, rtol=0)
        previous_plan = np.asarray(rows[12]["refinement"]["planned_commands"])
        candidates = {
            "shifted_previous_plan": np.vstack((previous_plan[1:], previous_plan[-1])),
            "linear_witness": np.asarray(producer["results"]["original_seed"]["linear_witness_commands"]),
        }
        for name, commands in candidates.items():
            delta = np.asarray([interface.decode(c) for c in commands])
            candidate_q, candidate_velocity = q.copy(), velocity.copy()
            inner = []
            for d in delta:
                rebuilt = build_joint_response(
                    mass, j, controller.kp, controller.kd, candidate_q, controller.initial_joint, null_kp=10.0, dt=0.002
                )
                for k in range(1, 26):
                    end = predict_joint_response(
                        rebuilt, candidate_velocity, d, limits[:, 0], limits[:, 1], substeps=k, enforce_limits=False
                    )
                    inner.append(end["joint_position"].copy())
                candidate_q, candidate_velocity = end["joint_position"], end["joint_velocity"]
            inner = np.asarray(inner)
            margin = float(np.min(np.minimum(inner - limits[:, 0], limits[:, 1] - inner)))
            rollout = rollout_joint_preview(
                local, velocity, delta, site, limits[:, 0], limits[:, 1], enforce_limits=False
            )
            objective = preview_objective(rollout, tool, targets[5:13], twists)
            first = objective["residual"][:6] / np.array([100.0] * 3 + [20.0] * 3)
            recorded = producer["results"][name]["initial_candidate"]
            np.testing.assert_allclose(margin, recorded["minimum_inner_margin_rad"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(first, recorded["first_tcp_error"], atol=1e-10, rtol=0)
            checks[name] = dict(
                minimum_inner_margin_rad=margin,
                position_error_m=float(np.linalg.norm(first[:3])),
                rotation_error_rad=float(np.linalg.norm(first[3:])),
                rotation_cap_rad=float(caps[1]),
                rotation_excess_rad=float(np.linalg.norm(first[3:]) - caps[1]),
                joint_margin_satisfied=bool(margin >= 0.05 - 1e-10),
                first_pose_satisfied=bool(
                    np.linalg.norm(first[:3]) <= caps[0] + 1e-10 and np.linalg.norm(first[3:]) <= caps[1] + 1e-10
                ),
            )
    finally:
        if env is not None:
            env.close()
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    producer_ref=producer_ref,
                    source_ref=reference(Path(__file__)),
                    checks=checks,
                    seed_tracking_objective=score["objective"],
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    control_qualified=False,
                    scope="independent original-model replay of captured seed and candidate constraint violations; no global infeasibility proof",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
