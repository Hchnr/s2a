"""Replay saved commands to the failed QP; no model query or fresh initial state."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, write_json
from s2a.clean_training_runtime import read, reference
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/preview_qp_failure_replay_20261027"
CASE = REPO / "outputs/serialized_preview_wam_development_20261026/development/cases/task2_init0_s42_raw"


def main():
    strict_native_environment()
    ROOT.mkdir(exist_ok=False)
    result = dict(complete=False, passed=False, replay_steps=0, new_model_calls=0, fresh_cases=0)
    env = None
    try:
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
        from s2a import coupled_pose_preview
        from s2a.coupled_pose_preview_runtime import solve_preview
        from s2a.gripper_motor_state import preserve_motor_target
        from s2a.libero_simulation import current_observation, make_environment, source_pose

        folder = next(CASE.glob("simulation/*/*/manifest.json")).parent
        refs = {
            name: reference(folder / name)
            for name in ("manifest.json", "controller.json", "steps.jsonl", "chunks.json", "result.json")
        }
        refs["endpoints.jsonl"] = reference(CASE / "endpoints.jsonl")
        result["input_refs"] = refs
        result["source_ref"] = reference(Path(__file__))
        manifest, controller = read(refs["manifest.json"]), read(refs["controller.json"])
        rows = [json.loads(line) for line in Path(refs["steps.jsonl"]["path"]).read_text().splitlines()]
        receipts = [json.loads(line) for line in (CASE / "endpoints.jsonl").read_text().splitlines()]
        chunks = read(refs["chunks.json"])
        assert len(rows) == 533 and len(receipts) == 534
        env, _ = make_environment(manifest["runtime_config"], ROOT, manifest["record"], 0)
        obs = current_observation(env)
        env.robots[0].gripper.current_action = preserve_motor_target(env.robots[0].gripper.current_action)
        for step in range(534):
            raw = np.asarray(env.sim.get_state().flatten())
            digest = hashlib.sha256(raw.tobytes()).hexdigest()
            assert digest == receipts[step]["sim_state_sha256"], f"Raw state mismatch at endpoint {step}"
            if step == 533:
                break
            row = rows[step]
            np.testing.assert_allclose(source_pose(obs), row["measured_source"], atol=1e-9, rtol=0)
            env.step(np.asarray(row["command"]))
            obs = current_observation(env)
            result["replay_steps"] += 1
            if (step + 1) % 100 == 0:
                print(json.dumps(dict(replayed=step + 1)), flush=True)
        import mujoco

        robot = env.robots[0]
        model, data = env.sim.model._model, env.sim.data._data
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.controller.eef_name)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        full_j = np.vstack((jp, jr))
        dofs = robot._ref_joint_vel_indexes
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
        mapping = np.asarray(controller["source_from_site"])
        h = np.asarray(json.loads(Path(manifest["runtime_config"]["tool_contract"]).read_text())["source_from_tcp"])
        chunk = chunks[-1]
        assert chunk["step"] == 528
        old_solver = coupled_pose_preview.solve_qp

        def capture(hessian, linear, constraints, bounds, initial):
            np.savez(
                ROOT / "failed_qp.npz",
                hessian=hessian,
                linear=linear,
                constraints=constraints,
                bounds=bounds,
                initial=initial,
            )
            return old_solver(hessian, linear, constraints, bounds, initial)

        coupled_pose_preview.solve_qp = capture
        try:
            solve_preview(
                mass[np.ix_(dofs, dofs)],
                full_j[:, dofs],
                pose,
                full_j @ data.qvel,
                np.asarray(chunk["effective_targets_tcp"]) @ np.linalg.inv(h) @ mapping,
                np.asarray(chunk["anchor_tcp"]) @ np.linalg.inv(h) @ mapping,
                533 - chunk["step"],
                np.asarray(rows[-1]["command"])[:6],
                controller["contract"],
                innovation=np.asarray(rows[-1]["coupled_response_history"]["innovation_after"]),
            )
        except RuntimeError as error:
            assert str(error) == "Dependent QP blocking constraint; original state retained", str(error)
            result["reproduced_error"] = str(error)
        else:
            raise AssertionError("Original failure did not reproduce")
        finally:
            coupled_pose_preview.solve_qp = old_solver
        result.update(
            complete=True,
            passed=True,
            matched_raw_endpoints=534,
            failed_step=533,
            qp_ref=reference(ROOT / "failed_qp.npz"),
            scope="deterministic command replay and QP capture only",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        write_json(ROOT / "result.json", result)


if __name__ == "__main__":
    main()
