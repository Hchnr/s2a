"""225 reconstruct actual protocol instances with all integration entrypoints blocked."""

import inspect
import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from run_matched_oracle_case import CONTROL, CONTROL_SHA

from s2a.artifacts import REPO, sha256
from s2a.durable_stage_supervisor import publish
from s2a.native_environment import restore_cv2_environment, strict_native_environment


def main():
    strict_native_environment()
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import configure_simulator, make_environment

    root = REPO / "outputs/actual_controller_coupling_probe_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/libero_simulation.py",
        "src/s2a/native_environment.py",
        "plans/v0.0.1/225_actual_controller_coupling_probe_20261026.md",
        CONTROL,
    ):
        e.read(path)
    closure = e.json("outputs/preview_control_serialized_20261026/closure_verified.json")
    assert closure["complete"] and closure["owner_normal_wait_verified"]
    for path, signature in closure["source_sha256"].items():
        e.check(path, signature)
    root.mkdir()
    results = []
    calls = {name: 0 for name in ("mj_step", "mj_step1", "mj_step2")}
    try:
        for index, step in ((15, 550), (6, 220), (14, 60)):
            directory = REPO / f"outputs/preview_control_20261026/case{index:02d}"
            case = next((directory / "simulation").glob("*/*"))
            manifest = e.json(case / "manifest.json")
            expected = e.json(case / "controller.json")["contract"]
            config, record = manifest["runtime_config"], manifest["record"]
            assert config["seed"] == 42 and config["refresh_controller_after_restore"]
            output = root / f"case{index:02d}"
            output.mkdir()
            configure_simulator(config, output)
            import mujoco

            original = {name: getattr(mujoco, name) for name in calls}

            def blocked(name):
                def reject(*args, **kwargs):
                    calls[name] += 1
                    raise RuntimeError(f"Integration forbidden during controller probe: {name}")

                return reject

            for name in calls:
                setattr(mujoco, name, blocked(name))
            env = None
            try:
                env, _ = make_environment(config, output, record, 0)
                ref = e.json(directory / "collection_result.json")["arrays_ref"]
                e.check(ref["path"], ref["sha256"])
                with np.load(ref["path"], allow_pickle=False) as archive:
                    state = archive["sim_states"][step].copy()
                    pose = archive["controller_site_pose"][step].copy()
                env.set_init_state(state)
                controller = env.robots[0].controller
                before = env.sim.get_state().flatten().copy()
                np.testing.assert_array_equal(before, state)
                controller.update(force=True)
                for key in ("kp", "kd", "input_min", "input_max", "output_min", "output_max"):
                    np.testing.assert_allclose(getattr(controller, key), expected[key], atol=1e-12, rtol=0)
                assert (
                    bool(controller.uncoupling) and bool(controller.use_delta) and controller.impedance_mode == "fixed"
                )
                np.testing.assert_allclose(controller.ee_pos, pose[:3, 3], atol=1e-9, rtol=0)
                np.testing.assert_allclose(controller.ee_ori_mat, pose[:3, :3], atol=1e-9, rtol=0)
                model, data = env.sim.model._model, env.sim.data._data
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, expected["controller_site"])
                assert sid >= 0
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jp, jr, sid)
                full_j = np.vstack([jp, jr])
                twist = np.r_[controller.ee_pos_vel, controller.ee_ori_vel]
                np.testing.assert_allclose(full_j @ data.qvel, twist, atol=1e-8, rtol=0)
                dofs = env.robots[0]._ref_joint_vel_indexes
                full_mass = np.zeros((model.nv, model.nv))
                mujoco.mj_fullM(model, full_mass, data.qM)
                mass = full_mass[np.ix_(dofs, dofs)]
                jacobian = full_j[:, dofs]
                np.testing.assert_allclose(controller.mass_matrix, mass, atol=1e-9, rtol=0)
                np.testing.assert_allclose(controller.J_full, jacobian, atol=1e-9, rtol=0)
                np.testing.assert_array_equal(before, env.sim.get_state().flatten())
                controller_path = str(Path(inspect.getfile(type(controller))).resolve())
                assert sha256(controller_path) == manifest["execution_sha256"][controller_path]
                e.read(controller_path)
                results.append(
                    dict(
                        case=index,
                        step=step,
                        task_id=record["task_id"],
                        actual_uncoupling=bool(controller.uncoupling),
                        actual_use_delta=bool(controller.use_delta),
                        actual_impedance_mode=controller.impedance_mode,
                        max_mass_difference=float(np.max(np.abs(controller.mass_matrix - mass))),
                        max_jacobian_difference=float(np.max(np.abs(controller.J_full - jacobian))),
                        state_unchanged=True,
                        mass=mass.tolist(),
                        jacobian=jacobian.tolist(),
                    )
                )
            finally:
                if env is not None:
                    env.close()
                for name, value in original.items():
                    setattr(mujoco, name, value)
        assert set(row["task_id"] for row in results) == {0, 1, 2}
        assert not any(calls.values())
        e.recheck()
        publish(
            root / "verified.json",
            dict(
                complete=True,
                instances=results,
                integration_attempts=calls,
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                seed=42,
                source_sha256=e.hashes,
                evidence_level="direct fields of same-protocol reconstructed instances;no historical process-field recording",
                control_qualified=False,
            ),
        )
        print("225 three task instances: actual uncoupling=True; M/J matched; no integration calls")
    except Exception as error:
        publish(
            root / "failure.json",
            dict(
                complete=False,
                instances=results,
                integration_attempts=calls,
                physics_steps=0,
                error_type=type(error).__name__,
                error=str(error),
                source_sha256=e.hashes,
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
