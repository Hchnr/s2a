"""225 reconstruct actual protocol instances with all integration entrypoints blocked."""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from current_coupled_independent import response, verify_inverse
from run_matched_oracle_case import CONTROL, CONTROL_SHA

from s2a.artifacts import REPO, sha256
from s2a.current_coupled_response import build_response, solve_translation
from s2a.current_coupled_state import read_current_dynamics
from s2a.durable_stage_supervisor import publish
from s2a.native_environment import restore_cv2_environment, strict_native_environment


def main():
    strict_native_environment()
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import configure_simulator, make_environment

    root = REPO / "outputs/current_coupled_reader_admission_20261026"
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
    for path in (
        "scripts/probe_actual_controller_coupling.py",
        "outputs/actual_controller_coupling_probe_20261026/failure.json",
        "outputs/controller_probe_step_semantics_20261026/engine_forward_2.3.7.c",
        "outputs/controller_probe_step_semantics_20261026/source.json",
        "plans/v0.0.1/226_nonintegrating_controller_probe_20261026.md",
    ):
        e.read(path)
    closure = e.json("outputs/preview_control_serialized_20261026/closure_verified.json")
    assert closure["complete"] and closure["owner_normal_wait_verified"]
    for path, signature in closure["source_sha256"].items():
        e.check(path, signature)
    root.mkdir()
    for path in (
        "src/s2a/current_coupled_state.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/coupled_response_history.py",
        "scripts/current_coupled_independent.py",
        "tests/test_current_coupled_response.py",
        "tests/test_coupled_response_history.py",
        "tests/test_coupled_rotation_goal.py",
        "plans/v0.0.1/228_current_coupled_inverse_20261026.md",
        "plans/v0.0.1/229_coupled_state_reader_admission_20261026.md",
    ):
        e.read(path)
    results = []
    expected_tasks = set()
    calls = {name: 0 for name in ("mj_step", "mj_step1", "mj_step2")}
    try:
        for index, step in ((15, 550), (6, 220), (14, 60)):
            directory = REPO / f"outputs/preview_control_20261026/case{index:02d}"
            case = next((directory / "simulation").glob("*/*"))
            manifest = e.json(case / "manifest.json")
            expected = e.json(case / "controller.json")["contract"]
            config, record = manifest["runtime_config"], manifest["record"]
            expected_tasks.add(record["task_id"])
            rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
            packet = {key: np.asarray(value).copy() for key, value in rows[step]["qualified_forward_input"].items()}
            candidate = np.asarray(rows[step]["command"]).copy()
            assert config["seed"] == 42 and config["refresh_controller_after_restore"]
            output = root / f"case{index:02d}"
            output.mkdir()
            configure_simulator(config, output)
            import mujoco

            assert mujoco.__version__ == "2.3.7"
            original = {name: getattr(mujoco, name) for name in calls}
            e.read(sys.modules[original["mj_step1"].__module__].__file__)
            for library in Path(mujoco.__file__).parent.glob("libmujoco.so*"):
                e.read(library)

            def blocked(name):
                def reject(*args, **kwargs):
                    calls[name] += 1
                    if name != "mj_step1":
                        raise RuntimeError(f"Integration forbidden during controller probe: {name}")
                    data = args[1]
                    before = np.r_[data.time, data.qpos.copy(), data.qvel.copy(), data.act.copy()]
                    controls = data.ctrl.copy()
                    result = original[name](*args, **kwargs)
                    np.testing.assert_array_equal(before, np.r_[data.time, data.qpos, data.qvel, data.act])
                    np.testing.assert_array_equal(controls, data.ctrl)
                    return result

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
                    actual_goal_after = archive["controller_goal_ori"][step + 1].copy()
                env.set_init_state(state)
                controller = env.robots[0].controller
                before = env.sim.get_state().flatten().copy()
                np.testing.assert_array_equal(before, state)
                controller.goal_pos = packet["x_previous_goal_pos"][0].copy()
                controller.goal_ori = packet["x_previous_goal_ori"][0].copy()
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
                dynamics = read_current_dynamics(env, packet, candidate)
                np.testing.assert_allclose(dynamics["mass"], controller.mass_matrix, atol=1e-9, rtol=0)
                np.testing.assert_allclose(dynamics["jacobian"], controller.J_full, atol=1e-9, rtol=0)
                np.testing.assert_allclose(dynamics["actual_rotation_goal"], actual_goal_after, atol=1e-8, rtol=0)
                stale = {key: value.copy() for key, value in packet.items()}
                stale["x_joint_position"][0, 0] += 0.001
                try:
                    read_current_dynamics(env, stale, candidate)
                except AssertionError:
                    stale_rejected = True
                else:
                    raise AssertionError("Stale joint packet accepted")
                old_flag = controller.uncoupling
                controller.uncoupling = False
                try:
                    try:
                        read_current_dynamics(env, packet, candidate)
                    except ValueError:
                        unsupported_rejected = True
                    else:
                        raise AssertionError("Unsupported controller accepted")
                finally:
                    controller.uncoupling = old_flag
                coupled = build_response(
                    dynamics["mass"],
                    dynamics["jacobian"],
                    dynamics["kp"],
                    dynamics["kd"],
                    dynamics["relative_rotation_goal"],
                )
                independent = response(
                    dynamics["mass"],
                    dynamics["jacobian"],
                    dynamics["kp"],
                    dynamics["kd"],
                    dynamics["relative_rotation_goal"],
                )
                for key in independent:
                    np.testing.assert_allclose(coupled[key], independent[key], atol=1e-8, rtol=0)
                target = np.asarray(rows[step]["desired_site"])[:3, 3]
                target_velocity = np.asarray(rows[step]["site_velocity_reference"]["velocity"])
                previous = packet["x_previous_command"][0, :3]
                bias = np.zeros((2, 3))
                solution = solve_translation(
                    pose[:3, 3],
                    packet["x_site_twist"][0],
                    target,
                    target_velocity,
                    previous,
                    expected,
                    coupled,
                    innovation=bias,
                )
                saved = json.loads(json.dumps(solution, default=lambda value: value.tolist(), allow_nan=False))
                certificate = verify_inverse(
                    saved,
                    pose[:3, 3],
                    packet["x_site_twist"][0],
                    target,
                    target_velocity,
                    previous,
                    expected,
                    independent,
                    bias,
                )
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
                        stale_joint_packet_rejected=stale_rejected,
                        unsupported_controller_rejected=unsupported_rejected,
                        native_rotation_goal_max_difference=float(
                            np.max(np.abs(dynamics["actual_rotation_goal"] - actual_goal_after))
                        ),
                        independent_inverse_certificate=certificate,
                        source_dynamics=json.loads(
                            json.dumps(dynamics, default=lambda value: value.tolist(), allow_nan=False)
                        ),
                        inverse_solution=saved,
                        mass=mass.tolist(),
                        jacobian=jacobian.tolist(),
                    )
                )
            finally:
                if env is not None:
                    env.close()
                for name, value in original.items():
                    setattr(mujoco, name, value)
        assert len(expected_tasks) == 3 and set(row["task_id"] for row in results) == expected_tasks
        assert calls["mj_step"] == calls["mj_step2"] == 0 and calls["mj_step1"] > 0
        e.recheck()
        publish(
            root / "verified.json",
            dict(
                complete=True,
                instances=results,
                current_dynamics_reader_verified=True,
                simulation_entry_calls=calls,
                integration_attempts=dict(mj_step=calls["mj_step"], mj_step2=calls["mj_step2"]),
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                seed=42,
                source_sha256=e.hashes,
                evidence_level="direct fields of same-protocol reconstructed instances;no historical process-field recording",
                control_qualified=False,
            ),
        )
        print(
            "229 native current M/J/rotation goal and inverse JSON KKT verified; stale/unsupported rejected; no integration"
        )
    except Exception as error:
        publish(
            root / "failure.json",
            dict(
                complete=False,
                instances=results,
                current_dynamics_reader_verified=False,
                simulation_entry_calls=calls,
                integration_attempts=dict(mj_step=calls["mj_step"], mj_step2=calls["mj_step2"]),
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
