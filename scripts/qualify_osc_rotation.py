"""CPU-only independent MuJoCo qualification of rotational OSC conversion."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import load_config, sha256
from s2a.geometry import inverse, pose_error
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.osc_rotation import bounded_osc_rotation_inverse


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, default=lambda x: x.tolist(), indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def motion_reference(initial_tcp, step, motion, dt):
    """Return an absolute world target; only a fixed initial pose and time enter."""
    wave = np.sin(2 * np.pi * (step + 1) * dt / motion["period_s"])
    rotation_wave = wave
    if "step_segments" in motion:
        rotation_wave = next(value for start, value in reversed(motion["step_segments"]) if step >= start)
    position_axis = np.asarray(motion["position_axis"], dtype=float)
    rotation_axis = np.asarray(motion["rotation_axis"], dtype=float)
    position_axis /= np.linalg.norm(position_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    desired = initial_tcp.copy()
    desired[:3, 3] += position_axis * motion["position_amplitude_m"] * wave
    angle = rotation_axis * motion["rotation_amplitude_rad"] * rotation_wave
    desired[:3, :3] = Rotation.from_rotvec(angle).as_matrix() @ initial_tcp[:3, :3]
    return desired


def fingerprints(config_path, config):
    template = json.loads(Path(config["policy_template"]).read_text())
    package = Path("outputs/libero_env311/lib/python3.11/site-packages/robosuite")
    paths = [
        Path(config_path),
        Path(config["policy_template"]),
        Path(config["fixture"]),
        Path(template["tool_contract"]),
        Path(template["libero_config"]),
        Path(__file__),
        Path("src/s2a/osc.py"),
        Path("src/s2a/osc_rotation.py"),
        Path("src/s2a/libero_simulation.py"),
        Path("src/s2a/geometry.py"),
        package / "controllers/osc.py",
        package / "controllers/base_controller.py",
        package / "utils/control_utils.py",
        package / "robots/manipulator.py",
        package / "models/grippers/panda_gripper.py",
    ]
    return {str(p.resolve()): sha256(p) for p in paths}


def check_inputs(expected):
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"qualification input changed: {path}")


def run_worker(args, config, root):
    from s2a.libero_simulation import (
        aperture,
        controller_contract,
        current_observation,
        make_environment,
        measured_twist,
        sha256_bytes,
        source_pose,
    )

    matrix = json.loads((root / "matrix_manifest.json").read_text())
    check_inputs(matrix["files_sha256"])
    fixture = json.loads(Path(config["fixture"]).read_text())
    record = next(r for r in fixture["records"] if r["name"] == args.record)
    motion = next(m for m in config["motions"] if m["name"] == args.motion)
    dest = root / args.record / args.motion / args.method
    dest.mkdir(parents=True, exist_ok=False)
    template = json.loads(Path(config["policy_template"]).read_text())
    runtime = dict(
        template,
        seed=args.seed,
        task_id=record["task_id"],
        control_fps=config["control_fps"],
        camera_observations=False,
        refresh_controller_after_restore=True,
        record_video=False,
    )
    h = np.asarray(json.loads(Path(template["tool_contract"]).read_text())["source_from_tcp"])
    write_new(
        dest / "manifest.json",
        dict(
            config=config,
            record_name=record["name"],
            initial_state_sha256=record["initial_state_sha256"],
            method=args.method,
            motion=motion,
            files_sha256=matrix["files_sha256"],
            seed=args.seed,
            pid=os.getpid(),
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            runtime_versions={p: importlib.metadata.version(p) for p in ("mujoco", "robosuite", "numpy", "scipy")},
            no_task_success_or_contact_read=True,
        ),
    )
    started = time.monotonic()
    env, _ = make_environment(runtime, dest, record, 0)
    rows, error = [], None
    try:
        obs = current_observation(env)
        controller, contract = controller_contract(env)
        dt = 1 / config["control_fps"]
        if not np.isclose(env.env.control_timestep, dt):
            raise ValueError("controller clock differs from the qualification protocol")
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        response = OSCStateToAction(
            interface,
            contract["kp"],
            contract["kd"],
            dt,
            compensate=True,
            inverse_weight=config["inverse_weight"],
            max_command_rate=config["max_command_rate"],
        )
        baseline = OSCStateToAction(
            interface,
            contract["kp"],
            contract["kd"],
            dt,
            compensate=False,
            max_command_rate=config["max_command_rate"],
        )
        initial_source = source_pose(obs)
        initial_tcp = initial_source @ h
        simulation_state = np.asarray(env.sim.get_state().flatten()).copy()
        gripper = env.robots[0].gripper
        initial_gripper_action = np.asarray(gripper.current_action).copy()
        gripper_indices = [env.sim.model.actuator_name2id(name) for name in gripper.actuators]
        held_gripper_motor_target = np.broadcast_to(initial_gripper_action, (len(gripper_indices),)).copy()
        control_range = env.sim.model.actuator_ctrlrange[gripper_indices]
        expected_gripper_control = control_range.mean(axis=1) + 0.5 * np.diff(control_range, axis=1)[:, 0] * (
            held_gripper_motor_target
        )
        write_new(
            dest / "initial.json",
            dict(
                fixture_initial_state_sha256=record["initial_state_sha256"],
                restored_simulation_state_sha256=sha256_bytes(simulation_state.tobytes()),
                source_pose=initial_source,
                tcp_pose=initial_tcp,
                measured_twist=measured_twist(env),
                controller_goal_rotation=controller.goal_ori.copy(),
                controller_feedback_rotation=controller.ee_ori_mat.copy(),
                gripper_current_action=initial_gripper_action,
                held_gripper_motor_target=held_gripper_motor_target,
                expected_gripper_control=expected_gripper_control,
                initial_gripper_control=env.sim.data.ctrl[gripper_indices].copy(),
                controller=contract,
                tool_source_from_tcp=h,
            ),
        )
        np.savez_compressed(dest / "initial_physics.npz", state=simulation_state, qpos=env.sim.data.qpos.copy())
        previous_command = np.zeros(6)
        with (dest / "steps.jsonl").open("x") as stream:
            for step in range(config["steps"]):
                measured = source_pose(obs)
                desired_tcp = motion_reference(initial_tcp, step, motion, dt)
                desired = desired_tcp @ inverse(h)
                velocity = measured_twist(env)
                goal_before = controller.goal_ori.copy()
                base = response.convert(measured, desired, velocity)
                command = base["command"].copy()
                rotation_diagnostic = None
                if args.method == "baseline_rotation":
                    old = baseline.convert(measured, desired, velocity)
                    command[3:] = old["command"][3:]
                elif args.method == "bounded_rotation":
                    rotation_diagnostic = bounded_osc_rotation_inverse(
                        measured[:3, :3],
                        desired[:3, :3],
                        velocity[3:],
                        interface=interface,
                        kp=contract["kp"][3:],
                        kd=contract["kd"][3:],
                        dt=dt,
                        previous_command=previous_command[3:],
                        current_goal_rotation=goal_before,
                        max_command_rate=config["max_command_rate"],
                        integration_steps=config["rotation_integration_steps"],
                        max_nfev=config["rotation_max_nfev"],
                        zero_goal_epsilon_rad=config["zero_goal_epsilon_rad"],
                    )
                    command[3:] = rotation_diagnostic["command"]
                if np.any(command < interface.input_min - 1e-12) or np.any(command > interface.input_max + 1e-12):
                    raise ValueError("sent command exceeds the controller affine limits")
                if np.any(np.abs(command - previous_command) > config["max_command_rate"] * dt + 1e-12):
                    raise ValueError("sent command exceeds the frozen rate limit")
                response.previous_command = command.copy()
                baseline.previous_command = command.copy()
                before_time = float(env.sim.data.time)
                env.step(np.r_[command, 0.0])
                obs = current_observation(env)
                actual_dt = float(env.sim.data.time) - before_time
                if not np.isclose(actual_dt, dt, atol=1e-9, rtol=0):
                    raise ValueError("physical time interval differs from the planned interval")
                if not np.array_equal(gripper.current_action, held_gripper_motor_target):
                    raise ValueError("zero gripper command changed its retained motor target")
                actual_gripper_control = env.sim.data.ctrl[gripper_indices].copy()
                np.testing.assert_allclose(actual_gripper_control, expected_gripper_control, atol=1e-12, rtol=0)
                after = source_pose(obs)
                row = dict(
                    step=step,
                    actual_dt=actual_dt,
                    measured_source=measured,
                    desired_source=desired,
                    desired_tcp=desired_tcp,
                    after_source=after,
                    after_tcp=after @ h,
                    world_twist=velocity,
                    command=np.r_[command, 0.0],
                    physical_delta=controller.scale_action(command).copy(),
                    controller_goal_rotation_before=goal_before,
                    controller_goal_rotation_after=controller.goal_ori.copy(),
                    controller_feedback_rotation=controller.ee_ori_mat.copy(),
                    controller_feedback_position=controller.ee_pos.copy(),
                    controller_goal_position=controller.goal_pos.copy(),
                    opening_after=aperture(obs),
                    gripper_control=actual_gripper_control,
                    tcp_error=pose_error(desired_tcp, after @ h),
                    original_response_diagnostics=base,
                    rotation_diagnostic=rotation_diagnostic,
                    command_at_affine_bound=bool(
                        np.any(np.isclose(command, interface.input_min, atol=1e-7, rtol=0))
                        or np.any(np.isclose(command, interface.input_max, atol=1e-7, rtol=0))
                    ),
                    command_at_rate_bound=bool(
                        np.any(
                            np.isclose(np.abs(command - previous_command), config["max_command_rate"] * dt, atol=1e-7)
                        )
                    ),
                )
                rows.append(row)
                stream.write(json.dumps(row, default=lambda x: x.tolist(), allow_nan=False) + "\n")
                stream.flush()
                previous_command = command.copy()
                if (step + 1) % 30 == 0:
                    print(
                        json.dumps(dict(record=args.record, motion=args.motion, method=args.method, steps=step + 1)),
                        flush=True,
                    )
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    result = dict(
        record=args.record,
        motion=args.motion,
        method=args.method,
        steps=len(rows),
        expected_steps=config["steps"],
        error=error,
        elapsed_s=time.monotonic() - started,
        physical_elapsed_s=sum(r["actual_dt"] for r in rows),
        evidence_level="independent MuJoCo robot execution, task-independent synthetic references",
        task_success_not_scored=True,
        files_sha256={
            str(p.resolve()): sha256(p) for p in (dest / "steps.jsonl", dest / "initial.json", dest / "manifest.json")
        },
    )
    if rows:
        errors = np.asarray([r["tcp_error"] for r in rows])

        def metrics(indices):
            selected = errors[indices]
            position, angle = np.linalg.norm(selected[:, :3], axis=1), np.linalg.norm(selected[:, 3:], axis=1)
            return dict(
                steps=len(selected),
                canonical_tcp_position_rmse_mm=float(1000 * np.sqrt(np.mean(position**2))),
                so3_rmse_rad=float(np.sqrt(np.mean(angle**2))),
                max_position_error_mm=float(1000 * max(position)),
                max_rotation_error_rad=float(max(angle)),
                within_both_tolerances_fraction=float(
                    np.mean((position <= config["position_tolerance_m"]) & (angle <= config["rotation_tolerance_rad"]))
                ),
            )

        result["overall"] = metrics(slice(None))
        result["last40"] = metrics(slice(-40, None))
        if motion["name"] == "large_rotation_step" and len(rows) == config["steps"]:
            result["large_rotation_hold_tail20"] = {
                str(start): metrics(slice(end - 20, end)) for start, end in ((20, 80), (80, 140), (140, 180))
            }
        result["affine_bound_steps"] = sum(r["command_at_affine_bound"] for r in rows)
        result["rate_bound_steps"] = sum(r["command_at_rate_bound"] for r in rows)
        result["zero_physical_rotation_steps"] = sum(np.all(r["physical_delta"][3:] == 0) for r in rows)
        result["retained_gripper_control_verified"] = True
    check_inputs(matrix["files_sha256"])
    write_new(dest / "result.json", result)
    if error:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/osc_rotation_qualification_20261021.json")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--record")
    parser.add_argument("--motion")
    parser.add_argument("--method")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("qualification requires explicit CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=''")
    config, root = load_config(args.config)
    if args.seed != config["seed"] or sha256(config["fixture"]) != config["fixture_sha256"]:
        raise ValueError("seed or frozen fixture SHA differs")
    if args.worker:
        return run_worker(args, config, root)
    # load_config creates the output directory. Require it to be empty, and
    # exclusively create each matrix artifact so a completed run cannot be
    # silently reused or overwritten.
    if any(root.iterdir()):
        raise FileExistsError(f"qualification output directory is not empty: {root}")
    inputs = fingerprints(args.config, config)
    write_new(root / "frozen_config.json", config)
    write_new(root / "matrix_manifest.json", dict(files_sha256=inputs, seed=args.seed, created_unix=time.time()))
    worker_env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        LP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    results = []

    def execute(record, motion, method):
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        command = [
            str(Path(config["simulation_python"]).resolve()),
            str(Path(__file__).resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--seed",
            str(args.seed),
            "--worker",
            "--record",
            record,
            "--motion",
            motion["name"],
            "--method",
            method,
        ]
        with (logs / f"{record}_{motion['name']}_{method}.log").open("x") as stream:
            process = subprocess.Popen(command, env=worker_env, stdout=stream, stderr=subprocess.STDOUT)
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"qualification worker failed: {record}/{motion['name']}/{method}, exit {returncode}")
        path = root / record / motion["name"] / method / "result.json"
        result = json.loads(path.read_text())
        if result["error"] or result["steps"] != config["steps"]:
            raise ValueError(f"incomplete qualification result: {path}")
        return dict(path=str(path.resolve()), sha256=sha256(path), **result)

    for record in config["records"]:
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            pending = {
                pool.submit(execute, record, motion, method): (motion["name"], method)
                for motion in config["motions"]
                for method in config["methods"]
            }
            for future in as_completed(pending):
                try:
                    result = future.result()
                except Exception:
                    for waiting in pending:
                        waiting.cancel()
                    raise
                results.append(result)
                print(
                    json.dumps(
                        dict(
                            completed=len(results),
                            record=record,
                            motion=result["motion"],
                            method=result["method"],
                            overall=result["overall"],
                        )
                    ),
                    flush=True,
                )
    check_inputs(inputs)
    expected = len(config["records"]) * len(config["motions"]) * len(config["methods"])
    summary = dict(
        seed=args.seed,
        expected=expected,
        completed=len(results),
        complete=len(results) == expected,
        results=results,
        inputs_sha256=inputs,
        limitations="This qualifies synthetic reference tracking in actual robot physics; it does not establish WAM task success or generalize contact dynamics.",
    )
    write_new(root / "qualification_summary.json", summary)
    print(json.dumps(dict(complete=summary["complete"], summary=str(root / "qualification_summary.json"))), flush=True)


if __name__ == "__main__":
    main()
