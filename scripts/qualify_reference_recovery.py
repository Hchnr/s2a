"""CPU MuJoCo qualification of recovery from an earlier blocked TCP reference."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from s2a.artifacts import load_config, sha256
from s2a.geometry import inverse, pose_error
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.reference_governor import ReferenceGovernor, ReferenceGovernorConfig
from s2a.reference_governor_v2 import RecoveringReferenceGovernor


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, default=lambda x: x.tolist(), indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


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
        *(
            Path("src/s2a") / name
            for name in (
                "artifacts.py",
                "osc.py",
                "geometry.py",
                "libero_simulation.py",
                "reference_governor.py",
                "reference_governor_v2.py",
            )
        ),
        *(
            package / name
            for name in (
                "controllers/osc.py",
                "controllers/base_controller.py",
                "utils/control_utils.py",
                "robots/manipulator.py",
                "models/grippers/panda_gripper.py",
            )
        ),
    ]
    return {str(p.resolve()): sha256(p) for p in paths}


def check_inputs(expected):
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"qualification input changed: {path}")


def generated_reference(anchor, translation, count):
    """Snapshot a world-frame straight path, without later measured-state input."""
    targets = np.repeat(anchor[None], count, axis=0)
    targets[:, :3, 3] += np.arange(1, count + 1)[:, None] / count * np.asarray(translation)
    return targets


def summarize_rows(rows, references, config):
    query, dt = config["requery_step"], 1 / config["control_fps"]
    down, recovery = rows[:query], rows[query:]
    diagnostic = config["diagnostics"]
    tail = down[-diagnostic["down_tail_steps"] :]
    tail_positions = np.asarray([r["after_tcp"] for r in tail])[:, :3, 3]
    tail_span = float(np.linalg.norm(np.ptp(tail_positions, axis=0)))
    median_scale = float(np.median([r["governor"]["tracking_scale"] for r in tail]))
    median_error = float(np.median([r["governor"]["preceding_position_error_m"] for r in tail]))
    tail_progress = float(tail[-1]["governor"]["model_time_s"] - tail[0]["governor"]["model_time_s"])
    blocked = bool(
        tail_span <= diagnostic["blocked_tail_max_position_span_m"]
        and median_scale <= diagnostic["blocked_tail_max_median_tracking_scale"]
        and median_error >= diagnostic["blocked_tail_min_median_preceding_error_m"]
        and tail_progress <= diagnostic["blocked_tail_max_model_progress_s"]
    )
    bridge_exit = next((i for i, r in enumerate(recovery) if not r["governor"]["bridging"]), None)
    final_target = np.asarray(references[-1]["targets_tcp"])[-1]
    final_error = pose_error(final_target, np.asarray(rows[-1]["after_tcp"]))
    errors = np.asarray([r["tcp_error"] for r in rows])
    success = bool(
        np.linalg.norm(final_error[:3]) <= diagnostic["endpoint_position_tolerance_m"]
        and np.linalg.norm(final_error[3:]) <= diagnostic["endpoint_rotation_tolerance_rad"]
        and recovery[-1]["governor"]["complete"]
    )
    return dict(
        first_down=dict(
            observed_blockage=blocked,
            global_unreachability_proven=False,
            blockage_definition="slow reference progression and measured TCP plateau; not strict zero-scale lock",
            tail_step_range=[tail[0]["step"], tail[-1]["step"]],
            tail_position_span_m=tail_span,
            tail_median_tracking_scale=median_scale,
            tail_min_tracking_scale=float(min(r["governor"]["tracking_scale"] for r in tail)),
            tail_max_tracking_scale=float(max(r["governor"]["tracking_scale"] for r in tail)),
            tail_exact_zero_scale_steps=sum(r["governor"]["tracking_scale"] == 0 for r in tail),
            tail_median_preceding_error_m=median_error,
            tail_preceding_position_errors_m=[r["governor"]["preceding_position_error_m"] for r in tail],
            tail_preceding_rotation_errors_rad=[r["governor"]["preceding_rotation_error_rad"] for r in tail],
            tail_model_times_s=[r["governor"]["model_time_s"] for r in tail],
            tail_complete=[r["governor"]["complete"] for r in tail],
            tail_model_progress_s=tail_progress,
            last_model_time_s=down[-1]["governor"]["model_time_s"],
            last_complete=down[-1]["governor"]["complete"],
            generated_endpoint_error_m=float(
                np.linalg.norm(
                    pose_error(np.asarray(references[0]["targets_tcp"])[-1], np.asarray(down[-1]["after_tcp"]))[:3]
                )
            ),
        ),
        after_requery=dict(
            bridge_recovery_time_s=None if bridge_exit is None else (bridge_exit + 1) * dt,
            bridge_exit_step=None if bridge_exit is None else query + bridge_exit,
            bridge_recovery_advanced_steps=sum(r["governor"].get("bridge_recovery_advanced", False) for r in recovery),
            final_model_time_s=recovery[-1]["governor"]["model_time_s"],
            final_retimed_clock_s=recovery[-1]["governor"]["retimed_clock_s"],
            final_complete=recovery[-1]["governor"]["complete"],
            final_generated_endpoint_position_error_m=float(np.linalg.norm(final_error[:3])),
            final_generated_endpoint_rotation_error_rad=float(np.linalg.norm(final_error[3:])),
            actual_upward_displacement_m=float(
                np.asarray(rows[-1]["after_tcp"])[2, 3] - np.asarray(references[-1]["anchor_tcp"])[2, 3]
            ),
            generated_endpoint_reached=success,
            recovery_from_observed_blockage_qualified=bool(blocked and success and bridge_exit is not None),
        ),
        tcp_position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        tcp_rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        affine_bound_steps=sum(r["command_at_affine_bound"] for r in rows),
        rate_bound_steps=sum(r["command_at_rate_bound"] for r in rows),
    )


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

    matrix = json.loads((root / "input_manifest.json").read_text())
    check_inputs(matrix["files_sha256"])
    fixture = json.loads(Path(config["fixture"]).read_text())
    record = next(r for r in fixture["records"] if r["name"] == args.record)
    dest = root / args.record / args.method
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
            record_name=args.record,
            method=args.method,
            task_id=record["task_id"],
            initial_state_sha256=record["initial_state_sha256"],
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
    rows, references, error = [], [], None
    try:
        obs = current_observation(env)
        controller, contract = controller_contract(env)
        dt = 1 / config["control_fps"]
        np.testing.assert_allclose(env.env.control_timestep, dt, atol=1e-12, rtol=0)
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        converter = OSCStateToAction(
            interface,
            contract["kp"],
            contract["kd"],
            dt,
            compensate=True,
            inverse_weight=config["inverse_weight"],
            max_command_rate=config["max_command_rate"],
        )
        governor_type = ReferenceGovernor if args.method == "governor_v1" else RecoveringReferenceGovernor
        governor = governor_type(config=ReferenceGovernorConfig(**config["governor"]))
        initial_source, initial_opening = source_pose(obs), aperture(obs)
        initial_tcp = initial_source @ h
        state = np.asarray(env.sim.get_state().flatten()).copy()
        gripper = env.robots[0].gripper
        initial_gripper_action = np.asarray(gripper.current_action).copy()
        indices = [env.sim.model.actuator_name2id(name) for name in gripper.actuators]
        held_motor_target = np.broadcast_to(initial_gripper_action, (len(indices),)).copy()
        ranges = env.sim.model.actuator_ctrlrange[indices]
        expected_motor_ctrl = ranges.mean(axis=1) + 0.5 * np.diff(ranges, axis=1)[:, 0] * held_motor_target
        write_new(
            dest / "initial.json",
            dict(
                fixture_initial_state_sha256=record["initial_state_sha256"],
                restored_simulation_state_sha256=sha256_bytes(state.tobytes()),
                source_pose=initial_source,
                tcp_pose=initial_tcp,
                world_twist=measured_twist(env),
                tool_source_from_tcp=h,
                controller=contract,
                controller_goal_position=controller.goal_pos.copy(),
                controller_goal_rotation=controller.goal_ori.copy(),
                controller_feedback_position=controller.ee_pos.copy(),
                controller_feedback_rotation=controller.ee_ori_mat.copy(),
                initial_opening=initial_opening,
                initial_gripper_current_action=initial_gripper_action,
                held_gripper_motor_target=held_motor_target,
                expected_gripper_control=expected_motor_ctrl,
                initial_gripper_control=env.sim.data.ctrl[indices].copy(),
            ),
        )
        np.savez_compressed(dest / "initial_physics.npz", state=state, qpos=env.sim.data.qpos.copy())
        previous_command = np.zeros(6)
        with (dest / "steps.jsonl").open("x") as stream:
            for step in range(config["steps"]):
                measured = source_pose(obs)
                measured_tcp = measured @ h
                # The only queries are the pre-registered initial query and step100.
                if step in (0, config["requery_step"]):
                    translation = config["initial_world_translation_m" if step == 0 else "requery_world_translation_m"]
                    anchor = initial_tcp.copy() if step == 0 else measured_tcp.copy()
                    targets = generated_reference(anchor, translation, config["reference_endpoints"])
                    openings = np.full(config["reference_endpoints"], initial_opening)
                    admitted = governor.set_reference(
                        anchor,
                        targets,
                        openings,
                        anchor_opening=aperture(obs),
                        model_dt=config["reference_duration_s"] / len(targets),
                    )
                    reference = dict(
                        query_step=step,
                        anchor_tcp=anchor,
                        measured_anchor_opening=aperture(obs),
                        targets_tcp=targets,
                        opening_endpoints=openings,
                        admission=admitted,
                    )
                    references.append(reference)
                    write_new(dest / f"reference_step{step}.json", reference)
                admitted = governor.advance(measured_tcp, dt)
                desired_tcp = admitted["target_tcp"]
                desired_source = desired_tcp @ inverse(h)
                velocity = measured_twist(env)
                converted = converter.convert(measured, desired_source, velocity)
                command = converted["command"].copy()
                if np.any(command < interface.input_min - 1e-12) or np.any(command > interface.input_max + 1e-12):
                    raise ValueError("sent command exceeds affine bounds")
                if np.any(np.abs(command - previous_command) > config["max_command_rate"] * dt + 1e-12):
                    raise ValueError("sent command exceeds frozen rate limit")
                goal_before = dict(position=controller.goal_pos.copy(), rotation=controller.goal_ori.copy())
                before_time = float(env.sim.data.time)
                env.step(np.r_[command, 0.0])
                obs = current_observation(env)
                actual_dt = float(env.sim.data.time) - before_time
                np.testing.assert_allclose(actual_dt, dt, atol=1e-9, rtol=0)
                np.testing.assert_array_equal(gripper.current_action, held_motor_target)
                actual_motor_ctrl = env.sim.data.ctrl[indices].copy()
                np.testing.assert_allclose(actual_motor_ctrl, expected_motor_ctrl, atol=1e-12, rtol=0)
                after_source = source_pose(obs)
                row = dict(
                    step=step,
                    reference_query_step=reference["query_step"],
                    actual_dt=actual_dt,
                    physical_time_before=before_time,
                    measured_source=measured,
                    measured_tcp=measured_tcp,
                    desired_source=desired_source,
                    desired_tcp=desired_tcp,
                    after_source=after_source,
                    after_tcp=after_source @ h,
                    world_twist=velocity,
                    governor=admitted,
                    command=np.r_[command, 0.0],
                    physical_delta=controller.scale_action(command).copy(),
                    controller_goal_before=goal_before,
                    controller_goal_position=controller.goal_pos.copy(),
                    controller_goal_rotation=controller.goal_ori.copy(),
                    controller_feedback_position=controller.ee_pos.copy(),
                    controller_feedback_rotation=controller.ee_ori_mat.copy(),
                    opening_after=aperture(obs),
                    gripper_current_action=np.asarray(gripper.current_action).copy(),
                    gripper_control=actual_motor_ctrl,
                    conversion=converted,
                    tcp_error=pose_error(desired_tcp, after_source @ h),
                    command_at_affine_bound=bool(
                        np.any(np.isclose(command, interface.input_min, atol=1e-7, rtol=0))
                        or np.any(np.isclose(command, interface.input_max, atol=1e-7, rtol=0))
                    ),
                    command_at_rate_bound=bool(
                        np.any(
                            np.isclose(
                                np.abs(command - previous_command), config["max_command_rate"] * dt, atol=1e-7, rtol=0
                            )
                        )
                    ),
                )
                stream.write(json.dumps(row, default=lambda x: x.tolist(), allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                previous_command = command.copy()
                if (step + 1) % 50 == 0:
                    print(json.dumps(dict(record=args.record, method=args.method, steps=step + 1)), flush=True)
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    result = dict(
        record=args.record,
        method=args.method,
        steps=len(rows),
        error=error,
        elapsed_s=time.monotonic() - started,
        physical_elapsed_s=sum(r["actual_dt"] for r in rows),
        evidence_level="independent MuJoCo execution of synthetic references; no WAM task success",
        task_success_not_scored=True,
        reference_queries=[r["query_step"] for r in references],
        retained_gripper_control_verified=error is None and len(rows) == config["steps"],
    )
    if len(rows) == config["steps"]:
        result["metrics"] = summarize_rows(rows, references, config)
    result["files_sha256"] = {str(p.resolve()): sha256(p) for p in dest.iterdir() if p.is_file()}
    check_inputs(matrix["files_sha256"])
    write_new(dest / "result.json", result)
    if error:
        raise RuntimeError(error)


def verify_and_plot(config, root, results):
    """Re-read raw records, recompute geometry/metrics and validate paired inputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verification, all_rows = [], {}
    for result in results:
        dest = root / result["record"] / result["method"]
        raw = [json.loads(line) for line in (dest / "steps.jsonl").read_text().splitlines()]
        references = [json.loads((dest / f"reference_step{s}.json").read_text()) for s in (0, config["requery_step"])]
        initial = json.loads((dest / "initial.json").read_text())
        contract = initial["controller"]
        commands = np.asarray([r["command"] for r in raw])
        assert [r["step"] for r in raw] == list(range(config["steps"]))
        assert result["reference_queries"] == [0, config["requery_step"]]
        assert {r["reference_query_step"] for r in raw[: config["requery_step"]]} == {0}
        assert {r["reference_query_step"] for r in raw[config["requery_step"] :]} == {config["requery_step"]}
        np.testing.assert_array_equal(commands[:, -1], np.zeros(len(raw)))
        assert np.all(commands[:, :6] >= np.asarray(contract["input_min"]) - 1e-12)
        assert np.all(commands[:, :6] <= np.asarray(contract["input_max"]) + 1e-12)
        assert np.max(np.abs(np.diff(np.vstack([np.zeros(7), commands]), axis=0))) <= (
            config["max_command_rate"] / config["control_fps"] + 1e-12
        )
        for reference in references:
            translation = config[
                "initial_world_translation_m" if reference["query_step"] == 0 else "requery_world_translation_m"
            ]
            expected = generated_reference(
                np.asarray(reference["anchor_tcp"]), translation, config["reference_endpoints"]
            )
            np.testing.assert_array_equal(reference["targets_tcp"], expected)
            np.testing.assert_array_equal(
                reference["opening_endpoints"], np.full(config["reference_endpoints"], initial["initial_opening"])
            )
        np.testing.assert_array_equal(references[0]["anchor_tcp"], initial["tcp_pose"])
        np.testing.assert_array_equal(references[1]["anchor_tcp"], raw[config["requery_step"]]["measured_tcp"])
        for row in raw:
            np.testing.assert_allclose(row["actual_dt"], 1 / config["control_fps"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(
                row["tcp_error"],
                pose_error(np.asarray(row["desired_tcp"]), np.asarray(row["after_tcp"])),
                atol=1e-12,
                rtol=0,
            )
            for source_key, tcp_key in (
                ("measured_source", "measured_tcp"),
                ("desired_source", "desired_tcp"),
                ("after_source", "after_tcp"),
            ):
                np.testing.assert_allclose(
                    np.asarray(row[source_key]) @ np.asarray(initial["tool_source_from_tcp"]),
                    row[tcp_key],
                    atol=1e-12,
                    rtol=0,
                )
            np.testing.assert_array_equal(row["gripper_current_action"], initial["held_gripper_motor_target"])
            np.testing.assert_allclose(row["gripper_control"], initial["expected_gripper_control"], atol=1e-12, rtol=0)
        recomputed = summarize_rows(raw, references, config)
        assert recomputed == result["metrics"]
        for path, digest in result["files_sha256"].items():
            assert sha256(path) == digest
        all_rows[(result["record"], result["method"])] = raw
        verification.append(
            dict(
                record=result["record"],
                method=result["method"],
                raw_steps=len(raw),
                metrics_recomputed_equal=True,
                geometry_and_command_checks=True,
            )
        )
    figures = root / "figures"
    figures.mkdir(exist_ok=False)
    for record in config["records"]:
        paired = [all_rows[(record, method)] for method in config["methods"]]
        for first, second in zip(paired[0][: config["requery_step"]], paired[1][: config["requery_step"]], strict=True):
            # v2 adds diagnostic keys even before requery, so compare all common
            # fields rather than requiring identical diagnostic dictionaries.
            assert {k: v for k, v in first.items() if k != "governor"} == {
                k: v for k, v in second.items() if k != "governor"
            }
            assert first["governor"] == {k: second["governor"][k] for k in first["governor"]}
        figure, axes = plt.subplots(7, 2, figsize=(15, 19), sharex=True, constrained_layout=True)
        for column, method in enumerate(config["methods"]):
            rows = all_rows[(record, method)]
            initial = json.loads((root / record / method / "initial.json").read_text())
            times = (np.arange(len(rows)) + 1) / config["control_fps"]
            actual = np.asarray([r["after_tcp"] for r in rows])[:, :3, 3]
            desired = np.asarray([r["desired_tcp"] for r in rows])[:, :3, 3]
            anchor = np.asarray(initial["tcp_pose"])[:3, 3]
            for axis, coordinate in enumerate("xyz"):
                ax = axes[axis, column]
                ax.plot(times, 1000 * (actual[:, axis] - anchor[axis]), label="actual TCP")
                ax.plot(times, 1000 * (desired[:, axis] - anchor[axis]), "--", label="admitted desired TCP")
                ax.set_ylabel(f"world {coordinate} from initial (mm)")
                ax.legend(loc="best", fontsize=8)
            axes[0, column].set_title(f"{record} / {method}: fixed down0.35m; one up0.03m query at5s")
            axes[3, column].plot(times, [r["governor"]["model_time_s"] for r in rows], label="model phase")
            axes[3, column].plot(times, [r["governor"]["retimed_clock_s"] for r in rows], "--", label="retimed clock")
            axes[3, column].set_ylabel("reference time (s)")
            axes[3, column].legend(fontsize=8)
            axes[4, column].plot(times, [r["governor"]["tracking_scale"] for r in rows], label="effective scale")
            axes[4, column].plot(
                times,
                [r["governor"].get("nominal_tracking_scale", r["governor"]["tracking_scale"]) for r in rows],
                "--",
                label="nominal tracking scale",
            )
            axes[4, column].fill_between(
                times, 0, 1, where=[r["governor"]["bridging"] for r in rows], color="orange", alpha=0.2, label="bridge"
            )
            axes[4, column].set_ylabel("scale / bridge")
            axes[4, column].legend(fontsize=8)
            error = np.asarray([r["tcp_error"] for r in rows])
            axes[5, column].plot(times, 1000 * np.linalg.norm(error[:, :3], axis=1), label="post-step TCP error")
            axes[5, column].plot(
                times,
                [1000 * r["governor"]["preceding_position_error_m"] for r in rows],
                "--",
                label="preceding admitted error",
            )
            axes[5, column].set_ylabel("position error (mm)")
            axes[5, column].legend(fontsize=8)
            axes[6, column].plot(times, np.asarray([r["command"][:6] for r in rows]))
            axes[6, column].set_ylabel("6D normalized arm command")
            axes[6, column].set_xlabel("physical elapsed time (s)")
            for ax in axes[:, column]:
                ax.axvline(config["requery_step"] / config["control_fps"], color="black", linestyle=":")
                ax.grid(alpha=0.25)
        for suffix in ("png", "pdf"):
            figure.savefig(figures / f"{record}_reference_recovery.{suffix}", dpi=160)
        plt.close(figure)
    write_new(
        root / "raw_verification.json",
        dict(
            cases=verification,
            paired_first100_identical=True,
            all_checks_passed=True,
            verified_steps=sum(x["raw_steps"] for x in verification),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reference_recovery_qualification_20261021.json")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--record")
    parser.add_argument("--method", choices=("governor_v1", "governor_v2"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("requires explicit CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=''")
    config, root = load_config(args.config)
    if args.seed != config["seed"] or sha256(config["fixture"]) != config["fixture_sha256"]:
        raise ValueError("seed or frozen fixture SHA differs")
    if config["protocol_version"] != 1 or config["steps"] != 200 or config["requery_step"] != 100:
        raise ValueError("requires the pre-registered200-step / one-step100-query protocol")
    if args.worker:
        return run_worker(args, config, root)
    # load_config creates this directory; empty is the only admissible prior state.
    if any(root.iterdir()):
        raise FileExistsError(f"qualification directory is not empty: {root}")
    inputs = fingerprints(args.config, config)
    write_new(root / "frozen_config.json", config)
    write_new(root / "input_manifest.json", dict(files_sha256=inputs, seed=args.seed, created_unix=time.time()))
    (root / "logs").mkdir(exist_ok=False)
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

    def execute(record, method):
        # Keep the venv symlink path; resolve() would select system Python.
        command = [
            str(Path(config["simulation_python"]).absolute()),
            str(Path(__file__).resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--seed",
            str(args.seed),
            "--worker",
            "--record",
            record,
            "--method",
            method,
        ]
        with (root / "logs" / f"{record}_{method}.log").open("x") as stream:
            process = subprocess.Popen(command, env=worker_env, stdout=stream, stderr=subprocess.STDOUT)
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"worker failed: {record}/{method}, exit{returncode}; failed artifacts retained")
        result = json.loads((root / record / method / "result.json").read_text())
        if result["error"] or result["steps"] != config["steps"]:
            raise ValueError(f"incomplete case: {record}/{method}")
        return result

    for record in config["records"]:
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            pending = [pool.submit(execute, record, method) for method in config["methods"]]
            for future in as_completed(pending):
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        dict(
                            completed=len(results),
                            record=result["record"],
                            method=result["method"],
                            metrics=result["metrics"],
                        )
                    ),
                    flush=True,
                )
    check_inputs(inputs)
    verify_and_plot(config, root, results)
    results.sort(key=lambda r: (r["record"], r["method"]))
    write_new(
        root / "qualification_summary.json",
        dict(
            seed=args.seed,
            expected=4,
            completed=len(results),
            complete=len(results) == 4,
            physical_steps=sum(r["steps"] for r in results),
            results=results,
            limitations="Synthetic reference recovery in independent MuJoCo; no WAM task-success or real-robot claim.",
        ),
    )
    check_inputs(inputs)
    output_files = {
        str(p.resolve()): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not any(part in ("numba_cache", "matplotlib") for part in p.parts)
    }
    write_new(
        root / "output_manifest.json",
        dict(files_sha256=output_files, inputs_sha256=inputs, output_manifest_self_excluded=True),
    )
    print(json.dumps(dict(complete=True, summary=str(root / "qualification_summary.json"))), flush=True)


if __name__ == "__main__":
    main()
