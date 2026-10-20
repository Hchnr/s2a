"""CPU-only paired oracle qualification for the new causal gripper adapter."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from s2a.action_reference import decode_action_targets, encode_oracle_targets
from s2a.artifacts import sha256, write_json
from s2a.checkpoint_simulation import object_progress
from s2a.geometry import inverse, pose_error
from s2a.gripper import OpeningIntent, opening_position_target
from s2a.gripper_tracking import (
    GripperTrackingAdapter,
    GripperTrackingConfig,
    panda_opening_from_motor_target,
)
from s2a.libero_simulation import (
    aperture,
    controller_contract,
    current_observation,
    make_environment,
    measured_twist,
    sha256_bytes,
    source_pose,
)
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.task_acceptance import task_outcome

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def protocol(config_path):
    c = read(config_path)
    if (c["seed"], c["steps"], c["control_fps"], c["generation_horizon"], c["execution_horizon"]) != (
        42,
        600,
        20,
        40,
        8,
    ):
        raise ValueError("qualification protocol differs")
    if c["records"] != ["task0_init0", "task1_init0", "task2_init0"] or c["modes"] != ["intent", "tracking"]:
        raise ValueError("qualification matrix differs")
    if asdict(GripperTrackingConfig()) != c["tracking"]:
        raise ValueError("qualification requires frozen adapter defaults")
    fixtures = read(c["fixture_path"])["records"]
    records = {r["name"]: r for r in fixtures if r["name"] in c["records"]}
    if set(records) != set(c["records"]):
        raise ValueError("missing fixed initial states")
    return c, REPO / c["output_dir"], records


def frozen_files(c, config_path, records):
    files = [Path(config_path), Path(c["fixture_path"]), Path(c["policy_template"])]
    template = read(c["policy_template"])
    files += [Path(template["tool_contract"]), Path(template["libero_config"])]
    files += [
        REPO / "src/s2a" / f"{name}.py"
        for name in (
            "gripper_tracking",
            "gripper",
            "osc",
            "geometry",
            "libero_simulation",
            "checkpoint_simulation",
            "action_reference",
            "task_acceptance",
            "artifacts",
        )
    ]
    files += [REPO / "scripts/qualify_gripper_tracking.py", REPO / "tests/test_gripper_tracking.py"]
    for record in records:
        source = Path(c["native_reference_root"]) / f"{record}_s42"
        files += [source / name for name in ("reference.npz", "result.json", "steps.jsonl", "manifest.json")]
        result = read(source / "result.json")
        steps = [json.loads(line) for line in (source / "steps.jsonl").read_text().splitlines()]
        if (
            result["error"]
            or result["steps"] != 600
            or not task_outcome([s["success"] for s in steps], expected_steps=600, stable_steps=40)["stable"]
        ):
            raise ValueError("qualification reference is not a complete stable native success")
    suite = REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite"
    files += [
        suite / p
        for p in (
            "models/grippers/panda_gripper.py",
            "models/assets/grippers/panda_gripper.xml",
            "robots/manipulator.py",
            "robots/single_arm.py",
            "environments/base.py",
        )
    ]
    return {str(p.resolve()): sha256(p) for p in files}


def unchanged(manifest):
    for path, signature in manifest["inputs_sha256"].items():
        if sha256(path) != signature:
            raise ValueError(f"qualification input changed: {path}")


def worker(c, root, record, mode):
    dest = root / "physics" / record["name"] / mode
    dest.mkdir(parents=True, exist_ok=False)
    simulation = dict(
        read(c["policy_template"]),
        seed=42,
        task_id=record["task_id"],
        task_text=record["task"],
        camera_observations=False,
        record_video=False,
        return_video=False,
        refresh_controller_after_restore=True,
    )
    reference_path = Path(c["native_reference_root"]) / f"{record['name']}_s42/reference.npz"
    with np.load(reference_path) as data:
        references, reference_openings = data["poses"], data["openings"]
    if len(references) != 600 or reference_openings.shape != (600,):
        raise ValueError("native successful reference budget differs")
    h = np.asarray(read(simulation["tool_contract"])["source_from_tcp"])
    write_json(
        dest / "manifest.json",
        dict(
            config=c,
            simulation=simulation,
            record=record,
            mode=mode,
            reference_sha256=sha256(reference_path),
            tool_sha256=sha256(simulation["tool_contract"]),
            environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONPATH")},
            adapter_contact_input=False,
            model_query=False,
            source="oracle_qualification",
        ),
    )
    env, _ = make_environment(simulation, dest, record, 0)
    rows, chunks = [], []
    error = None
    try:
        obs = current_observation(env)
        _, contract = controller_contract(env)
        write_json(dest / "controller.json", contract)
        if contract["gripper_model"] != "PandaGripper" or not np.isclose(env.env.control_timestep, 0.05):
            raise ValueError("wrong native gripper or control period")
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            0.05,
            compensate=True,
            inverse_weight=c["inverse_weight"],
            max_command_rate=c["max_command_rate"],
        )
        initial_opening = aperture(obs)
        gripper = env.robots[0].gripper
        gripper.current_action = opening_position_target(initial_opening)
        initial = np.asarray(env.sim.get_state().flatten()).copy()
        write_json(
            dest / "initial.json",
            dict(
                physics_sha256=sha256_bytes(initial.tobytes()),
                pose=source_pose(obs),
                qpos=obs["robot0_gripper_qpos"],
                opening=initial_opening,
                motor_target=gripper.current_action.copy(),
                motor_target_origin="Aligned from initial observed opening, same as old oracle run_case; not the historical expert native motor target.",
            ),
        )
        (dest / "model.xml").write_text(env.sim.model.get_xml())
        adapter = GripperTrackingAdapter(initial_opening, config=GripperTrackingConfig(**c["tracking"]))
        intent = OpeningIntent(initial_opening, band=c["opening_hysteresis"])
        previous, past_dt = 0.0, 0.05
        with (dest / "steps.jsonl").open("x") as stream:
            for step in range(600):
                measured = source_pose(obs)
                if step % c["execution_horizon"] == 0:
                    before = np.asarray(env.sim.get_state().flatten()).copy()
                    indices = np.minimum(step + np.arange(c["generation_horizon"]), 599)
                    anchor = measured @ h
                    actions = encode_oracle_targets(anchor, references[indices] @ h, reference_openings[indices])
                    targets, openings = decode_action_targets(anchor, actions, horizon=c["execution_horizon"])
                    if not np.array_equal(before, env.sim.get_state().flatten()):
                        raise ValueError("physics changed during oracle reference generation")
                    chunks.append(
                        dict(
                            step=step,
                            anchor_tcp=anchor,
                            actions=actions,
                            targets_tcp=targets,
                            physics_unchanged=True,
                            physics_sha256=sha256_bytes(before.tobytes()),
                        )
                    )
                index = step % c["execution_horizon"]
                desired = targets[index] @ inverse(h)
                opening = float(openings[index])
                opening_before = aperture(obs)
                motor_before = gripper.current_action.copy()
                motor_opening = panda_opening_from_motor_target(motor_before)
                diagnostics = None
                if mode == "tracking":
                    diagnostics = adapter.update(
                        opening, opening_before, motor_opening=motor_opening, previous_command=previous, dt=past_dt
                    )
                    grip_command = diagnostics.command
                else:
                    grip_command = intent.update(opening)
                converted = converter.convert(measured, desired, measured_twist(env))
                command = np.r_[converted["command"], grip_command]
                qpos_before = np.asarray(obs["robot0_gripper_qpos"]).copy()
                time_before = float(env.sim.data.time)
                env.step(command)
                obs = current_observation(env)
                after = source_pose(obs)
                past_dt = float(env.sim.data.time) - time_before
                if not np.isclose(past_dt, 0.05, atol=1e-10, rtol=0):
                    raise ValueError("physical period differs")
                # Privileged object/contact fields are evaluated only after the
                # command and never supplied to either gripper adapter.
                progress = object_progress(env)
                row = dict(
                    step=step,
                    command_start_s=step * 0.05,
                    endpoint_s=(step + 1) * 0.05,
                    measured_source=measured,
                    desired_source=desired,
                    after_source=after,
                    opening_target=opening,
                    opening_before=opening_before,
                    opening_after=aperture(obs),
                    qpos_before=qpos_before,
                    qpos_after=obs["robot0_gripper_qpos"],
                    motor_target_before=motor_before,
                    motor_target_after=gripper.current_action.copy(),
                    motor_opening_before=motor_opening,
                    motor_opening_after=panda_opening_from_motor_target(gripper.current_action),
                    actual_actuator_control=env.sim.data.ctrl[env.robots[0]._ref_joint_gripper_actuator_indexes].copy(),
                    previous_actual_gripper_command=previous,
                    command=command,
                    adapter=asdict(diagnostics) if diagnostics else None,
                    actual_dt=past_dt,
                    tracking_tcp_error=pose_error(desired @ h, after @ h),
                    **progress,
                )
                rows.append(row)
                stream.write(json.dumps(row, default=lambda v: v.tolist(), allow_nan=False) + "\n")
                stream.flush()
                previous = grip_command
                if step % 100 == 99:
                    write_json(dest / "progress.json", dict(steps=step + 1, mode=mode, record=record["name"]))
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    write_json(dest / "chunks.json", chunks)
    result = dict(mode=mode, record=record["name"], steps=len(rows), error=error)
    if len(rows) == 600:
        result["outcome"] = task_outcome([r["success"] for r in rows], expected_steps=600, stable_steps=40)
        errors = np.asarray([r["tracking_tcp_error"] for r in rows])
        commands = [r["command"][-1] for r in rows]
        nonzero = [x for x in commands if x]
        result.update(
            position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            opening_mae=float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in rows])),
            opening_peak_error=float(max(abs(r["opening_after"] - r["opening_target"]) for r in rows)),
            motor_compression_gap_mean=float(
                np.mean([max(0, r["opening_before"] - r["motor_opening_before"]) for r in rows])
            ),
            command_counts={str(v): commands.count(v) for v in (-1.0, 0.0, 1.0)},
            nonzero_direction_reversals=sum(a != b for a, b in zip(nonzero, nonzero[1:])),
            object_contact_steps={
                name: sum(r["objects"][name]["grasp_contact"] for r in rows) for name in rows[0]["objects"]
            },
        )
        for direction, sign in (("opening", -1), ("closing", 1), ("hold", 0)):
            selected = [r for r in rows if r["command"][-1] == sign]
            result[f"{direction}_opening_mae"] = (
                float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in selected])) if selected else None
            )
        contact_rows = [r for r in rows if any(o["grasp_contact"] for o in r["objects"].values())]
        result["any_object_native_contact_endpoints"] = len(contact_rows)
        result["contact_opening_mae"] = (
            float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in contact_rows]))
            if contact_rows
            else None
        )
        result["contact_motor_compression_gap_mean"] = (
            float(np.mean([r["opening_after"] - r["motor_opening_after"] for r in contact_rows]))
            if contact_rows
            else None
        )
        motor_motion = [abs(r["motor_opening_after"] - r["motor_opening_before"]) for r in rows if r["command"][-1]]
        result["motor_aperture_change_nonzero_command_max"] = float(max(motor_motion, default=0))
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result


def report(c, root):
    rows, files = [], {}
    for record in c["records"]:
        initials = []
        for mode in c["modes"]:
            dest = root / "physics" / record / mode
            r = read(dest / "result.json")
            steps = [json.loads(s) for s in (dest / "steps.jsonl").read_text().splitlines()]
            if r["error"] or r["steps"] != 600 or [s["step"] for s in steps] != list(range(600)):
                raise ValueError("qualification is incomplete")
            if r["outcome"] != task_outcome([s["success"] for s in steps], expected_steps=600, stable_steps=40):
                raise ValueError("task summary differs from endpoints")
            initials.append(read(dest / "initial.json"))
            rows.append(r)
            for p in dest.iterdir():
                if p.is_file():
                    files[str(p)] = sha256(p)
        if initials[0] != initials[1]:
            raise ValueError("paired initial physics/measurement/motor target differs")
    gate = all(
        r["outcome"]["stable"]
        and r["position_rmse_m"] <= c["acceptance"]["position_rmse_max_m"]
        and r["rotation_rmse_rad"] <= c["acceptance"]["rotation_rmse_max_rad"]
        for r in rows
    )
    value = dict(
        config=c,
        cases=rows,
        qualification_passed=gate,
        steps=3600,
        files_sha256=files,
        evidence="CPU MuJoCo oracle executor qualification; not WAM closed-loop success",
        limitations=[
            "Three old scenes/one seed; no unseen/general task reliability claim.",
            "Aperture/command and stall proxies do not certify grasp force or stable holding.",
            "Defaults frozen; failed qualification is reported without fitting to WAM task labels.",
        ],
    )
    if (root / "summary.json").exists():
        raise FileExistsError(root / "summary.json")
    write_json(root / "summary.json", value)
    print(json.dumps(dict(qualification_passed=gate, cases=rows), ensure_ascii=False), flush=True)


def main():
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("run from S2A with CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=''")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/gripper_tracking_qualification_20261021.json")
    p.add_argument("--seed", type=int, required=True, choices=(42,))
    p.add_argument("--stage", choices=("prepare", "run", "worker", "report"), required=True)
    p.add_argument("--record")
    p.add_argument("--mode", choices=("intent", "tracking"))
    args = p.parse_args()
    c, root, records = protocol(args.config)
    if args.stage == "prepare":
        inputs = frozen_files(c, args.config, records)
        root.mkdir(parents=True, exist_ok=False)
        write_json(
            root / "manifest.json",
            dict(
                config=c,
                inputs_sha256=inputs,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                scope="Six oracle-only qualification trials; no WAM query or adapter contact input.",
            ),
        )
        return
    manifest = read(root / "manifest.json")
    if manifest["config"] != c:
        raise ValueError("frozen qualification changed")
    unchanged(manifest)
    if args.stage == "worker":
        worker(c, root, records[args.record], args.mode)
    elif args.stage == "report":
        report(c, root)
    else:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            LD_LIBRARY_PATH="",
            MUJOCO_GL="osmesa",
            PYOPENGL_PLATFORM="osmesa",
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="2",
            LP_NUM_THREADS="2",
            PYTHONPATH=str(REPO / "src"),
        )

        def execute(job):
            record, mode = job
            log = root / f"{record}_{mode}.log"
            command = [
                str((REPO / c["simulation_python"]).absolute()),
                str(Path(__file__).resolve()),
                "--config",
                args.config,
                "--seed",
                "42",
                "--stage",
                "worker",
                "--record",
                record,
                "--mode",
                mode,
            ]
            with log.open("x") as stream:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                write_json(
                    root / f"{record}_{mode}_process.json",
                    dict(pid=process.pid, command=command, CUDA_VISIBLE_DEVICES=""),
                )
                code = process.wait()
            if code:
                raise RuntimeError(f"{record}/{mode} worker exited {code}; retained evidence, no automatic retry")

        jobs = [(r, m) for r in c["records"] for m in c["modes"]]
        with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
            list(pool.map(execute, jobs))
        unchanged(manifest)
        report(c, root)
        write_json(
            root / "complete.json",
            dict(complete=True, qualification_passed=read(root / "summary.json")["qualification_passed"], steps=3600),
        )


if __name__ == "__main__":
    main()
