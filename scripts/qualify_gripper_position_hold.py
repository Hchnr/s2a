"""Freeze and run the nine004 oracle cases through the shared position executor.

This coordinator starts at most two CPU/OSMesa workers and never a model. It
first proves exact physical baseline repetition before admitting new cases.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from s2a.artifacts import REPO, sha256
from s2a.stability_position_simulation import execution_inputs, position_variant
from s2a.task_acceptance import task_outcome

VARIANTS = ("baseline_v2", "position_hold", "recovering_position_hold")
RECORDS = ("task0_init0", "task1_init0", "task2_init0")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def protocol(path):
    c = read(path)
    fixed = dict(
        seed=42,
        control_fps=20,
        online_max_steps=600,
        online_chunk_size=40,
        online_execution_horizon=40,
        max_observation_interval_steps=80,
        stable_success_last_steps=40,
        inverse_weight=0.5,
        max_command_rate=5.0,
    )
    if any(c[k] != v for k, v in fixed.items()) or c["records"] != list(RECORDS) or c["sampling_seeds"] != [42]:
        raise ValueError("qualification does not match frozen004 protocol")
    if [v["name"] for v in c["variants"]] != list(VARIANTS):
        raise ValueError("requires exactly the three fixed004 variants")
    for v in c["variants"]:
        position_variant(c, v["name"], source="oracle")
    if c["qualification"] != {
        "workers": 2,
        "expected_cases": 9,
        "expected_physics_steps": 5400,
        "position_rmse_max_m": 0.01,
        "rotation_rmse_max_rad": 0.05,
        "historical_baseline_root": "outputs/wam_stability_20261021/recovery_executor_qualification/simulation/oracle_intent_hold",
        "baseline_exact_before_candidate": True,
    }:
        raise ValueError("qualification budget or limits differ")
    if sha256(c["fixture"]) != c["fixture_sha256"]:
        raise ValueError("fixture differs")
    root = (REPO / c["output_dir"]).resolve()
    if not root.is_relative_to(REPO / "outputs"):
        raise ValueError("output must be below repository outputs")
    return c, root


def freeze(c, path):
    values = execution_inputs(c, path)
    for name in (
        "scripts/qualify_gripper_position_hold.py",
        "scripts/verify_gripper_position_hold.py",
        "scripts/audit_wam_recovery.py",
        "tests/test_gripper_position_hold.py",
    ):
        values[str(REPO / name)] = sha256(REPO / name)
    for record in RECORDS:
        native = REPO / c["oracle_reference_root"] / f"{record}_s42"
        for name in ("reference.npz", "steps.jsonl", "result.json", "manifest.json"):
            values[str(native / name)] = sha256(native / name)
        rows = [json.loads(line) for line in (native / "steps.jsonl").read_text().splitlines()]
        if not task_outcome([r["success"] for r in rows], expected_steps=600, stable_steps=40)["stable"]:
            raise ValueError("oracle source is not a complete stable success")
        historical = REPO / c["qualification"]["historical_baseline_root"] / f"{record}_s42"
        for name in ("manifest.json", "controller.json", "result.json", "steps.jsonl", "chunks.json", "model.xml"):
            values[str(historical / name)] = sha256(historical / name)
        previous = read(historical / "manifest.json")
        for filename, digest in previous["simulator_source_sha256"].items():
            if sha256(filename) != digest:
                raise ValueError(f"external simulator changed: {filename}")
            values[filename] = digest
    return values


def unchanged(values):
    for path, digest in values.items():
        if sha256(path) != digest:
            raise ValueError(f"qualification input changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("run CPU qualification from repository with empty CUDA/LD paths")
    c, root = protocol(args.config)
    manifest_path = root / "qualification_launch_manifest.json"
    if args.prepare_only:
        if root.exists():
            raise FileExistsError(root)
        signatures = freeze(c, args.config)
        # Validation-only entry checks must not create an experiment directory.
        commands = []
        for variant in VARIANTS:
            command = [
                str(REPO / ".venv/bin/python"),
                str(REPO / "scripts/run_wam_position_matrix.py"),
                "--config",
                str(Path(args.config).absolute()),
                "--seed",
                "42",
                "--source",
                "oracle",
                "--variant",
                variant,
                "--workers",
                "2",
                "--dry-run",
            ]
            result = subprocess.run(
                command, check=True, cwd=REPO, env=os.environ.copy(), capture_output=True, text=True
            )
            commands.append(dict(command=command, result=json.loads(result.stdout)))
        if root.exists():
            raise RuntimeError("dry-run created output")
        root.mkdir(parents=True, exist_ok=False)
        copies = root / "review_sources"
        copies.mkdir()
        copied = {}
        for filename, digest in signatures.items():
            source = Path(filename)
            if source.is_relative_to(REPO) and source.parts[len(REPO.parts)] in ("src", "scripts", "configs", "tests"):
                destination = copies / str(source.relative_to(REPO)).replace("/", "__")
                shutil.copyfile(source, destination)
                assert sha256(destination) == digest
                copied[str(destination)] = digest
        unchanged(signatures)
        write_new(
            manifest_path,
            dict(
                seed=42,
                config=str(Path(args.config).resolve()),
                input_sha256=signatures,
                review_source_sha256=copied,
                dry_runs=commands,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                model_started=False,
                physics_started=False,
                registered_cases=9,
                registered_physics_steps=5400,
                max_cpu_workers=2,
            ),
        )
        print(json.dumps(dict(prepared=True, manifest_sha256=sha256(manifest_path))))
        return
    manifest = read(manifest_path)
    unchanged(manifest["input_sha256"])
    logs = root / "qualification_coordinator_logs"
    logs.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    completed = []
    for variant in VARIANTS:
        unchanged(manifest["input_sha256"])
        command = [
            str(REPO / ".venv/bin/python"),
            str(REPO / "scripts/run_wam_position_matrix.py"),
            "--config",
            str(Path(args.config).absolute()),
            "--seed",
            "42",
            "--source",
            "oracle",
            "--variant",
            variant,
            "--workers",
            "2",
        ]
        with (logs / f"{variant}.log").open("x") as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            write_new(logs / f"{variant}.process.json", dict(pid=process.pid, command=command, cuda="", workers=2))
            code = process.wait()
        unchanged(manifest["input_sha256"])
        completed.append(dict(variant=variant, returncode=code))
        print(json.dumps(completed[-1]), flush=True)
        if code:
            write_new(
                root / "qualification_execution_error.json",
                dict(completed=completed, reason="matrix error; no automatic restart or threshold change"),
            )
            raise SystemExit(code)
        if variant == "baseline_v2":
            from verify_gripper_position_hold import verify_baseline

            check = verify_baseline(c, root)
            write_new(root / "baseline_exact_repetition.json", check)
            if not check["all_physical_fields_exact"]:
                raise RuntimeError("baseline mismatch; candidates were not started")
    subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            str(REPO / "scripts/verify_gripper_position_hold.py"),
            "--config",
            str(Path(args.config).absolute()),
            "--seed",
            "42",
        ],
        check=True,
        cwd=REPO,
        env=env,
    )
    print(json.dumps(dict(executed_cases=9, registered_physics_steps=5400, model_started=False)), flush=True)


if __name__ == "__main__":
    main()
