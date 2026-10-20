"""Freeze and execute the twelve CPU-only oracle cases registered by007."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from s2a.artifacts import REPO, sha256
from s2a.stability_position_simulation import execution_inputs, position_variant
from s2a.task_acceptance import task_outcome

ROOT = REPO / "outputs/wam_stability_20261021/short_replan_qualification"
CONFIGS = tuple(REPO / f"configs/short_replan_qualification_{n}_20261021.json" for n in (8, 16))
PRIOR_CONFIG = REPO / "configs/gripper_position_hold_qualification_20261021.json"
RECORDS = ("task0_init0", "task1_init0", "task2_init0")
VARIANTS = ("position_hold", "recovering_position_hold")
OWN_FILES = (
    "scripts/qualify_short_replan.py",
    "scripts/verify_short_replan.py",
    "scripts/audit_wam_position.py",
    "scripts/audit_wam_recovery.py",
    "scripts/verify_gripper_position_hold.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def unchanged(signatures):
    for path, expected in signatures.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen qualification input changed: {path}")


def require_cpu():
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("run from S2A with empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")


def protocol(path):
    """Permit only the registered interval/output/budget changes from004."""
    path = Path(path).resolve()
    if path not in CONFIGS:
        raise ValueError("configuration is not one of the two registered007 inputs")
    c, previous = read(path), read(PRIOR_CONFIG)
    interval = (8, 16)[CONFIGS.index(path)]
    expected_root = ROOT / f"interval{interval}"
    if (REPO / c["output_dir"]).resolve() != expected_root or c["max_observation_interval_steps"] != interval:
        raise ValueError("registered output or interval differs")
    varying = {"output_dir", "notes", "variants", "qualification", "max_observation_interval_steps"}
    if {k: v for k, v in c.items() if k not in varying} != {k: v for k, v in previous.items() if k not in varying}:
        raise ValueError("007 changes an execution parameter other than its requery interval")
    if c["variants"] != previous["variants"][1:] or [v["name"] for v in c["variants"]] != list(VARIANTS):
        raise ValueError("requires the qualified two raw-alpha1 v3 variants")
    expected = dict(
        workers=2,
        expected_cases=6,
        expected_physics_steps=3600,
        position_rmse_max_m=0.01,
        rotation_rmse_max_rad=0.05,
        timing_reference_interval_steps=80,
        preexisting_qualification=(
            "outputs/wam_stability_20261021/gripper_position_hold_qualification/qualification_independent_verification.json"
        ),
    )
    if c["qualification"] != expected or c["records"] != list(RECORDS) or c["sampling_seeds"] != [42]:
        raise ValueError("qualification budget/limits/records differ from007")
    if sha256(c["fixture"]) != c["fixture_sha256"]:
        raise ValueError("fixture SHA differs")
    for variant in VARIANTS:
        position_variant(c, variant, source="oracle")
    return c, expected_root


def freeze():
    signatures = {str(PRIOR_CONFIG): sha256(PRIOR_CONFIG)}
    for filename in OWN_FILES:
        signatures[str(REPO / filename)] = sha256(REPO / filename)
    old = read(PRIOR_CONFIG)
    old_root = REPO / old["output_dir"]
    for name in (
        "qualification_launch_manifest.json",
        "qualification_independent_verification.json",
        "qualification_qpos_opening_supplement.json",
    ):
        signatures[str(old_root / name)] = sha256(old_root / name)
    prior = read(old_root / "qualification_independent_verification.json")
    supplement = read(old_root / "qualification_qpos_opening_supplement.json")
    if not prior["qualified"] or not supplement["qualified"] or prior["verified_cases"] != 9:
        raise ValueError("prior004 qualification is not complete and qualified")
    for path in CONFIGS:
        c, _ = protocol(path)
        signatures.update(execution_inputs(c, path))
    for record in RECORDS:
        native = REPO / old["oracle_reference_root"] / f"{record}_s42"
        for name in ("reference.npz", "steps.jsonl", "result.json", "manifest.json"):
            signatures[str(native / name)] = sha256(native / name)
        rows = [json.loads(line) for line in (native / "steps.jsonl").read_text().splitlines()]
        if not task_outcome([r["success"] for r in rows], expected_steps=600, stable_steps=40)["stable"]:
            raise ValueError("native oracle is not a complete stable success")
        for variant in VARIANTS:
            case = old_root / "simulation" / f"oracle_{variant}" / f"{record}_s42"
            for name in ("manifest.json", "controller.json", "result.json", "steps.jsonl", "chunks.json", "model.xml"):
                signatures[str(case / name)] = sha256(case / name)
            signatures.update(read(case / "manifest.json")["simulator_source_sha256"])
    unchanged(signatures)
    return signatures


def command(config, variant, *, dry_run=False):
    value = [
        str(REPO / ".venv/bin/python"),
        str(REPO / "scripts/run_wam_position_matrix.py"),
        "--config",
        str(config),
        "--seed",
        "42",
        "--source",
        "oracle",
        "--variant",
        variant,
        "--workers",
        "2",
    ]
    return value + (["--dry-run"] if dry_run else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require_cpu()
    manifest_path = ROOT / "qualification_launch_manifest.json"
    if args.prepare_only:
        if ROOT.exists():
            raise FileExistsError(ROOT)
        signatures = freeze()
        checks = []
        for config in CONFIGS:
            for variant in VARIANTS:
                cmd = command(config, variant, dry_run=True)
                result = subprocess.run(
                    cmd, cwd=REPO, env=os.environ.copy(), check=True, capture_output=True, text=True
                )
                checks.append(dict(command=cmd, result=json.loads(result.stdout)))
        if ROOT.exists():
            raise RuntimeError("validation-only entry created output")
        ROOT.mkdir(parents=True, exist_ok=False)
        copies = ROOT / "review_sources"
        copies.mkdir()
        copied = {}
        for name, digest in signatures.items():
            source = Path(name)
            if source.is_relative_to(REPO) and source.relative_to(REPO).parts[0] in ("src", "scripts", "configs"):
                target = copies / str(source.relative_to(REPO)).replace("/", "__")
                shutil.copyfile(source, target)
                assert sha256(target) == digest
                copied[str(target)] = digest
        unchanged(signatures)
        write_new(
            manifest_path,
            dict(
                seed=42,
                configs=list(map(str, CONFIGS)),
                registered_cases=12,
                registered_physics_steps=7200,
                max_cpu_workers=2,
                model_started=False,
                physics_started=False,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                input_sha256=signatures,
                review_source_sha256=copied,
                dry_runs=checks,
            ),
        )
        print(json.dumps(dict(prepared=True, manifest_sha256=sha256(manifest_path))), flush=True)
        return
    manifest = read(manifest_path)
    unchanged(manifest["input_sha256"])
    unchanged(manifest["review_source_sha256"])
    prior_path = ROOT / "prior80_independent_verification.json"
    prior = read(prior_path)
    if not prior["qualified"] or prior["verified_cases"] != 6:
        raise ValueError("independent read-only replay of004 reference is required before physics")
    unchanged(prior["input_sha256"])
    write_new(
        ROOT / "qualification_execution_launch.json",
        dict(
            coordinator_pid=os.getpid(),
            launch_manifest_sha256=sha256(manifest_path),
            prior80_verification_sha256=sha256(prior_path),
            registered_cases=12,
            registered_physics_steps=7200,
            model_started=False,
            cuda_visible_devices="",
        ),
    )
    logs = ROOT / "coordinator_logs"
    logs.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    completed = []
    for config in CONFIGS:
        c, _ = protocol(config)
        for variant in VARIANTS:
            unchanged(manifest["input_sha256"])
            cmd = command(config, variant)
            label = f"interval{c['max_observation_interval_steps']}_{variant}"
            with (logs / f"{label}.log").open("x") as stream:
                process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
                write_new(logs / f"{label}.process.json", dict(pid=process.pid, command=cmd, cuda="", workers=2))
                code = process.wait()
            unchanged(manifest["input_sha256"])
            completed.append(dict(group=label, returncode=code))
            print(json.dumps(completed[-1]), flush=True)
            if code:
                write_new(ROOT / "qualification_execution_error.json", dict(completed=completed, automatic_retry=False))
                raise SystemExit(code)
    subprocess.run(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/verify_short_replan.py"), "--seed", "42"],
        cwd=REPO,
        env=env,
        check=True,
    )
    print(json.dumps(dict(executed_cases=12, registered_physics_steps=7200, model_started=False)), flush=True)


if __name__ == "__main__":
    main()
