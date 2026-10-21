"""Wait on the frozen 20261020 study, then run CPU-only verification and reporting.

--check-only is read-only and returns status=pending until every gate is ready.
A durable run owns a new outputs/wam_independent_20261020/finalizer directory.
Existing stage outputs or a previous finalizer require manual review; no resume,
replacement, model launch, physical retry or Git commit is performed.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRIMARY = REPO / "outputs/wam_independent_20261020"
DIAGNOSIS = REPO / "outputs/wam_horizon_diagnosis_20261020"
OUTPUT = PRIMARY / "finalizer"
FINGERPRINTS = PRIMARY / "checkpoint_fingerprints.json"
CONFIGS = (
    ("main", "configs/wam_independent_20261020.json", 144, 8, 4),
    ("execute16", "configs/wam_horizon_execute16_20261020.json", 36, 16, 2),
    ("execute40", "configs/wam_horizon_execute40_20261020.json", 36, 40, 2),
)
REPORT = REPO / "reports/wam_independent_20261020.md"
REPORT_MANIFEST = PRIMARY / "report_manifest.json"
POLL_SECONDS = 30


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")


def process(pid):
    """Inspect ownership only; this script never signals study processes."""
    base = Path("/proc") / str(pid)
    try:
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(ticks=fields[19], argv=(base / "cmdline").read_bytes().split(b"\0"))
    except FileNotFoundError:
        return None


def owned_process(pid, script, pending, errors):
    identity = process(pid)
    if identity and script.encode() not in identity["argv"]:
        errors.append(f"PID {pid} is active with a different identity; manual review required")
    elif identity:
        pending.append(f"{script}: PID {pid} is still running")
    return identity


def optional_json(path, pending):
    try:
        return read_json(path)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # Producers write JSON directly; a file may be observed mid-write.
        pending.append(f"JSON not yet readable: {path.relative_to(REPO)}")
        return None


def studies():
    result = []
    for name, filename, count, horizon, initial_count in CONFIGS:
        config = read_json(REPO / filename)
        root = (REPO / config["output_dir"]).resolve()
        expected_root = PRIMARY if name == "main" else DIAGNOSIS / name
        checkpoints = [x["name"] for x in config["checkpoints"]]
        expected_checkpoints = (
            ["warm6200_masked", "8400_masked", "8600_masked", "8800_masked"] if name == "main" else ["8800_masked"]
        )
        if (
            root != expected_root
            or config["seed"] != 42
            or config["sampling_seeds"] != [42, 43]
            or config["online_max_steps"] != 600
            or config["control_fps"] != 20
            or config["online_chunk_size"] != 40
            or config["online_execution_horizon"] != horizon
            or config["online_initial_states_per_task"] != initial_count
            or len(config["tasks"]) != 3
            or checkpoints != expected_checkpoints
        ):
            raise ValueError(f"frozen study protocol differs: {filename}")
        fixture_path = root / "simulation/fixtures.json"
        fixtures = read_json(fixture_path)
        records = fixtures["records"]
        grid = {(r["task_index"], r["initial_index"]) for r in records}
        if (
            fixtures["config"] != config
            or len(records) != 3 * initial_count
            or len({r["name"] for r in records}) != len(records)
            or grid != {(task, initial) for task in range(3) for initial in range(initial_count)}
        ):
            raise ValueError(f"frozen fixture matrix differs: {fixture_path}")
        expected = {
            root / "simulation" / label / f"{r['name']}_s{seed}" / "result.json": (label, r["name"], seed)
            for label in ("native", "oracle", *checkpoints)
            for r in records
            for seed in config["sampling_seeds"]
        }
        if len(expected) != count:
            raise ValueError(f"frozen case count differs: {filename}")
        result.append(dict(name=name, filename=filename, config=config, root=root, expected=expected, records=records))
    return result


def stage_outputs(all_studies):
    return [p for study in all_studies for p in (study["root"] / "summary.json", study["root"] / "integrity.json")] + [
        REPORT,
        REPORT_MANIFEST,
    ]


def launch_jobs(path, script):
    jobs = read_json(path)
    by_partition = {job["partition"]: job for job in jobs}
    if set(by_partition) != set(range(4)) or len(jobs) != 4:
        raise ValueError(f"current four-GPU launch identity differs: {path}")
    for partition, job in by_partition.items():
        command = job["command"]
        if (
            job["gpu"] != 4 + partition
            or script not in command
            or command[command.index("--partition") + 1] != str(partition)
            or command[command.index("--seed") + 1] != "42"
        ):
            raise ValueError(f"current launch command/GPU differs: {path}")
    return by_partition


def inspect_partition(study, partition, coordinator, pending, errors):
    root = study["root"] / "partitions" / str(partition)
    identity = (
        owned_process(coordinator["pid"], "scripts/run_wam_independent_partition.py", pending, errors)
        if coordinator
        else None
    )
    for marker in ("error.json", "cancelled.json", "canceled.json"):
        if (root / marker).exists():
            errors.append(f"current partition has {marker}: {root.relative_to(REPO)}; no model failure inferred")
    manifest = optional_json(root / "manifest.json", pending)
    complete = optional_json(root / "complete.json", pending)
    exit_record = optional_json(root / "service_exit.json", pending)
    if manifest is None or complete is None or exit_record is None:
        pending.append(f"{study['name']}/partition{partition}: waiting for manifest, complete and service_exit")
        if coordinator and identity is None:
            errors.append(f"{study['name']}/partition{partition}: coordinator ended without complete release evidence")
        return
    if coordinator is None:
        errors.append(f"completed partition has no registered coordinator: {root.relative_to(REPO)}")
    config = study["config"]
    replicas = 4 // len(config["checkpoints"])
    checkpoint = config["checkpoints"][partition // replicas]
    indices = list(range(partition % replicas, len(study["records"]) * 2, replicas))
    if (
        manifest["config"] != config
        or manifest["checkpoint"] != checkpoint
        or manifest["gpu"] != 4 + partition
        or manifest["port"] != 8904 + partition
        or manifest["job_indices"] != indices
        or complete.get("complete") is not True
        or complete.get("job_indices") != indices
    ):
        errors.append(f"partition protocol/completion identity differs: {root.relative_to(REPO)}")
    service = optional_json(root / "process.json", pending)
    replica = optional_json(root / "replica_check.json", pending)
    if service is None or replica is None:
        errors.append(f"partition service/reproduction identity missing: {root.relative_to(REPO)}")
        return
    if service["pid"] != exit_record["pid"] or (coordinator and service["coordinator_pid"] != coordinator["pid"]):
        errors.append(f"partition service PID identity differs: {root.relative_to(REPO)}")
    # The runner intentionally terminates its owned service after successful jobs.
    if exit_record["returncode"] not in (0, -signal.SIGTERM):
        errors.append(
            f"service failed exit {exit_record['returncode']}: {root.relative_to(REPO)}; manual review required"
        )
    if process(exit_record["pid"]):
        errors.append(f"recorded released service PID is active/reused: {exit_record['pid']}")
    if replica.get("identical") is not True or replica.get("max_abs") != 0:
        errors.append(f"partition did not reproduce the audited H40 request: {root.relative_to(REPO)}")


def snapshot(all_studies, cache):
    pending, errors, matrices = [], [], []
    existing = [str(path.relative_to(REPO)) for path in stage_outputs(all_studies) if path.exists()]
    if existing:
        errors.append(
            f"existing stage outputs require manual review; nothing will be skipped or overwritten: {existing}"
        )
    for study in all_studies:
        present = set((study["root"] / "simulation").glob("*/*/result.json"))
        extra = present - study["expected"].keys()
        if extra:
            errors.append(
                f"unexpected result paths in {study['name']}: {sorted(str(p.relative_to(REPO)) for p in extra)}"
            )
        valid = 0
        for path, (label, record, seed) in study["expected"].items():
            if any(path.with_name(marker).exists() for marker in ("error.json", "cancelled.json", "canceled.json")):
                errors.append(f"current trial has error/cancellation evidence: {path.parent.relative_to(REPO)}")
                continue
            result = optional_json(path, pending)
            if result is None:
                continue
            source = label if label in ("native", "oracle") else "wam"
            if (
                result["record"] != record
                or result["sampling_seed"] != seed
                or result["source"] != source
                or result["checkpoint"] != (label if source == "wam" else None)
                or result["error"] is not None
                or result["steps"] != 600
                or result["expected_steps"] != 600
            ):
                errors.append(
                    f"errored, incomplete or mismatched evidence: {path.relative_to(REPO)}; no model failure inferred"
                )
                continue
            trace_path = path.with_name("steps.jsonl")
            required = [trace_path, path.with_name("manifest.json"), path.with_name("controller.json")]
            if source != "native":
                required.append(path.with_name("chunks.json"))
            if not all(item.is_file() for item in required):
                errors.append(f"completed result is missing required evidence: {path.relative_to(REPO)}")
                continue
            signature = (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                trace_path.stat().st_size,
                trace_path.stat().st_mtime_ns,
            )
            if cache.get(path) != signature:
                steps = [json.loads(line) for line in trace_path.read_text().splitlines()]
                if (
                    len(steps) != 600
                    or [step["step"] for step in steps] != list(range(600))
                    or any(
                        not math.isfinite(step["actual_dt"]) or abs(step["actual_dt"] - 0.05) > 1e-10 for step in steps
                    )
                ):
                    errors.append(
                        f"completed result lacks 600 contiguous 20Hz physical steps: {path.relative_to(REPO)}"
                    )
                    continue
                cache[path] = signature
            valid += 1
        matrices.append(
            dict(
                study=study["name"],
                expected=len(study["expected"]),
                complete=valid,
                missing=len(study["expected"].keys() - present),
            )
        )
        if valid != len(study["expected"]):
            pending.append(f"{study['name']}: {valid}/{len(study['expected'])} complete registered trials")
    primary_jobs = launch_jobs(PRIMARY / "launch.json", "scripts/run_wam_independent_partition.py")
    schedulers = launch_jobs(DIAGNOSIS / "launch.json", "scripts/schedule_wam_diagnosis.py")
    for partition in range(4):
        inspect_partition(all_studies[0], partition, primary_jobs[partition], pending, errors)
        scheduler_root = DIAGNOSIS / "scheduling" / str(partition)
        if any((scheduler_root / marker).exists() for marker in ("error.json", "cancelled.json", "canceled.json")):
            errors.append(f"current scheduler has error/cancellation evidence: {partition}")
        scheduler_identity = owned_process(
            schedulers[partition]["pid"], "scripts/schedule_wam_diagnosis.py", pending, errors
        )
        scheduler_manifest = optional_json(scheduler_root / "manifest.json", pending)
        scheduler_complete = optional_json(scheduler_root / "complete.json", pending)
        primary_identity = process(primary_jobs[partition]["pid"])
        if scheduler_manifest is None or scheduler_manifest["primary_job"] != primary_jobs[partition]:
            errors.append(f"current scheduler prerequisite identity missing/different: {partition}")
        elif primary_identity and scheduler_manifest["start_ticks"] != primary_identity["ticks"]:
            errors.append(f"primary PID reused since scheduler registration: {partition}")
        if scheduler_complete is None:
            pending.append(f"scheduler{partition}: waiting for successful execute16 and execute40 completion")
            if scheduler_identity is None:
                errors.append(
                    f"scheduler{partition} ended without a successful complete marker; manual review required"
                )
        elif scheduler_complete.get("complete") is not True:
            errors.append(f"scheduler{partition} did not complete successfully")
        for study in all_studies[1:]:
            coordinator = optional_json(scheduler_root / f"{study['name']}_process.json", pending)
            inspect_partition(study, partition, coordinator, pending, errors)
    return dict(
        status="error" if errors else "pending" if pending else "ready",
        checked_at=now(),
        matrices=matrices,
        pending=pending,
        errors=errors,
        model_failures_inferred=False,
    )


def commands(all_studies):
    python = str(REPO / ".venv/bin/python")
    return [
        (
            study["name"],
            [
                python,
                "scripts/summarize_wam_independent.py",
                "--config",
                study["filename"],
                "--seed",
                "42",
                "--fingerprints",
                str(FINGERPRINTS),
            ],
            [study["root"] / "summary.json", study["root"] / "integrity.json"],
        )
        for study in all_studies
    ] + [("report", [python, "scripts/report_wam_independent.py", "--seed", "42"], [REPORT, REPORT_MANIFEST])]


def frozen_inputs(all_studies):
    paths = [REPO / name for _, name, *_ in CONFIGS]
    paths += [study["root"] / "simulation/fixtures.json" for study in all_studies]
    paths += [PRIMARY / "source_audit.json", FINGERPRINTS, PRIMARY / "launch.json", DIAGNOSIS / "launch.json"]
    paths += [REPO / "configs/wam_gripper_replay_20261020.json", REPO / "reports/wam_gripper_replay_20261020.md"]
    paths += [REPO / f"outputs/wam_gripper_replay_20261020/{name}.json" for name in ("manifest", "summary")]
    paths += [
        REPO / f"scripts/{name}.py"
        for name in (
            "finalize_wam_independent",
            "summarize_wam_independent",
            "verify_wam_acceptance",
            "report_wam_independent",
        )
    ]
    paths += list((REPO / "src/s2a").glob("*.py"))
    return {str(path): digest(path) for path in paths}


def save_progress(value):
    temporary = OUTPUT / f"progress.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(OUTPUT / "progress.json")
    with (OUTPUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def interrupted(signum, _frame):
    raise InterruptedError(f"finalizer received signal {signum}; original study evidence retained")


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.add_argument(
        "--check-only", action="store_true", help="read-only gate inspection; pending is not a model failure"
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        parser.error(f"run from the repository: {REPO}")
    started = False
    try:
        all_studies = studies()
        inputs = frozen_inputs(all_studies)
        state = snapshot(all_studies, {})
        if args.check_only:
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 1 if state["errors"] else 0
        if any(path.exists() for path in stage_outputs(all_studies)):
            raise FileExistsError(
                "existing summary/integrity/report requires manual review; no stage is skipped or overwritten"
            )
        OUTPUT.mkdir(parents=False, exist_ok=False)
        started = True
        schedule = commands(all_studies)
        manifest = dict(
            seed=args.seed,
            started_at=now(),
            pid=os.getpid(),
            python=sys.executable,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
            inputs_sha256=inputs,
            stages=[dict(name=name, command=cmd, outputs=[str(p) for p in outputs]) for name, cmd, outputs in schedule],
        )
        with (OUTPUT / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        cache = {}
        while True:
            for path, signature in inputs.items():
                if digest(Path(path)) != signature:
                    raise ValueError(f"frozen finalization input changed: {path}; manual review required")
            state = snapshot(all_studies, cache)
            save_progress(state)
            if state["errors"]:
                return 1
            if state["status"] == "ready":
                break
            time.sleep(POLL_SECONDS)
        owned = {}
        for name, command, outputs in schedule:
            for path, signature in inputs.items():
                if digest(Path(path)) != signature:
                    raise ValueError(f"frozen finalization input changed before stage {name}: {path}")
            for path in stage_outputs(all_studies):
                if path.exists() and str(path) not in owned:
                    raise FileExistsError(f"unexpected existing stage output requires manual review: {path}")
            for path, signature in owned.items():
                if digest(Path(path)) != signature:
                    raise ValueError(f"previous finalizer stage output changed: {path}")
            save_progress(dict(status="running_stage", stage=name, command=command, checked_at=now()))
            with (OUTPUT / f"{name}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=REPO, env=cpu_environment(), stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = child.wait()
                except BaseException:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    raise
            if code:
                raise RuntimeError(f"CPU stage {name} failed with exit {code}; outputs retained for manual review")
            if not all(path.is_file() for path in outputs):
                raise RuntimeError(f"CPU stage {name} exited without all its required outputs")
            owned.update({str(path): digest(path) for path in outputs})
            save_progress(dict(status="stage_complete", stage=name, checked_at=now(), outputs_sha256=owned))
        save_progress(dict(status="complete", checked_at=now(), outputs_sha256=owned))
        return 0
    except Exception as exc:
        state = dict(
            status="cancelled" if isinstance(exc, InterruptedError) else "error",
            checked_at=now(),
            error=repr(exc),
            model_failures_inferred=False,
            action="Manual review required; no automatic retry or stage reuse.",
        )
        if started:
            save_progress(state)
        else:
            print(json.dumps(state, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
