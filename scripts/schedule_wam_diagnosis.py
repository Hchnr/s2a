"""Use each owned GPU for pre-registered diagnostics after its primary partition."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from s2a.artifacts import code_fingerprint, write_json


def process_identity(pid):
    """Return state, start tick and argv; a reused PID is never a prerequisite."""
    base = Path("/proc") / str(pid)
    try:
        stat = base.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        return stat[0], stat[19], base.joinpath("cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partition", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("frozen diagnostic seed is 42")
    primary = Path("outputs/wam_independent_20261020")
    job = next(x for x in json.loads((primary / "launch.json").read_text()) if x["partition"] == args.partition)
    prerequisite = primary / "partitions" / str(args.partition)
    initial = process_identity(job["pid"])
    if initial and initial[0] != "Z" and b"scripts/run_wam_independent_partition.py" not in initial[2]:
        raise ValueError("prerequisite PID identity differs")
    output = Path("outputs/wam_horizon_diagnosis_20261020/scheduling") / str(args.partition)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "manifest.json",
        dict(code=code_fingerprint(), primary_job=job, start_ticks=initial[1] if initial else None, pid=os.getpid()),
    )
    while True:
        current = process_identity(job["pid"])
        alive = current is not None and current[0] != "Z"
        if alive and initial and current[1] != initial[1]:
            raise ValueError("prerequisite PID reused")
        if not alive:
            break
        if (prerequisite / "error.json").exists():
            raise RuntimeError("primary failed; no automatic retry or GPU takeover")
        time.sleep(10)
    complete = json.loads((prerequisite / "complete.json").read_text())
    exit_record = json.loads((prerequisite / "service_exit.json").read_text())
    service = process_identity(exit_record["pid"])
    if not complete["complete"] or (service is not None and service[0] != "Z"):
        raise ValueError("primary did not finish and release its owned service")
    for horizon in (16, 40):
        config = f"configs/wam_horizon_execute{horizon}_20261020.json"
        command = [
            str(Path(".venv/bin/python").absolute()),
            "scripts/run_wam_independent_partition.py",
            "--config",
            config,
            "--seed",
            str(args.seed),
            "--partition",
            str(args.partition),
        ]
        with (output / f"execute{horizon}.log").open("x") as log:
            process = subprocess.Popen(
                command,
                env=dict(os.environ, LD_LIBRARY_PATH="", CUDA_VISIBLE_DEVICES="4,5,6,7"),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write_json(output / f"execute{horizon}_process.json", dict(pid=process.pid, command=command))
            status = process.wait()
        if status:
            raise RuntimeError(f"execution-{horizon} diagnostic failed; evidence retained")
    write_json(output / "complete.json", dict(complete=True))


if __name__ == "__main__":
    main()
