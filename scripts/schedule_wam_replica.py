"""Reuse completed offline GPUs for disjoint online jobs with verified replicas."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def wait_file(path, predicate):
    while not path.exists():
        time.sleep(10)
    value = json.loads(path.read_text())
    if not predicate(value):
        raise ValueError(f"prerequisite failed: {path}")


def stop_owned(pid, required):
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return
    arguments = path.read_bytes().replace(b"\0", b" ").decode()
    if not all(part in arguments for part in required):
        raise ValueError(f"process identity differs for PID {pid}")
    os.kill(pid, signal.SIGTERM)
    # A paused coordinator must be continued to handle its pending termination.
    try:
        os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        return
    for _ in range(60):
        if not path.exists():
            return
        try:
            if path.with_name("stat").read_text().split()[2] == "Z":
                return
        except FileNotFoundError:
            return
        time.sleep(1)
    raise TimeoutError(f"owned process {pid} has not exited; no force kill")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_masked_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=("8400", "8600", "8800", "warm6200"), required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    plans = {
        "8400": dict(gpu=2, port=8602, old_port=8582, jobs="1,3,5,7,9,11"),
        "8600": dict(gpu=3, port=8603, old_port=8583, jobs="1,3,5,7,9,11"),
        "8800": dict(gpu=4, port=8604, old_port=8584, jobs="2,3,4,5"),
        "warm6200": dict(gpu=1, port=8601, old_port=8595, jobs="1,3,5,7,9,11"),
    }
    plan = plans[args.role]
    output = root / "scheduling" / args.role
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", dict(plan=plan, code=code_fingerprint(), config=c))

    def run_jobs(checkpoint, port, jobs):
        command = [
            str(Path(".venv/bin/python").absolute()),
            "scripts/run_wam_simulation_matrix.py",
            "--config",
            args.config,
            "--seed",
            str(args.seed),
            "--stage",
            "wam",
            "--checkpoint",
            checkpoint,
            "--server-port",
            str(port),
            "--job-indices",
            jobs,
            "--resume",
        ]
        with (output / f"jobs_{checkpoint}.log").open("w") as stream:
            result = subprocess.call(
                command, env=dict(os.environ, LD_LIBRARY_PATH=""), stdout=stream, stderr=subprocess.STDOUT
            )
        if result:
            raise RuntimeError(f"online partition failed: {checkpoint}")

    if args.role == "warm6200":
        first = root / "simulation/8800_masked/task0_init0_s42/result.json"
        wait_file(first, lambda r: r["error"] is None and r["steps"] == r["expected_steps"] == 600)
        transition = json.loads((root / "simulation/partition_transition.json").read_text())
        parent = next(r["pid"] for r in transition if r["action"].startswith("pause coordinator"))
        stop_owned(parent, ["run_wam_simulation_matrix.py", "8800_masked", "8595"])
        run_jobs("8800_masked", 8595, "0,1")
        candidates = []
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                parts = path.read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            if b"scripts/serve_wam_acceptance.py" in parts and b"8595" in parts:
                candidates.append(int(path.parent.name))
        if len(candidates) != 1:
            raise ValueError("cannot uniquely identify the completed 8800 online replica")
        stop_owned(candidates[0], ["serve_wam_acceptance.py", "--port 8595"])
    else:
        wait_file(root / "state_queries" / args.role / "complete.json", lambda r: r["cases"] == 384)
        process_path = root / "services" / args.role / f"server_process_{plan['old_port']}.json"
        process = json.loads(process_path.read_text())
        stop_owned(process["pid"], ["action128_server", f"--port {plan['old_port']}", str(root)])

    command = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--seed",
        str(args.seed),
        "--checkpoint",
        args.role,
        "--port",
        str(plan["port"]),
        "--active-slot-mask",
        "--exact-published-caption",
    ]
    if args.role == "warm6200":
        current = next(x for x in c["checkpoints"] if x["name"] == "8800_masked")
        command += ["--warm-start-contract-run", current["run"]]
    env = dict(
        os.environ,
        LD_LIBRARY_PATH="",
        CUDA_VISIBLE_DEVICES=str(plan["gpu"]),
        MASTER_PORT=str(30000 + plan["gpu"]),
        COSMOS_TRAINING="0",
        PYTHONPATH=c["cosmos_repo"] + ":" + str(Path("src").absolute()),
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "replica_cache"),
    )
    log = (output / "service.log").open("w")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(output / "process.json", dict(pid=process.pid, command=command, gpu=plan["gpu"]))
    session = requests.Session()
    session.trust_env = False
    endpoint = f"http://127.0.0.1:{plan['port']}"
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("replica exited during startup; evidence retained")
        try:
            response = session.get(endpoint + "/info", timeout=2)
            response.raise_for_status()
            info = response.json()
            break
        except requests.RequestException:
            time.sleep(5)
    else:
        raise TimeoutError("replica did not become ready")
    case = root / "state_queries" / f"{args.role}_masked" / "task0_ep1172_approach_h16_task_only_s42"
    request = json.loads((case / "request.json").read_text())
    response = session.post(endpoint + "/generate", json=request, timeout=900)
    response.raise_for_status()
    content = response.json()
    expected = json.loads((case / "response.json").read_text())
    actual_action = np.asarray(content["outputs"][0]["content"]["action"])
    expected_action = np.asarray(expected["outputs"][0]["content"]["action"])
    check = dict(
        reference_sha256=sha256(case / "response.json"),
        server_info=info,
        max_abs=float(np.max(np.abs(actual_action - expected_action))),
        identical=bool(np.array_equal(actual_action, expected_action)),
    )
    write_json(output / "replica_check.json", check)
    if not check["identical"]:
        raise ValueError("replica does not reproduce the paired frozen response")
    run_jobs(args.role + "_masked", plan["port"], plan["jobs"])
    stop_owned(process.pid, ["serve_wam_acceptance.py", f"--port {plan['port']}"])
    process.wait(timeout=10)
    log.close()
    write_json(output / "complete.json", dict(complete=True, plan=plan))


if __name__ == "__main__":
    main()
