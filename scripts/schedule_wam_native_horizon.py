"""Run frozen H40 jobs only after each owned GPU finishes its primary jobs."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from schedule_wam_replica import stop_owned, wait_file

from s2a.artifacts import code_fingerprint, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    config_path = "configs/wam_checkpoint_native_h40_20261019v2.json"
    c, root = load_config(config_path)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    plans = {
        0: dict(primary="warm6200_masked", jobs="8", old_port=8590, prerequisites=[0, 2, 4, 6, 8, 10]),
        1: dict(replica="warm6200", jobs="0,1,2"),
        2: dict(replica="8400", jobs="6"),
        3: dict(replica="8600", jobs="7"),
        4: dict(replica="8800", jobs="3,4,5"),
        5: dict(primary="8800_masked", jobs="9", old_port=8594, prerequisites=[6, 7, 8, 9, 10, 11]),
        6: dict(primary="8400_masked", jobs="11", old_port=8592, prerequisites=[0, 2, 4, 6, 8, 10]),
        7: dict(primary="8600_masked", jobs="10", old_port=8593, prerequisites=[0, 2, 4, 6, 8, 10]),
    }
    if sorted(int(i) for p in plans.values() for i in p["jobs"].split(",")) != list(range(12)):
        raise ValueError("supplementary jobs must cover the matrix exactly once")
    plan = plans[args.gpu]
    output = root / "scheduling" / str(args.gpu)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", dict(plan=plan, config=c, code=code_fingerprint()))
    primary = root.parent
    if "replica" in plan:
        wait_file(primary / "scheduling" / plan["replica"] / "complete.json", lambda x: x["complete"])
    else:
        wait_file(primary / "state_queries" / plan["primary"] / "complete.json", lambda x: x["cases"] == 384)
        fixtures = json.loads((root / "simulation/fixtures.json").read_text())["records"]
        jobs = [(r, s) for r in fixtures for s in c["sampling_seeds"]]
        for index in plan["prerequisites"]:
            record, seed = jobs[index]
            path = primary / "simulation" / plan["primary"] / f"{record['name']}_s{seed}/result.json"
            wait_file(path, lambda x: x["error"] is None and x["steps"] == x["expected_steps"] == 600)
        candidates = []
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                parts = path.read_bytes().split(b"\0")
                cwd = path.parent.joinpath("cwd").resolve()
            except FileNotFoundError:
                continue
            if (
                b"scripts/serve_wam_acceptance.py" in parts
                and str(plan["old_port"]).encode() in parts
                and cwd == Path.cwd()
            ):
                candidates.append(int(path.parent.name))
        if len(candidates) != 1:
            raise ValueError("owned primary service cannot be uniquely identified")
        stop_owned(candidates[0], ["serve_wam_acceptance.py", f"--port {plan['old_port']}"])

    # Defer runtime dependencies until the primary jobs have finished.
    import time

    import numpy as np
    import requests

    port = 8700 + args.gpu
    command = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--seed",
        str(args.seed),
        "--checkpoint",
        "8800",
        "--port",
        str(port),
        "--active-slot-mask",
        "--exact-published-caption",
    ]
    env = dict(
        os.environ,
        LD_LIBRARY_PATH="",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        MASTER_PORT=str(30100 + args.gpu),
        COSMOS_TRAINING="0",
        PYTHONPATH=c["cosmos_repo"] + ":" + str(Path("src").absolute()),
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "replica_cache"),
    )
    with (output / "service.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / "process.json", dict(pid=process.pid, command=command, gpu=args.gpu))
        session = requests.Session()
        session.trust_env = False
        endpoint = f"http://127.0.0.1:{port}"
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError("native-window replica exited during startup")
            try:
                response = session.get(endpoint + "/info", timeout=2)
                response.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(5)
        else:
            raise TimeoutError("native-window replica not ready")
        case = primary / "state_queries/8800_masked/task0_ep1172_approach_h40_task_only_s42"
        response = session.post(
            endpoint + "/generate", json=json.loads((case / "request.json").read_text()), timeout=900
        )
        response.raise_for_status()
        expected = np.asarray(json.loads((case / "response.json").read_text())["outputs"][0]["content"]["action"])
        actual = np.asarray(response.json()["outputs"][0]["content"]["action"])
        identical = bool(np.array_equal(actual, expected))
        write_json(
            output / "replica_check.json", dict(identical=identical, max_abs=float(np.max(abs(actual - expected))))
        )
        if not identical:
            raise ValueError("native-window replica did not reproduce the frozen full response")
        command = [
            str(Path(".venv/bin/python").absolute()),
            "scripts/run_wam_simulation_matrix.py",
            "--config",
            config_path,
            "--seed",
            str(args.seed),
            "--stage",
            "wam",
            "--checkpoint",
            "8800_masked",
            "--server-port",
            str(port),
            "--job-indices",
            plan["jobs"],
            "--resume",
        ]
        with (output / "jobs.log").open("w") as stream:
            status = subprocess.call(
                command, env=dict(os.environ, LD_LIBRARY_PATH=""), stdout=stream, stderr=subprocess.STDOUT
            )
        if status:
            raise RuntimeError("native-window physics partition failed; evidence retained")
        stop_owned(process.pid, ["serve_wam_acceptance.py", f"--port {port}"])
        process.wait(timeout=10)
    write_json(output / "complete.json", dict(complete=True, plan=plan))


if __name__ == "__main__":
    main()
