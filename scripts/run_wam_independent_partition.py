"""Run a disjoint frozen matrix partition with a replay-verified owned service."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_independent_20261020.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partition", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"] or len(c["checkpoints"]) not in (1, 2, 4):
        raise ValueError("requires frozen seed and a checkpoint count dividing the four allowed GPUs")
    replicas = 4 // len(c["checkpoints"])
    ck = c["checkpoints"][args.partition // replicas]
    records = json.loads((root / "simulation/fixtures.json").read_text())["records"]
    count = len(records) * len(c["sampling_seeds"])
    indices = list(range(args.partition % replicas, count, replicas))
    gpu = (4, 5, 6, 7)[args.partition]
    port = 8900 + gpu
    output = root / "partitions" / str(args.partition)
    output.mkdir(parents=True, exist_ok=False)
    reference = Path("outputs/wam_checkpoint_acceptance_20261019v2/state_queries") / ck["name"]
    case = reference / "task0_ep1172_approach_h40_task_only_s42"
    request = json.loads((case / "request.json").read_text())
    expected = np.asarray(json.loads((case / "response.json").read_text())["outputs"][0]["content"]["action"])
    write_json(
        output / "manifest.json",
        dict(
            config=c,
            code=code_fingerprint(),
            checkpoint=ck,
            job_indices=indices,
            gpu=gpu,
            port=port,
            reference_request_sha256=sha256(case / "request.json"),
            reference_response_sha256=sha256(case / "response.json"),
        ),
    )
    command = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--config",
        args.config,
        "--seed",
        str(args.seed),
        "--checkpoint",
        ck["name"],
        "--port",
        str(port),
        "--active-slot-mask",
    ]
    if "spec_run" in ck:
        command += ["--warm-start-contract-run", ck["spec_run"]]
    env = dict(
        os.environ,
        LD_LIBRARY_PATH="",
        CUDA_VISIBLE_DEVICES=str(gpu),
        MASTER_PORT=str(30500 + gpu),
        COSMOS_TRAINING="0",
        PYTHONPATH=c["cosmos_repo"] + ":" + str(Path("src").absolute()),
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "model_cache" / str(args.partition)),
    )
    with (output / "service.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / "process.json", dict(pid=process.pid, command=command, coordinator_pid=os.getpid()))
        try:
            with requests.Session() as session:
                session.trust_env = False
                endpoint = f"http://127.0.0.1:{port}"
                for _ in range(180):
                    if process.poll() is not None:
                        raise RuntimeError("owned model process exited during startup")
                    try:
                        response = session.get(endpoint + "/info", timeout=2)
                        response.raise_for_status()
                        info = response.json()
                        break
                    except requests.RequestException:
                        time.sleep(5)
                else:
                    raise TimeoutError("model service startup timeout")
                expected_path = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
                if Path(info["checkpoint_path"]).resolve() != expected_path.resolve() or info["ema_weights"]:
                    raise ValueError("incorrect checkpoint identity")
                if not info["acceptance_intervention"]["active_slot_mask"]:
                    raise ValueError("mask correction missing")
                if "spec_run" in ck and not info["acceptance_intervention"]["parameters_unchanged"]:
                    raise ValueError("warm-start migration modified parameters")
                response = session.post(endpoint + "/generate", json=request, timeout=900)
                response.raise_for_status()
                actual = np.asarray(response.json()["outputs"][0]["content"]["action"])
                identical = bool(np.array_equal(actual, expected))
                write_json(
                    output / "replica_check.json",
                    dict(identical=identical, max_abs=float(np.max(abs(actual - expected))), info=info),
                )
                if not identical:
                    raise ValueError("service differs from audited H40 prediction")
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
                ck["name"],
                "--server-port",
                str(port),
                "--job-indices",
                ",".join(map(str, indices)),
                "--resume",
            ]
            with (output / "jobs.log").open("x") as stream:
                status = subprocess.call(
                    command,
                    env=dict(os.environ, LD_LIBRARY_PATH="", CUDA_VISIBLE_DEVICES=""),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if status:
                raise RuntimeError("physics partition failed; original evidence retained")
            write_json(output / "complete.json", dict(complete=True, job_indices=indices))
        except Exception as exc:
            write_json(output / "error.json", dict(error=repr(exc)))
            raise
        finally:
            # Only the child created by this invocation is eligible for cleanup.
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=60)
            write_json(output / "service_exit.json", dict(pid=process.pid, returncode=process.returncode))


if __name__ == "__main__":
    main()
