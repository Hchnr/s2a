"""Own024's single GPU6 numerical worker after an immutable CPU source review."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import identity, owned_child, write_once


def reference(path):
    return dict(path=str(Path(path).absolute()), sha256=sha256(path))


def checked_json(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"source-bound JSON differs: {path}")
    return json.loads(Path(path).read_text())


def interrupted(signum, frame):
    raise InterruptedError(f"024 owner received signal {signum}")


def external_identity(repo):
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], text=True)
    if changed.strip():
        raise ValueError("024 requires the recorded external tracked source tree to be clean")
    return dict(head=head, tracked_source_clean=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--source-peer", type=Path, required=True)
    parser.add_argument("--source-peer-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        raise ValueError("024 must run from the S2A repository")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("024 owner requires explicitly empty CUDA/LD")
    prepared = checked_json(args.prepared, args.prepared_sha256)
    peer = checked_json(args.source_peer, args.source_peer_sha256)
    if peer.get("passed") is not True or peer.get("prepared") != reference(args.prepared):
        raise ValueError("024 independent CPU source review did not admit this preparation")
    for mapping in (prepared["input_sha256"], prepared["snapshot_sha256"], peer["input_sha256"]):
        for path, digest in mapping.items():
            if sha256(path) != digest:
                raise ValueError(f"024 source changed: {path}")
    if str(Path(__file__).resolve()) not in prepared["source_sha256"]:
        raise ValueError("024 preparation omitted its actual owner source")
    for path, entry in prepared["weight_files"].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
            raise ValueError(f"024 actual8400 weight stat changed: {path}")
    c = prepared["config"]
    if len(c["checkpoints"]) != 1 or c["checkpoints"][0]["gpu"] != 6 or c["denoise_budget"] != 27:
        raise ValueError("024 owner requires its registered singleGPU6/27-forward protocol")
    cosmos_identity = external_identity(c["cosmos_repo"])
    root = Path(prepared["output_root"])
    if root != (REPO / c["output_dir"]).resolve() or not root.is_relative_to(REPO / "outputs"):
        raise ValueError("024 output identity differs")
    command = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/audit_wam_current_numerics.py",
        "--run",
        "--seed",
        "42",
        "--prepared",
        str(args.prepared.absolute()),
        "--prepared-sha256",
        args.prepared_sha256,
    ]
    if args.dry_run:
        print(json.dumps(dict(command=command, physical_gpu=6, denoise_calls=27, sampling_calls=0, physics_steps=0)))
        return
    directory = root / "orchestration"
    directory.mkdir(exist_ok=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="6",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=c["cosmos_repo"] + ":" + str(REPO / "src"),
        MASTER_PORT="31046",
        COSMOS_TRAINING="0",
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_CACHE_DIR=str(root / "model_cache"),
    )
    ownership = directory / "model_ownership"
    ownership.mkdir()
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = subprocess.check_output(
            ["nvidia-smi", "--id=6", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        if int(memory.strip()) != 0:
            raise RuntimeError("024 requires GPU6 memory strictly0; existing processes are untouched")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 31046))
        with (
            (directory / "model.log").open("x") as stream,
            owned_child(command, environment=environment, cwd=REPO, folder=ownership, stream=stream) as (model, birth),
        ):
            write_once(
                directory / "launch.json",
                dict(
                    owner=identity(os.getpid()),
                    model=birth,
                    command=command,
                    environment={k: environment[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MASTER_PORT")},
                    prepared=reference(args.prepared),
                    source_peer=reference(args.source_peer),
                    external_cosmos=cosmos_identity,
                    automatic_retry=False,
                    denoise_budget=27,
                    sampling_budget=0,
                    physics_budget=0,
                ),
            )
            deadline = time.monotonic() + 30
            started = root / "worker_started.json"
            observations = []
            while True:
                if model.poll() is not None:
                    raise ChildProcessError("024 worker exited before its actual Python startup receipt")
                if started.exists():
                    observed = identity(model.pid)
                    observations.append(observed)
                    if observed["start_ticks"] != birth["start_ticks"] or observed["ppid"] != os.getpid():
                        raise ValueError("024 worker ownership differs")
                    if observed["command"] != command:
                        raise ValueError("024 worker command differs after its actual Python startup")
                    expected = dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH="")
                    for key, value in expected.items():
                        if observed["environment"][key] is not None and observed["environment"][key] != value:
                            raise ValueError("024 worker actual environment differs from its launch")
                    if all(observed["environment"][key] is not None for key in expected):
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError("024 worker exec identity was not observable")
                time.sleep(0.01)
            write_once(directory / "identity_observations.json", observations)
            write_once(directory / "actual_model_identity.json", observed)
            code = model.wait(timeout=1800)
            write_once(directory / "exit.json", dict(returncode=code, actual_wait=True, automatic_retry=False))
            if code:
                raise ChildProcessError(f"024 numerical worker returned {code}; evidence retained without retry")
            final_cosmos = external_identity(c["cosmos_repo"])
            write_once(directory / "external_cosmos_after.json", final_cosmos)
            if final_cosmos != cosmos_identity:
                raise ValueError("024 external tracked source identity changed during the worker")
    print(json.dumps(dict(exit=reference(directory / "exit.json"), release=reference(ownership / "release.json"))))


if __name__ == "__main__":
    main()
