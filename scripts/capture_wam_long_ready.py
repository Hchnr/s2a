"""Capture live017 owned processes and GET-only service identity; no generation."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests
from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, write_new

from s2a.artifacts import REPO, sha256


def live_identity(role, registered, cuda, command, parent=None):
    """Bind an existing process to ticks, command, parentage and environment."""
    pid = registered["pid"]
    base = Path(f"/proc/{pid}")
    before = (base / "stat").read_text().rpartition(")")[2].split()
    environment = dict(
        item.split("=", 1) for item in (base / "environ").read_bytes().decode().split("\0") if "=" in item
    )
    actual_command = [p.decode() for p in (base / "cmdline").read_bytes().split(b"\0") if p]
    cwd = (base / "cwd").resolve()
    after = (base / "stat").read_text().rpartition(")")[2].split()
    assert int(before[19]) == int(after[19]) == registered["start_ticks"]
    assert after[0] not in ("Z", "X") and actual_command == command and cwd == REPO
    assert environment["CUDA_VISIBLE_DEVICES"] == cuda and environment["LD_LIBRARY_PATH"] == ""
    if parent is not None:
        assert int(after[1]) == parent
    if role == "physics":
        assert environment["MUJOCO_GL"] == environment["PYOPENGL_PLATFORM"] == "osmesa"
    return dict(
        role=role,
        pid=pid,
        start_ticks=int(after[19]),
        ppid=int(after[1]),
        proc_state=after[0],
        cwd=str(cwd),
        command=actual_command,
        environment={
            key: environment.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM")
        },
    )


def capture(launch_path, launch_sha, evidence):
    evidence.check(launch_path, launch_sha)
    launch = evidence.json(launch_path)
    part = launch["partition"]
    assert part in (0, 1, 2) and launch["physical_gpu"] == 5 + part and launch["port"] == 9655 + part
    config_path = REPO / f"configs/wam_long_raw_task{part}_20261021.json"
    config = evidence.json(config_path)
    gate_path, gate_sha = launch["prerequisite_gate_path"], launch["prerequisite_gate_sha256"]
    evidence.check(gate_path, gate_sha)
    gate = evidence.json(gate_path)
    for name, digest in gate["input_sha256"].items():
        evidence.check(name, digest)
    assert gate["all_prerequisites_passed"] and gate["input_sha256"][str(config_path)] == sha256(config_path)
    directory = REPO / config["output_dir"] / "orchestration"
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["partition"] == part and manifest["physical_gpu"] == 5 + part
    assert manifest["source_sha256"][str(Path(gate_path).resolve())] == gate_sha
    assert manifest["coordinator"]["pid"] == launch["coordinator_pid"]
    assert manifest["coordinator"]["start_ticks"] == launch["coordinator_start_ticks"]
    identities = [live_identity("coordinator", manifest["coordinator"], "", launch["command"])]
    model = evidence.json(directory / "model_process.json")
    matrix = evidence.json(directory / "matrix_process.json")
    for role, registered, cuda, command in (
        ("model", model, str(5 + part), manifest["model_command"]),
        ("matrix", matrix, "", manifest["job_command"]),
    ):
        identities.append(live_identity(role, registered, cuda, command, manifest["coordinator"]["pid"]))
    worker = evidence.json(
        REPO / config["output_dir"] / "simulation/logs/8400_masked_position_hold" / f"task{part}_init0_s42.process.json"
    )
    identities.append(live_identity("physics", worker, "", worker["command"], matrix["pid"]))
    saved = evidence.json(directory / "server_info.json")
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"http://127.0.0.1:{9655 + part}/info", timeout=5)
        response.raise_for_status()
        info = response.json()
    assert info == saved
    assert info["attention_backend"] == "sdpa" and info["acceptance_intervention"]["active_slot_mask"]
    assert not info["ema_weights"] and info["port"] == 9655 + part
    for name, digest in manifest["source_sha256"].items():
        evidence.check(name, digest)
    return dict(
        partition=part,
        physical_gpu=5 + part,
        port=9655 + part,
        ready=True,
        source_identity_verified=True,
        actual_http_info_equals_saved=True,
        actual_info=info,
        local_registered_processes=identities,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    assert args.launch.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(args.launch, args.launch_sha256)
    group = evidence.json(args.launch)
    assert group["registered_budget"] == dict(
        wam_trials=3, physics_steps=3600, maximum_h40_queries=450, fresh_initial_states=0
    )
    assert [p["partition"] for p in group["partitions"]] == [0, 1, 2]
    snapshots = [capture(p["path"], p["sha256"], evidence) for p in group["partitions"]]
    evidence.recheck()
    value = dict(
        schema_version="s2a.long-raw-live-ready-snapshot.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        ready=True,
        partitions=snapshots,
        gpu_snapshot=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True
        ),
        nvml_compute_snapshot=subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], text=True
        ),
        nvml_to_local_pid_mapping_claimed=False,
        local_network_operations="three GET /info requests only",
        new_model_requests=0,
        new_physics_steps=0,
        input_sha256=evidence.hashes,
    )
    write_new(args.output, value)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), ready=True)))


if __name__ == "__main__":
    main()
