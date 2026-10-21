"""Bind the one021 continuation's four live identities with GET /info only."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests
from audit_wam_position import Evidence
from capture_wam_long_ready import live_identity
from run_wam_short_replan import cpu_environment, write_new

from s2a.artifacts import REPO, sha256


def capture(launch, evidence):
    assert launch["partition"] == 2 and launch["physical_gpu"] == 6 and launch["port"] == 9946
    config_path = REPO / "configs/wam_temporal_equal_task2_continuation_20261021.json"
    assert Path(launch["config_path"]) == config_path
    evidence.check(config_path, launch["config_sha256"])
    config = evidence.json(config_path)
    root = REPO / config["output_dir"]
    gate_path, gate_sha = launch["prerequisite_gate_path"], launch["prerequisite_gate_sha256"]
    evidence.check(gate_path, gate_sha)
    gate = evidence.json(gate_path)
    assert gate["all_prerequisites_passed"]
    for name, digest in gate["input_sha256"].items():
        evidence.check(name, digest)
    directory = root / "orchestration"
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["schema_version"] == "s2a.temporal-zero-step-continuation-service.v1"
    assert manifest["partition"] == 2 and manifest["physical_gpu"] == 6
    assert manifest["coordinator"]["pid"] == launch["coordinator"]["pid"]
    assert manifest["coordinator"]["start_ticks"] == launch["coordinator"]["start_ticks"]
    assert manifest["config"] == config and manifest["config_sha256"] == sha256(config_path)
    assert manifest["source_sha256"][str(Path(gate_path).resolve())] == gate_sha
    identities = [live_identity("coordinator", launch["coordinator"], "", launch["command"])]
    model = evidence.json(directory / "model_process.json")
    matrix = evidence.json(directory / "matrix_process.json")
    identities.append(live_identity("model", model, "6", manifest["model_command"], launch["coordinator"]["pid"]))
    identities.append(live_identity("matrix", matrix, "", manifest["job_command"], launch["coordinator"]["pid"]))
    logs = root / "simulation/logs/8400_masked_temporal_equal"
    worker = evidence.json(logs / "task2_init0_s42.process.json")
    identities.append(live_identity("physics", worker, "", worker["command"], matrix["pid"]))
    snapshot = evidence.json(logs / "execution_snapshot.json")
    assert snapshot["coordinator_pid"] == matrix["pid"] and snapshot["cpu_workers"] == 1
    assert snapshot["jobs"] == [["task2_init0", 42]]
    assert snapshot["config_sha256"] == sha256(config_path)
    assert snapshot["identity"] == ["wam", "temporal_equal", "8400_masked"]
    for name, digest in snapshot["execution_sha256"].items():
        evidence.check(name, digest)
    ready_path = logs / "task2_init0_s42.handshake/ready.json"
    assert Path(worker["ready_path"]) == ready_path
    evidence.check(ready_path, worker["ready_sha256"])
    child_ready = evidence.json(ready_path)
    child_go = evidence.json(ready_path.with_name("go.json"))
    assert child_ready["schema_version"] == "s2a.cpu-worker-ready.v1"
    assert child_go["schema_version"] == "s2a.cpu-worker-go.v1"
    assert child_ready["identity"] == worker["identity"] == child_go["identity"]
    assert child_ready["token"] == child_go["token"]
    assert child_go["ready_sha256"] == sha256(ready_path)
    assert child_go["parent_identity"] == child_ready["parent_identity"]
    assert child_ready["parent_identity"] == {key: matrix[key] for key in ("pid", "start_ticks")}
    saved = evidence.json(directory / "server_info.json")
    with requests.Session() as session:
        session.trust_env = False
        response = session.get("http://127.0.0.1:9946/info", timeout=5)
        response.raise_for_status()
        info = response.json()
    assert info == saved and info["port"] == 9946
    assert info["attention_backend"] == "sdpa" and not info["ema_weights"]
    assert info["acceptance_intervention"]["active_slot_mask"]
    assert Path(info["checkpoint_path"]) == Path(config["checkpoints"][0]["run"]) / "checkpoints/iter_000008400/model"
    for name, digest in manifest["source_sha256"].items():
        evidence.check(name, digest)
    return dict(
        partition=2,
        physical_gpu=6,
        port=9946,
        ready=True,
        actual_http_info_equals_saved=True,
        actual_info=info,
        local_registered_processes=identities,
        child_ready_sha256=sha256(ready_path),
        child_go_sha256=sha256(ready_path.with_name("go.json")),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    assert Path.cwd().resolve() == REPO
    assert args.launch.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.read("scripts/capture_wam_long_ready.py")
    evidence.check(args.launch, args.launch_sha256)
    launch = evidence.json(args.launch)
    assert launch["schema_version"] == "s2a.temporal-zero-step-continuation-root-launch.v1"
    value = capture(launch, evidence)
    evidence.recheck()
    result = dict(
        schema_version="s2a.temporal-zero-step-continuation-ready.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        concurrent_scope="continuation partition2 only",
        **value,
        input_sha256=evidence.hashes,
        gpu_snapshot=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True
        ),
        nvml_compute_snapshot=subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], text=True
        ),
        nvml_to_local_pid_mapping_claimed=False,
        local_network_operations="one GET /info request only",
        new_model_requests=0,
        new_physics_steps=0,
    )
    write_new(args.output, result)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), ready=True)))


if __name__ == "__main__":
    main()
