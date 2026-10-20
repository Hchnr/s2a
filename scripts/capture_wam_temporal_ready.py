"""Bind020 actual owned processes and exactly three GET/info identities."""

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


def capture(part, evidence):
    index = part["partition"]
    assert index in (0, 1, 2) and part["physical_gpu"] == 4 + index and part["port"] == 9844 + index
    config_path = REPO / f"configs/wam_temporal_equal_task{index}_20261021.json"
    config = evidence.json(config_path)
    root = REPO / config["output_dir"]
    gate_path, gate_sha = part["prerequisite_gate_path"], part["prerequisite_gate_sha256"]
    evidence.check(gate_path, gate_sha)
    gate = evidence.json(gate_path)
    assert gate["all_prerequisites_passed"] and gate["schema_version"] == "s2a.temporal-equal-launch-gate.v1"
    for name, digest in gate["input_sha256"].items():
        evidence.check(name, digest)
    directory = root / "orchestration"
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["schema_version"] == "s2a.temporal-equal-service.v1"
    assert manifest["partition"] == index and manifest["physical_gpu"] == 4 + index
    assert manifest["coordinator"]["pid"] == part["coordinator"]["pid"]
    assert manifest["coordinator"]["start_ticks"] == part["coordinator"]["start_ticks"]
    assert manifest["config"] == config and manifest["config_sha256"] == sha256(config_path)
    assert manifest["source_sha256"][str(Path(gate_path).resolve())] == gate_sha
    identities = [live_identity("coordinator", part["coordinator"], "", part["command"])]
    model = evidence.json(directory / "model_process.json")
    matrix = evidence.json(directory / "matrix_process.json")
    identities.append(
        live_identity("model", model, str(4 + index), manifest["model_command"], part["coordinator"]["pid"])
    )
    identities.append(live_identity("matrix", matrix, "", manifest["job_command"], part["coordinator"]["pid"]))
    logs = root / "simulation/logs/8400_masked_temporal_equal"
    worker = evidence.json(logs / f"task{index}_init0_s42.process.json")
    identities.append(live_identity("physics", worker, "", worker["command"], matrix["pid"]))
    snapshot = evidence.json(logs / "execution_snapshot.json")
    assert snapshot["coordinator_pid"] == matrix["pid"] and snapshot["cpu_workers"] == 1
    assert snapshot["jobs"] == [[f"task{index}_init0", 42]]
    assert snapshot["config_sha256"] == sha256(config_path)
    assert snapshot["identity"] == ["wam", "temporal_equal", "8400_masked"]
    for name, digest in snapshot["execution_sha256"].items():
        evidence.check(name, digest)
    saved = evidence.json(directory / "server_info.json")
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"http://127.0.0.1:{9844 + index}/info", timeout=5)
        response.raise_for_status()
        info = response.json()
    assert info == saved and info["port"] == 9844 + index
    assert (
        info["attention_backend"] == "sdpa"
        and not info["ema_weights"]
        and info["acceptance_intervention"]["active_slot_mask"]
    )
    assert Path(info["checkpoint_path"]) == Path(config["checkpoints"][0]["run"]) / "checkpoints/iter_000008400/model"
    for name, digest in manifest["source_sha256"].items():
        evidence.check(name, digest)
    return dict(
        partition=index,
        physical_gpu=4 + index,
        port=9844 + index,
        ready=True,
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
    assert Path.cwd().resolve() == REPO
    assert args.launch.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.read("scripts/capture_wam_long_ready.py")
    evidence.check(args.launch, args.launch_sha256)
    launch = evidence.json(args.launch)
    assert launch["schema_version"] == "s2a.temporal-equal-root-launch.v1"
    assert launch["registered_budget"] == dict(
        wam_trials=3, physics_steps=1800, maximum_h40_queries=225, fresh_initial_states=0
    )
    assert [part["partition"] for part in launch["partitions"]] == [0, 1, 2]
    snapshots = [capture(part, evidence) for part in launch["partitions"]]
    evidence.recheck()
    result = dict(
        schema_version="s2a.temporal-equal-live-ready-snapshot.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        ready=True,
        partitions=snapshots,
        input_sha256=evidence.hashes,
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
    )
    write_new(args.output, result)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), ready=True)))


if __name__ == "__main__":
    main()
