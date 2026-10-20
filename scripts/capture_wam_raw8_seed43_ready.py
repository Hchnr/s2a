"""GET-only019 identity capture for unchanged generic services and CPU workers."""

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


def with_ticks(registration):
    value = dict(registration)
    state = (Path(f"/proc/{value['pid']}") / "stat").read_text().rpartition(")")[2].split()
    if state[0] in ("Z", "X"):
        raise ValueError("owned process already exited during readiness capture")
    value["start_ticks"] = int(state[19])
    return value


def capture(partition, evidence):
    part = partition["partition"]
    assert part in (0, 1, 2)
    config_path = REPO / f"configs/wam_raw8_seed43_task{part}_20261021.json"
    config = evidence.json(config_path)
    assert config["sampling_seeds"] == [43] and config["records"] == [f"task{part}_init0"]
    ck = config["checkpoints"][0]
    assert (ck["gpu"], ck["port"]) == (4 + part, 9744 + part)
    root = REPO / config["output_dir"]
    directory = root / "orchestration/8400_masked_position_hold"
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["coordinator_pid"] == partition["coordinator"]["pid"]
    assert manifest["config"] == config and manifest["config_sha256"] == sha256(config_path)
    assert manifest["physical_gpu"] == ck["gpu"] and len(manifest["jobs"]) == 1
    assert manifest["jobs"][0][0] == "position_hold"
    identities = [live_identity("coordinator", partition["coordinator"], "", partition["command"])]
    model = with_ticks(evidence.json(directory / "model_process.json"))
    matrix = with_ticks(evidence.json(directory / "position_hold_process.json"))
    assert model["gpu"] == ck["gpu"] and model["port"] == ck["port"]
    assert matrix["command"] == manifest["jobs"][0][1]
    identities.append(
        live_identity("model", model, str(ck["gpu"]), manifest["model_command"], partition["coordinator"]["pid"])
    )
    identities.append(live_identity("matrix", matrix, "", matrix["command"], partition["coordinator"]["pid"]))
    logs = root / "simulation/logs/8400_masked_position_hold"
    worker = with_ticks(evidence.json(logs / f"task{part}_init0_s43.process.json"))
    identities.append(live_identity("physics", worker, "", worker["command"], matrix["pid"]))
    snapshot = evidence.json(logs / "execution_snapshot.json")
    assert snapshot["jobs"] == [[f"task{part}_init0", 43]] and snapshot["cpu_workers"] == 1
    assert snapshot["coordinator_pid"] == matrix["pid"] and snapshot["config_sha256"] == sha256(config_path)
    for name, digest in snapshot["execution_sha256"].items():
        evidence.check(name, digest)
    saved = evidence.json(directory / "server_info.json")
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"http://127.0.0.1:{ck['port']}/info", timeout=5)
        response.raise_for_status()
        info = response.json()
    assert info == saved and info["port"] == ck["port"]
    assert info["attention_backend"] == "sdpa" and not info["ema_weights"]
    assert info["acceptance_intervention"]["active_slot_mask"]
    assert Path(info["checkpoint_path"]) == Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
    return dict(
        partition=part,
        physical_gpu=ck["gpu"],
        port=ck["port"],
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
    assert launch["schema_version"] == "s2a.raw8-seed43-root-launch.v1"
    evidence.check(launch["preflight"]["path"], launch["preflight"]["sha256"])
    preflight = evidence.json(launch["preflight"]["path"])
    assert preflight["schema_version"] == "s2a.raw8-seed43-preflight.v1" and preflight["complete"]
    assert (
        launch["registered_budget"]
        == preflight["registered_budget"]
        == dict(wam_trials=3, physics_steps=1800, h40_queries=225, new_oracle_trials=0, fresh_initial_states=0)
    )
    for name, digest in preflight["input_sha256"].items():
        evidence.check(name, digest)
    assert [part["partition"] for part in launch["partitions"]] == [0, 1, 2]
    for live, prepared in zip(launch["partitions"], preflight["partitions"], strict=True):
        assert live["partition"] == prepared["partition"] and live["command"] == prepared["command"]
    snapshots = [capture(part, evidence) for part in launch["partitions"]]
    evidence.recheck()
    value = dict(
        schema_version="s2a.raw8-seed43-live-ready-snapshot.v1",
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
    write_new(args.output, value)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), ready=True)))


if __name__ == "__main__":
    main()
