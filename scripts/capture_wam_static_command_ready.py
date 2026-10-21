"""Capture022's actual three live identities for one query phase; no HTTP."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_position import Evidence
from capture_wam_long_ready import live_identity
from run_wam_short_replan import cpu_environment, write_new

from s2a.artifacts import REPO, sha256


def capture(launch, phase, evidence):
    assert launch["schema_version"] == "s2a.static-command-root-launch.v1"
    assert launch["physical_gpu"] == 7 and launch["port"] == 10047
    budget = dict(reproduction=48, expanded=48, physics_steps=0, fresh_initial_states=0)
    assert launch["registered_budget"] == budget and not launch["automatic_retry"]
    reference = launch["prepared"]
    evidence.check(reference["path"], reference["sha256"])
    protocol = evidence.json(reference["path"])
    assert protocol["schema_version"] == "s2a.static-command-preparation.v1"
    assert protocol["query_budget"] == budget and len(protocol["requests"]) == 96
    for filename, signature in protocol["input_sha256"].items():
        evidence.check(filename, signature)
    for label in ("cpu_condition_proof", "independent_preflight"):
        item = launch[label]
        evidence.check(item["path"], item["sha256"])
    proof = evidence.json(launch["cpu_condition_proof"]["path"])
    assert proof["complete"] and proof["verified_pairs"] == 48
    assert proof["prepared"] == reference and proof["new_model_requests"] == proof["new_physics_steps"] == 0
    root = Path(protocol["root"])
    assert root == REPO / "outputs/wam_stability_20261021/static_command_state"
    directory = root / "orchestration"
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["prepared"] == reference and manifest["coordinator_pid"] == launch["coordinator"]["pid"]
    assert manifest["physical_gpu"] == 7 and manifest["port"] == 10047 and not manifest["automatic_retry"]
    command = launch["command"]
    assert Path(command[1]).resolve() == REPO / "scripts/run_wam_static_command.py"
    assert command[command.index("--prepared") + 1] == reference["path"]
    assert command[command.index("--prepared-sha256") + 1] == reference["sha256"]
    assert command[command.index("--seed") + 1] == "42" and "--dry-run" not in command
    coordinator = live_identity("coordinator", launch["coordinator"], "", command)
    model = evidence.json(directory / "model_ownership/pending.json")
    saved_model = evidence.json(directory / "model_ready.json")
    assert model["pid"] == manifest["model"]["pid"] == saved_model["pid"]
    assert model["start_ticks"] == manifest["model"]["start_ticks"] == saved_model["start_ticks"]
    assert model["command"] == manifest["model_command"] == launch["model_command"]
    live_model = live_identity("model", model, "7", model["command"], coordinator["pid"])
    config_ref = protocol["configs"][phase]
    evidence.check(config_ref["path"], config_ref["sha256"])
    query_command = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_state_matrix.py",
        "--config",
        config_ref["path"],
        "--seed",
        "42",
        "--checkpoint",
        "8400_masked",
    ]
    query = evidence.json(directory / f"{phase}_ownership/pending.json")
    assert query["command"] == query_command
    live_query = live_identity("query", query, "", query_command, coordinator["pid"])
    assert len({p["pid"] for p in (coordinator, live_model, live_query)}) == 3
    info = saved_model["info"]
    config = evidence.json(config_ref["path"])
    checkpoint = config["checkpoints"][0]
    assert (checkpoint["gpu"], checkpoint["port"]) == (7, 10047)
    assert Path(info["checkpoint_path"]) == Path(checkpoint["run"]) / "checkpoints/iter_000008400/model"
    assert info["port"] == 10047 and info["attention_backend"] == "sdpa" and not info["ema_weights"]
    assert info["acceptance_intervention"]["active_slot_mask"]
    assert info["static_command_protocol"] == reference
    assert Path(info["actual_caption_ledger"]) == root / "text_input_receipts"
    if phase == "expanded":
        admission = evidence.json(root / "reproduction_admission.json")
        assert admission["passed"] and admission["verified_queries"] == 48
        assert admission["full128_raw_and_normalized_exact"]
        for filename, signature in admission["input_sha256"].items():
            evidence.check(filename, signature)
    return dict(
        phase=phase,
        physical_gpu=7,
        port=10047,
        ready=True,
        concurrent_scope=f"coordinator, model and {phase} query worker only",
        actual_saved_info=info,
        local_registered_processes=[coordinator, live_model, live_query],
        other_phase_concurrently_claimed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--phase", choices=("reproduction", "expanded"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert Path.cwd().resolve() == REPO
    assert args.launch.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.read(REPO / "scripts/capture_wam_long_ready.py")
    evidence.check(args.launch, args.launch_sha256)
    result = capture(evidence.json(args.launch), args.phase, evidence)
    evidence.recheck()
    output = dict(
        schema_version="s2a.static-command-phase-live-ready.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        **result,
        launch=dict(path=str(args.launch.resolve()), sha256=args.launch_sha256),
        input_sha256=evidence.hashes,
        local_network_operations="none",
        nvml_to_local_pid_mapping_claimed=False,
        new_model_requests=0,
        new_physics_steps=0,
    )
    write_new(args.output, output)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), phase=args.phase, ready=True)))


if __name__ == "__main__":
    main()
