"""Explicit021 outer audit: original020 partitions0/1 plus one zero-step continuation.

The original89 full-ready checker is unchanged. Its frozen case/first-query
and scope-independent helpers are reused; this entry authenticates a different
registered lifecycle with historical8+new4 identities, never12 concurrent ones.
"""

import argparse
import hashlib
import json
import re
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import audit_wam_temporal_wam as original
from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, process_record, write_new

from s2a.artifacts import REPO, sha256

REGISTRATION = "ea3f26d"
PLAN = "plans/v0.0.1/021_temporal_zero_step_continuation_20261021.md"
CONFIG = "configs/wam_temporal_equal_task2_continuation_20261021.json"
ORIGINAL_HASHES = dict(
    launch="d702b78dcd3f781b7326417ab404986f34d2d6b877cfe41fdd1344731ddb179e",
    partial_ready="869d4b4732457ca89da54a2eae8b2988b23956b1aa6396cfc07ec88b5446f9a2",
    incident="8588d03ea99c403bd155663788fbad01215750b3447e26f7469ccd2a9698b63f",
    zero_effect="bc117d00bcf2ed79f28356faf78c1cc8ce6a2744b6bfd54e2158e39f876d1528",
)
CROSSCHECK_SHA = "0006f3da0694cadbdb537483a792009bb90b7dc169db27bfe8136a2b279b993b"
CONTINUATION_BUDGET = dict(
    continuation_wam_trials=1,
    physics_steps=600,
    maximum_h40_queries=75,
    new_oracle_trials=0,
    fresh_initial_states=0,
)


def registered(relative, evidence):
    digest = hashlib.sha256(
        subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
    ).hexdigest()
    evidence.check(REPO / relative, digest)
    return digest


def same_effect(old, current):
    """Only the predeclared output, port and explanatory notes may change."""
    assert old["sampling_seeds"] == current["sampling_seeds"] == [42]
    assert current["output_dir"] == "outputs/wam_stability_20261021/temporal_equal_development/task2_continuation"
    assert current["records"] == ["task2_init0"]
    expected = json.loads(json.dumps(old))
    expected.update(output_dir=current["output_dir"], notes=current["notes"])
    expected["checkpoints"] = [dict(old["checkpoints"][0], port=9946)]
    assert current == expected and current["checkpoints"][0]["gpu"] == 6
    assert current["notes"][:-1] == old["notes"]
    assert (current["online_max_steps"], current["online_chunk_size"], current["max_observation_interval_steps"]) == (
        600,
        40,
        8,
    )
    return True


def check_partial_ready(value):
    assert value["schema_version"] == "s2a.temporal-equal-live-ready-partial-snapshot.v1"
    assert value["complete_group_ready"] is False and value["captured_partitions"] == [0, 1]
    assert value["missing_partition"] == 2 and [p["partition"] for p in value["partitions"]] == [0, 1]
    assert value["launch_sha256"] == ORIGINAL_HASHES["launch"]
    assert value["local_network_operations"] == "two GET /info requests only"
    assert value["new_model_requests"] == value["new_physics_steps"] == 0
    identities = [identity for part in value["partitions"] for identity in part["local_registered_processes"]]
    assert len(identities) == len({(p["pid"], p["start_ticks"]) for p in identities}) == 8
    return True


def old_worker_matches(command):
    """Find only an exact original task2 worker command, without guessing its PID."""
    if len(command) < 2 or Path(command[1]).resolve() != REPO / "scripts/run_wam_temporal_matrix.py":
        return False
    try:
        config = command[command.index("--config") + 1]
        record = command[command.index("--record") + 1]
        seed = command[command.index("--sampling-seed") + 1]
    except (ValueError, IndexError):
        return False
    return (
        Path(config).resolve() == REPO / "configs/wam_temporal_equal_task2_20261021.json"
        and record == "task2_init0"
        and seed == "42"
    )


def scan_old_workers():
    matches = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            command = [p.decode() for p in (folder / "cmdline").read_bytes().split(b"\0") if p]
            if old_worker_matches(command):
                fields = (folder / "stat").read_text().rpartition(")")[2].split()
                if fields[0] not in ("Z", "X"):
                    matches.append(dict(pid=int(folder.name), start_ticks=int(fields[19]), command=command))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return matches


def validate_zero_files(case, service, log):
    assert case.is_dir() and not list(case.iterdir()), "original task2 case acquired experiment data"
    assert sorted(p.name for p in service.iterdir()) == ["intervention.json", "runtime.json"]
    assert not list(service.glob("generate_*"))
    assert not re.findall(r'"POST [^ ]+ HTTP/1\.[01]"', log) and '"POST ' not in log
    return True


def zero_effect(value, old_launch, old_config, evidence):
    assert value["schema_version"] == "s2a.temporal-zero-step-original-failure.v1"
    assert value["scored_physics_steps"] == value["model_generations"] == 0
    assert value["original_failed_task_is_scientific_failure"] is False
    assert value["original_unregistered_worker_pid_unknown"] is True
    assert (
        value["worker_log_passed_python_env_guards"]
        and value["worker_failed_at_info_get_before_manifest_and_env_factory"]
    )
    assert not value["case_entries"] and not value["generate_post_lines"] and not value["matching_live_worker_commands"]
    for filename, digest in value["input_sha256"].items():
        evidence.check(filename, digest)
    cross = value["original_evidence"]["incident_crosscheck"]
    assert cross["sha256"] == CROSSCHECK_SHA
    evidence.check(cross["path"], CROSSCHECK_SHA)
    root = REPO / old_config["output_dir"]
    case = root / "simulation/8400_masked_temporal_equal/task2_init0_s42"
    service = root / "intervention_services/9846"
    assert (
        value["original_task2_root"] == str(root)
        and value["case_path"] == str(case)
        and value["service_path"] == str(service)
    )
    directory = root / "orchestration"
    validate_zero_files(case, service, evidence.read(directory / "service.log").decode())
    assert not (directory / "complete.json").exists()
    assert not (root / "simulation/temporal_matrix_8400_masked_temporal_equal.json").exists()
    error, release = evidence.json(directory / "error.json"), evidence.json(directory / "release.json")
    assert error == value["retained_error"] and release == value["retained_release"]
    assert error["scientific_failure_inferred"] is False and not error["automatic_retry"]
    assert release["last_job_returncode"] == 1 and release["model_returncode"] in (0, -15)
    assert not release["cleanup_errors"] and not release["automatic_retry"]
    assert release["only_owned_process_groups_stopped"]
    known = {"coordinator": old_launch["partitions"][2]["coordinator"]}
    for role in ("model", "matrix"):
        known[role] = evidence.json(directory / f"{role}_process.json")
    assert (
        release["model_pid"] == known["model"]["pid"] and release["model_start_ticks"] == known["model"]["start_ticks"]
    )
    assert release["last_job_pid"] == known["matrix"]["pid"]
    for role, key in (("model", "model"), ("matrix", "job")):
        assert release["stop_evidence"][key]["pid"] == known[role]["pid"]
    states = {
        role: process_record(p["pid"], "6" if role == "model" else "", start_ticks=p["start_ticks"])
        for role, p in known.items()
    }
    assert not any(p["present"] for p in states.values())
    matches = scan_old_workers()
    assert not matches, "an exact original unregistered worker command remains live"
    assert not list((root / "simulation/logs/8400_masked_temporal_equal").glob("task2_init0_s42.process.json"))
    return dict(
        startup_failure_retained=True,
        scored_task_failure=False,
        scored_physics_steps=0,
        model_generations=0,
        current_case_empty=True,
        current_service_post_and_generate_count=0,
        known_registered_processes=states,
        current_exact_old_worker_command_matches=matches,
        unregistered_original_worker_pid_unknown=True,
        original_exact_failure_cause_unknown=True,
        original_complete_record_absent=True,
    )


def original_contract(refs, configs, evidence):
    for key, expected in ORIGINAL_HASHES.items():
        assert refs[key]["sha256"] == expected
        evidence.check(refs[key]["path"], expected)
    launch = evidence.json(refs["launch"]["path"])
    assert (
        launch["schema_version"] == "s2a.temporal-equal-root-launch.v1"
        and launch["registered_budget"] == original.BUDGET
    )
    assert [p["partition"] for p in launch["partitions"]] == [0, 1, 2]
    ready = evidence.json(refs["partial_ready"]["path"])
    check_partial_ready(ready)
    assert ready["launch_path"] == str(Path(refs["launch"]["path"]).resolve())
    for name, digest in ready["input_sha256"].items():
        evidence.check(name, digest)
    gates = [
        original.check_gate(p["prerequisite_gate_path"], p["prerequisite_gate_sha256"], i, configs[i], evidence)
        for i, p in enumerate(launch["partitions"])
    ]
    zero = evidence.json(refs["zero_effect"]["path"])
    for key in ("launch", "partial_ready", "incident"):
        assert zero["original_evidence"][key] == refs[key]
    boundary = zero_effect(zero, launch, configs[2], evidence)
    return launch, ready, gates, boundary


def verify_scored_case(config, record, case, evidence):
    result = original.verify_case(
        config,
        record,
        config["variants"][0],
        case,
        evidence,
        source="wam",
        checkpoint="8400_masked",
        full=True,
        sampling_seed=42,
        first_query_validator=original.first_query,
    )
    assert result["steps"] == 600 and result["chunks"] == 75
    assert not result["raw_checkpoint_evidence"] and not result["component_ablation"]
    assert result["reference_intervention"] == "causal_temporal_equal_ensemble"
    assert result["independent_temporal_ensemble_replay"]["passed"]
    assert result["independent_temporal_ensemble_replay"]["verified_endpoints"] == 600
    chunks = evidence.json(case / "chunks.json")
    assert [c["step"] for c in chunks] == list(range(0, 600, 8))
    port = config["checkpoints"][0]["port"]
    provenance = original.service_artifacts(
        case, chunks, REPO / config["output_dir"] / "intervention_services" / str(port), evidence
    )
    rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
    result["opening_tracking"] = original.opening_tracking(rows)
    result["ensemble_reference_usage"] = original.ensemble_reference_usage(rows, chunks)
    result["latest_query_clock_diagnostics"] = result.pop("governor_diagnostics")
    result["latest_query_clock_diagnostics_scope"] = "Latest query cursor only, not ensemble suffix discard."
    result["first_query020_provenance"] = result.pop("first_query003_regression")
    result["success_duration_s"] = result["outcome"]["successful_steps"] / 20
    result["trailing_success_duration_s"] = result["outcome"]["trailing_success_steps"] / 20
    return result, provenance


def stable_identity(value):
    return {key: value[key] for key in ("pid", "start_ticks", "ppid", "command", "cwd", "environment")}


def handshake_contract(
    ready,
    registry,
    go,
    pending,
    *,
    config_path,
    config_sha,
    snapshot_path,
    snapshot_sha,
    registry_path,
    ready_sha,
    registry_sha,
    parent,
):
    """Independently authenticate saved admission records, without production helpers."""
    assert ready["schema_version"] == "s2a.cpu-worker-ready.v1" and go["schema_version"] == "s2a.cpu-worker-go.v1"
    identity = ready["identity"]
    assert identity == registry["identity"] == go["identity"] == stable_identity(registry)
    assert identity["pid"] == pending["pid"] and identity["start_ticks"] == pending["start_ticks"]
    assert identity["ppid"] == pending["ppid"] == parent["pid"]
    assert pending["command"] == identity["command"] and not pending["automatic_retry"]
    assert identity["cwd"] == str(REPO)
    expected_env = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    assert identity["environment"] == expected_env
    assert ready["parent_identity"] == go["parent_identity"] == parent
    assert ready["experiment_work_started"] is False and registry["automatic_retry"] is False
    assert isinstance(ready["token"], str) and re.fullmatch(r"[a-f0-9]{32}", ready["token"])
    assert go["token"] == ready["token"]
    assert Path(ready["config_path"]) == config_path and ready["config_sha256"] == config_sha
    assert Path(ready["snapshot_path"]) == snapshot_path
    assert ready["snapshot_sha256"] == registry["snapshot_sha256"] == snapshot_sha
    assert Path(go["registry_path"]) == registry_path and go["registry_sha256"] == registry_sha
    assert go["ready_sha256"] == registry["ready_sha256"] == ready_sha
    assert go["config_sha256"] == config_sha and go["snapshot_sha256"] == snapshot_sha
    assert Path(registry["ready_path"]) == registry_path.parent / "task2_init0_s42.handshake/ready.json"
    for observations in (ready["self_observations"], registry["parent_observations"]):
        assert isinstance(observations, list) and observations
        for observation in observations:
            assert {k: observation[k] for k in identity if k != "environment"} == {
                k: v for k, v in identity.items() if k != "environment"
            }
            assert set(observation["environment"]) == set(expected_env)
            assert all(observation["environment"][key] in (None, value) for key, value in expected_env.items())
        assert stable_identity(observations[-1]) == identity
    return dict(
        ready_go_registry_binding_passed=True,
        no_unknown_environment_normalized_to_empty=True,
        experiment_work_started_before_ready=False,
        token=ready["token"],
        identity=identity,
    )


def continuation_commands(config):
    model = [
        str(Path(config["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--checkpoint",
        "8400_masked",
        "--port",
        "9946",
        "--active-slot-mask",
    ]
    matrix = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_temporal_continuation_matrix.py",
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--source",
        "wam",
        "--variant",
        "temporal_equal",
        "--checkpoint",
        "8400_masked",
        "--workers",
        "1",
    ]
    return model, matrix


def gate_metadata(gate, launch, old_gate, review):
    """Check the separate one-trial launch contract, never replace the full budget."""
    assert gate["schema_version"] == "s2a.temporal-zero-step-continuation-gate.v1"
    assert gate["all_prerequisites_passed"] and gate["registered_budget"] == CONTINUATION_BUDGET
    assert (gate["partition"], gate["physical_gpu"], gate["port"]) == (2, 6, 9946)
    assert gate["config_path"] == launch["config_path"] == str(REPO / CONFIG)
    assert gate["config_sha256"] == launch["config_sha256"] == sha256(REPO / CONFIG)
    assert gate["original_evidence"] == launch["original_evidence"]
    assert gate["source_review"] == launch["source_review"] and review["review_passed"] is True
    assert gate["original020_gate"] == dict(
        path=old_gate["prerequisite_gate_path"], sha256=old_gate["prerequisite_gate_sha256"]
    )
    assert gate["model_started"] is False and gate["physics_started"] is False
    assert gate["new_model_requests"] == gate["new_physics_steps"] == 0 and gate["automatic_retry"] is False
    for name, digest in review["input_sha256"].items():
        assert gate["input_sha256"][name] == digest
    checkpoint = gate["actual_checkpoint_stat"]
    assert checkpoint == dict(
        files_stat_rechecked=163,
        original_full_fingerprint_sha256="624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8",
        weight_rehashed=False,
    )
    return True


def coordinator_command(launch):
    command = launch["command"]
    assert command[0] == str(REPO / ".venv/bin/python")
    assert Path(command[1]).resolve() == REPO / "scripts/run_wam_temporal_continuation_service.py"
    expected = {
        "--partition": "2",
        "--seed": "42",
        "--gate": str(Path(launch["prerequisite_gate_path"]).resolve()),
        "--gate-sha256": launch["prerequisite_gate_sha256"],
    }
    assert len(command) == 2 + 2 * len(expected)
    assert set(command[2::2]) == set(expected)
    assert all(command[command.index(flag) + 1] == value for flag, value in expected.items())
    return True


def continuation_resources(config, launch, ready, evidence):
    assert ready["schema_version"] == "s2a.temporal-zero-step-continuation-ready.v1"
    assert ready["concurrent_scope"] == "continuation partition2 only"
    assert ready["partition"] == launch["partition"] == 2
    assert ready["physical_gpu"] == launch["physical_gpu"] == 6 and ready["port"] == launch["port"] == 9946
    assert ready["ready"] and ready["actual_http_info_equals_saved"]
    assert ready["local_network_operations"] == "one GET /info request only"
    assert (
        ready["new_model_requests"] == ready["new_physics_steps"] == 0
        and not ready["nvml_to_local_pid_mapping_claimed"]
    )
    for path, digest in ready["input_sha256"].items():
        evidence.check(path, digest)
    entries = ready["local_registered_processes"]
    assert len(entries) == len({(p["pid"], p["start_ticks"]) for p in entries}) == 4
    assert {p["role"] for p in entries} == {"coordinator", "model", "matrix", "physics"}
    roles = {p["role"]: p for p in entries}
    root = REPO / config["output_dir"]
    directory = root / "orchestration"
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["schema_version"] == "s2a.temporal-zero-step-continuation-service.v1"
    assert manifest["partition"] == 2 and manifest["physical_gpu"] == 6 and manifest["config"] == config
    assert manifest["config_sha256"] == sha256(REPO / CONFIG)
    gate = evidence.json(launch["prerequisite_gate_path"])
    assert manifest["source_review"] == launch["source_review"]
    assert manifest["original_evidence"] == launch["original_evidence"]
    assert manifest["actual_checkpoint_stat"] == gate["actual_checkpoint_stat"]
    assert manifest["automatic_retry"] is False and manifest["external_sources_readonly"] is True
    assert roles["coordinator"]["pid"] == manifest["coordinator"]["pid"] == launch["coordinator"]["pid"]
    assert (
        roles["coordinator"]["start_ticks"]
        == manifest["coordinator"]["start_ticks"]
        == launch["coordinator"]["start_ticks"]
    )
    model_command, matrix_command = continuation_commands(config)
    assert manifest["model_command"] == roles["model"]["command"] == model_command
    assert manifest["job_command"] == roles["matrix"]["command"] == matrix_command
    assert roles["coordinator"]["command"] == launch["command"]
    coordinator_command(launch)
    for role, item in roles.items():
        assert type(item["pid"]) is int and type(item["start_ticks"]) is int and item["start_ticks"] > 0
        assert item["cwd"] == str(REPO) and item["proc_state"] not in ("Z", "X")
        assert item["environment"]["CUDA_VISIBLE_DEVICES"] == ("6" if role == "model" else "")
        assert item["environment"]["LD_LIBRARY_PATH"] == ""
        if role in ("model", "matrix"):
            assert item["ppid"] == roles["coordinator"]["pid"]
    physics = roles["physics"]
    assert physics["ppid"] == roles["matrix"]["pid"]
    assert physics["environment"] == dict(
        CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa"
    )
    for role in ("model", "matrix"):
        record = evidence.json(directory / f"{role}_process.json")
        assert record["pid"] == roles[role]["pid"] and record["start_ticks"] == roles[role]["start_ticks"]
        assert record["command"] == roles[role]["command"]
        assert record["cuda_visible_devices"] == ("6" if role == "model" else "") and record["ld_library_path"] == ""
    for path, digest in manifest["source_sha256"].items():
        evidence.check(path, digest)
    gate = str(Path(launch["prerequisite_gate_path"]).resolve())
    assert manifest["source_sha256"][gate] == launch["prerequisite_gate_sha256"]
    logs = root / "simulation/logs/8400_masked_temporal_equal"
    snapshot_path = logs / "execution_snapshot.json"
    snapshot = evidence.json(snapshot_path)
    assert snapshot["identity"] == ["wam", "temporal_equal", "8400_masked"] and snapshot["jobs"] == [
        ["task2_init0", 42]
    ]
    assert (
        snapshot["coordinator_pid"] == roles["matrix"]["pid"]
        and snapshot["cpu_workers"] == 1
        and not snapshot["model_started"]
    )
    assert snapshot["coordinator_identity"]["start_ticks"] == roles["matrix"]["start_ticks"]
    assert snapshot["config_sha256"] == sha256(REPO / CONFIG)
    for path, digest in snapshot["execution_sha256"].items():
        evidence.check(path, digest)
    registry_path = logs / "task2_init0_s42.process.json"
    registry = evidence.json(registry_path)
    assert stable_identity(registry) == stable_identity(physics)
    command = registry["command"]
    assert command[0] == str((REPO / config["simulation_python"]).absolute())
    assert Path(command[1]) == REPO / "scripts/run_wam_temporal_continuation_matrix.py"
    for flag, expected in (
        ("--config", str(REPO / CONFIG)),
        ("--seed", "42"),
        ("--source", "wam"),
        ("--variant", "temporal_equal"),
        ("--checkpoint", "8400_masked"),
        ("--record", "task2_init0"),
        ("--sampling-seed", "42"),
        ("--workers", "1"),
        ("--worker-snapshot", str(snapshot_path)),
    ):
        assert command[command.index(flag) + 1] == expected
    folder = logs / "task2_init0_s42.handshake"
    ready_path, go_path = folder / "ready.json", folder / "go.json"
    child_ready, child_go = evidence.json(ready_path), evidence.json(go_path)
    assert ready["child_ready_sha256"] == sha256(ready_path) and ready["child_go_sha256"] == sha256(go_path)
    assert command[command.index("--handshake-dir") + 1] == str(folder)
    assert command[command.index("--handshake-token") + 1] == child_ready["token"]
    admission = handshake_contract(
        child_ready,
        registry,
        child_go,
        evidence.json(folder / "pending.json"),
        config_path=REPO / CONFIG,
        config_sha=sha256(REPO / CONFIG),
        snapshot_path=snapshot_path,
        snapshot_sha=sha256(snapshot_path),
        registry_path=registry_path,
        ready_sha=sha256(ready_path),
        registry_sha=sha256(registry_path),
        parent={k: roles["matrix"][k] for k in ("pid", "start_ticks")},
    )
    info = evidence.json(directory / "server_info.json")
    assert info == ready["actual_info"]
    ck = config["checkpoints"][0]
    assert (
        Path(info["checkpoint_path"]).resolve()
        == (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
    )
    assert Path(info["config_file"]).resolve() == (Path(ck["run"]) / "config.yaml").resolve()
    assert info["port"] == 9946 and info["attention_backend"] == "sdpa" and not info["ema_weights"]
    assert Path(info["output_dir"]) == root / "intervention_services/9946"
    assert all(info[k] == 1 for k in ("world_size", "dp_shard_size", "cp_size", "cfgp_size"))
    assert not any(
        info[k]
        for k in (
            "compile",
            "fuse_qkv",
            "cuda_graphs",
            "reuse_small_action_und",
            "reuse_small_action_metadata",
            "fixed_training_buckets",
        )
    )
    intervention = info["acceptance_intervention"]
    assert (
        intervention["active_slot_mask"]
        and not intervention["exact_published_caption"]
        and intervention["warm_start_contract_run"] is None
    )
    for relative in ("scripts/serve_wam_acceptance.py", "src/s2a/wam_runtime.py"):
        evidence.check(relative, intervention["code"]["files_sha256"][relative])
    processes = {
        role: process_record(p["pid"], p["environment"]["CUDA_VISIBLE_DEVICES"], start_ticks=p["start_ticks"])
        for role, p in roles.items()
    }
    complete = original.optional_json(directory / "complete.json", evidence)
    release = original.optional_json(directory / "release.json", evidence)
    error = original.optional_json(directory / "error.json", evidence)
    matrix = original.optional_json(root / "simulation/temporal_matrix_8400_masked_temporal_equal.json", evidence)
    worker_release = original.optional_json(folder / "release.json", evidence)
    normal, http = False, None
    if complete is not None and release is not None and not any(p["present"] for p in processes.values()):
        assert error is None and not (folder / "error.json").exists()
        assert complete == dict(complete=True, partition=2, wam_trials=1, physics_steps=600, queries=75)
        original.check_release(release, roles)
        assert matrix is not None and matrix["complete"] and not matrix["model_started"]
        assert matrix["execution_snapshot_sha256"] == sha256(snapshot_path) and len(matrix["cases"]) == 1
        trial = matrix["cases"][0]
        assert trial["record"] == "task2_init0" and trial["sampling_seed"] == 42 and trial["returncode"] == 0
        assert stable_identity(trial) == stable_identity(physics)
        assert worker_release is not None and worker_release["reaped"] and worker_release["returncode"] == 0
        assert worker_release["pid"] == physics["pid"] and worker_release["start_ticks"] == physics["start_ticks"]
        assert worker_release["original_error"] is None and worker_release["only_owned_child_signalled"]
        http = original.http_budget(evidence.read(directory / "service.log").decode(), roles["model"]["pid"], 9946)
        normal = True
    return dict(
        partition=2,
        physical_gpu=6,
        port=9946,
        processes=processes,
        complete_record=complete,
        release=release,
        recorded_error=error,
        matrix_result=matrix,
        worker_release=worker_release,
        handshake=admission,
        final_http_budget=http,
        owned_service_released_normally=normal,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--ready-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    for commit, path in (
        ("5e2090c", "scripts/audit_wam_raw8_seed43.py"),
        ("89c38d3", "scripts/audit_wam_temporal_wam.py"),
        ("af147e1", "scripts/wam_temporal_position_checker.py"),
    ):
        digest = hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO)).hexdigest()
        evidence.check(REPO / path, digest)
    proof = original.source_proof(evidence)
    registered(PLAN, evidence)
    registered(CONFIG, evidence)
    old_configs = original.configurations(evidence)
    config = evidence.json(REPO / CONFIG)
    same_effect(old_configs[2], config)
    evidence.check(args.launch, args.launch_sha256)
    launch = evidence.json(args.launch)
    assert launch["schema_version"] == "s2a.temporal-zero-step-continuation-root-launch.v1"
    assert (launch["partition"], launch["physical_gpu"], launch["port"]) == (2, 6, 9946)
    assert Path(launch["config_path"]) == REPO / CONFIG and launch["config_sha256"] == sha256(REPO / CONFIG)
    old_launch, old_ready, gates, boundary = original_contract(launch["original_evidence"], old_configs, evidence)
    review = launch["source_review"]
    evidence.check(review["path"], review["sha256"])
    gate_path, gate_sha = launch["prerequisite_gate_path"], launch["prerequisite_gate_sha256"]
    evidence.check(gate_path, gate_sha)
    gate = evidence.json(gate_path)
    gate_metadata(gate, launch, old_launch["partitions"][2], evidence.json(review["path"]))
    assert gate["qualification"] == gates[2]["qualification"] and gate["prior_seed43"] == gates[2]["prior_seed43"]
    for name, digest in gate["input_sha256"].items():
        evidence.check(name, digest)
    for relative in (
        PLAN,
        CONFIG,
        "src/s2a/worker_handshake.py",
        "scripts/run_wam_temporal_continuation_matrix.py",
        "scripts/run_wam_temporal_continuation_service.py",
        "scripts/prepare_wam_temporal_continuation.py",
        "scripts/capture_wam_temporal_continuation_ready.py",
        "scripts/capture_wam_long_ready.py",
    ):
        assert gate["input_sha256"][str(REPO / relative)] == sha256(REPO / relative)
    for item in launch["original_evidence"].values():
        assert gate["input_sha256"][str(Path(item["path"]).resolve())] == item["sha256"]
    assert gate["input_sha256"][str(Path(review["path"]).resolve())] == review["sha256"]
    # Reusing the original physical callable is a byte-identity source contract.
    core = "src/s2a/temporal_ensemble_simulation.py"
    evidence.check(
        REPO / core, hashlib.sha256(subprocess.check_output(["git", "show", f"49778dd:{core}"], cwd=REPO)).hexdigest()
    )
    evidence.check(args.ready, args.ready_sha256)
    new_ready = evidence.json(args.ready)
    assert new_ready["input_sha256"][str(args.launch.resolve())] == args.launch_sha256
    resources_list = [
        original.resources(i, old_configs[i], old_launch["partitions"][i], old_ready["partitions"][i], evidence)
        for i in (0, 1)
    ]
    resources_list.append(continuation_resources(config, launch, new_ready, evidence))
    combined = [p for part in old_ready["partitions"] for p in part["local_registered_processes"]] + new_ready[
        "local_registered_processes"
    ]
    assert len(combined) == len({(p["pid"], p["start_ticks"]) for p in combined}) == 12
    configs = [old_configs[0], old_configs[1], config]
    stats_before = original.checkpoint_stats(config, evidence)
    fixture = evidence.json(REPO / config["fixture"])
    cases, errors = [], []
    for i, cfg in enumerate(configs):
        case = REPO / cfg["output_dir"] / "simulation/8400_masked_temporal_equal" / f"task{i}_init0_s42"
        if original.optional_json(case / "result.json", evidence) is None:
            cases.append(dict(partition=i, record=f"task{i}_init0", status="pending", result=None))
            continue
        try:
            record = next(r for r in fixture["records"] if r["name"] == f"task{i}_init0")
            result, provenance = verify_scored_case(cfg, record, case, evidence)
            cases.append(
                dict(
                    partition=i, record=record["name"], status="verified", result=result, service_provenance=provenance
                )
            )
        except Exception:
            errors.append(dict(partition=i, case=str(case), error=traceback.format_exc()))
            cases.append(dict(partition=i, record=f"task{i}_init0", status="error", result=None))
    verified = [c for c in cases if c["status"] == "verified"]
    all_cases = len(verified) == 3 and not errors
    released = all(r["owned_service_released_normally"] for r in resources_list)
    stats_after = original.checkpoint_stats(config, evidence)
    assert stats_before == stats_after
    evidence.recheck()
    result = dict(
        schema_version="s2a.temporal-equal-explicit-continuation-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=all_cases and released,
        all_cases_independently_verified=all_cases,
        normal_process_release=released,
        original_startup_failure=boundary,
        original_failed_trial_in_scored_denominator=False,
        registered_budget=original.BUDGET,
        actual_service_launches=4,
        scored_wam_trials=3,
        ready_evidence_scope="Historical original0/1 eight identities plus continuation2 four; never12 concurrent ready",
        verified_cases=len(verified),
        verified_physics_steps=sum(c["result"]["steps"] for c in verified),
        verified_queries=sum(c["result"]["chunks"] for c in verified),
        verified_http_queries=sum(
            r["final_http_budget"]["successful_http_requests"] for r in resources_list if r["final_http_budget"]
        ),
        raw_checkpoint_evidence=False,
        reference_intervention="causal_temporal_equal_ensemble",
        source_proof=proof,
        cases=cases,
        group_outcome=original.group_outcome(cases),
        resources=resources_list,
        errors=errors,
        prerequisite_gates=[
            dict(partition=i, qualification=g["qualification"], prior_seed43=g["prior_seed43"])
            for i, g in enumerate(gates)
        ],
        actual_checkpoint_stat=stats_after,
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_initial_states=0,
        limitations=[
            "021 adds one explicit zero-step startup continuation, not a fourth scored trial or an automatic retry; original error remains retained.",
            "The original unregistered worker PID was never recorded; only three known original failed identities and exact-command scan are asserted, never an invented fourth PID.",
            "Scored reference/control checking is the unchanged af147e1 independent checker;89 whole-ready guards remain unmodified and do not accept the partial snapshot.",
            "Causal ensembles change model references, so this is not a raw checkpoint score; admitted and latest raw/model/nominal tracking remain separate.",
            "Task/contact/opening limits and no-fresh scope are unchanged; neither quality completion nor smoothed references prove broader task ability.",
        ],
    )
    write_new(args.output, result)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                complete=result["complete"],
                verified_cases=len(verified),
                errors=len(errors),
            )
        )
    )
    if errors or (args.full and not result["complete"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
