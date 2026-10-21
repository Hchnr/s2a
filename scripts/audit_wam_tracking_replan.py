"""Read-only016 WAM audit: three old scenes, causal timing and full mechanics.

The oracle qualification is referenced without rerunning/counting it. Pending
trials receive no outcome or failure; the rate is populated only for all three
verified trials and complete requires the owned service's normal release.
"""

import argparse
import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_earlier_position import verify_first_query006
from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, process_record, write_new
from wam_tracking_replan_checker import source_proof, verify_case

from s2a.artifacts import REPO, sha256

CONFIG = "configs/tracking_replan_wam_20261021.json"
QUALIFICATION_SHA = "353d7669f8060f99b6099f25b458394d04457cbf77b2845fd562ff2d58cf1a49"
FINGERPRINT = "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
FINGERPRINT_SHA = "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8"
SCHEMA = "s2a.tracking-replan-readonly-wam-audit.v1"


def optional_json(path, evidence):
    if not path.exists():
        return None
    try:
        json.loads(path.read_bytes())
    except json.JSONDecodeError:
        return None
    return evidence.json(path)


def first_query(config, record, variant, case, chunk, evidence, checkpoint, *, sampling_seed):
    assert type(sampling_seed) is int and sampling_seed == 42
    return verify_first_query006(config, record, variant, case, chunk, evidence, checkpoint)


def preflight(launch_path, launch_sha, evidence):
    evidence.check(launch_path, launch_sha)
    launch = evidence.json(launch_path)
    assert launch["budget_git_commit"] == "9492840" and launch["runtime_git_commit"] == "68fa982"
    assert launch["model_coordinator_commit"] == "3a49a5f"
    assert launch["registered_budget"] == dict(
        wam_trials=3, physics_steps=1800, maximum_h40_queries=225, fresh_initial_states=0
    )
    assert launch["cpu_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
    assert launch["physical_gpu"] == 7 and launch["port"] == 9597 and launch["gpu_memory_mib_before"] == 0
    assert launch["external_references_read_only"] and not launch["automatic_retry"]
    command = launch["command"]
    assert Path(command[1]).resolve() == REPO / "scripts/run_wam_tracking_service.py"
    assert command[command.index("--seed") + 1] == "42"
    gate_path, gate_sha = command[command.index("--gate") + 1], command[command.index("--gate-sha256") + 1]
    assert gate_path == launch["prerequisite_gate_path"] and gate_sha == launch["prerequisite_gate_sha256"]
    evidence.check(gate_path, gate_sha)
    gate = evidence.json(gate_path)
    assert gate["schema_version"] == "s2a.tracking-replan-launch-gate.v1" and gate["all_prerequisites_passed"]
    for filename, digest in gate["input_sha256"].items():
        evidence.check(filename, digest)
    config = evidence.json(CONFIG)
    assert gate["input_sha256"][str(REPO / CONFIG)] == sha256(CONFIG)
    assert hashlib.sha256(subprocess.check_output(["git", "show", f"9492840:{CONFIG}"])).hexdigest() == sha256(CONFIG)
    assert config["seed"] == 42 and config["sampling_seeds"] == [42]
    assert config["records"] == ["task0_init0", "task1_init0", "task2_init0"]
    assert len(config["variants"]) == len(config["checkpoints"]) == 1
    assert (
        config["online_max_steps"],
        config["online_chunk_size"],
        config["control_fps"],
        config["stable_success_last_steps"],
    ) == (600, 40, 20, 40)
    checkpoint = config["checkpoints"][0]
    assert (checkpoint["name"], checkpoint["iteration"], checkpoint["gpu"], checkpoint["port"]) == (
        "8400_masked",
        "iter_000008400",
        7,
        9597,
    )
    assert (config["sampling_steps"], config["guidance"], config["shift"]) == (35, 1, 5)
    qualification = gate["qualification"]
    assert qualification["sha256"] == QUALIFICATION_SHA
    evidence.check(qualification["path"], qualification["sha256"])
    qualified = evidence.json(qualification["path"])
    assert qualified["qualification_passed"] and qualified["all_cases_complete"]
    assert (qualified["oracle_trials"], qualified["physical_steps"]) == (3, 1800)
    assert qualified["normal_process_release"]["all_registered_processes_absent"]
    for filename, digest in qualified["input_sha256"].items():
        assert gate["input_sha256"][filename] == digest
        evidence.check(filename, digest)
    backend = gate["backend_release"]
    evidence.check(backend["path"], backend["sha256"])
    previous = evidence.json(backend["path"])
    assert previous["quality_complete"] and previous["verified_queries"] == 48
    assert previous["owned_service_released_normally"] and not previous["errors"]
    assert previous["release"]["model_returncode"] == -15 and previous["release"]["job_returncode"] == 0
    assert not previous["release"]["cleanup_errors"]
    for filename, digest in previous["input_sha256"].items():
        assert gate["input_sha256"][filename] == digest
        evidence.check(filename, digest)
    for owned in backend["owned_processes"]:
        assert not process_record(owned["pid"], owned["cuda_visible_devices"], start_ticks=owned["start_ticks"])[
            "present"
        ]
    evidence.check(FINGERPRINT, FINGERPRINT_SHA)
    weight = next(c for c in evidence.json(FINGERPRINT)["checkpoints"] if c["name"] == "8400_masked")
    assert len(weight["files"]) == 163
    assert (
        Path(weight["checkpoint"]).resolve()
        == (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    )
    for filename, item in weight["files"].items():
        stat = Path(filename).stat()
        assert stat.st_size == item["bytes"] and stat.st_mtime_ns == item["mtime_ns"] and len(item["sha256"]) == 64
    return config, gate, gate_path, gate_sha, launch


def service_artifacts(case, chunks, service_root, evidence):
    generated = []
    for index in range(len(chunks)):
        response = evidence.json(case / f"response_{index:04d}.json")
        directory = Path(response["args"]["output_dir"]).resolve()
        assert directory.parent == service_root and directory.name.startswith("generate_")
        assert evidence.json(directory / "sample_args.json") == response["args"]
        assert evidence.json(directory / "sample_outputs.json") == response
        generated.append(str(directory))
    assert len(set(generated)) == len(chunks)
    return dict(all_saved_service_args_and_responses_exact=True, generated_request_directories=generated)


def resources(config, gate, gate_path, gate_sha, launch, evidence):
    root = REPO / config["output_dir"]
    directory = root / "orchestration"
    coordinator = process_record(launch["coordinator_pid"], "", start_ticks=launch["coordinator_start_ticks"])
    manifest = optional_json(directory / "manifest.json", evidence)
    if manifest is None:
        return dict(coordinator=coordinator, owned_service_released_normally=False, status="not_started")
    assert (
        manifest["physical_gpu"],
        manifest["port"],
        manifest["wam_trials"],
        manifest["physics_steps"],
        manifest["maximum_h40_queries"],
    ) == (7, 9597, 3, 1800, 225)
    assert manifest["coordinator"]["pid"] == launch["coordinator_pid"]
    assert manifest["coordinator"]["start_ticks"] == launch["coordinator_start_ticks"]
    assert manifest["source_sha256"][str(Path(gate_path).resolve())] == gate_sha
    for filename, digest in manifest["source_sha256"].items():
        evidence.check(filename, digest)
    assert all(manifest["source_sha256"][p] == digest for p, digest in gate["input_sha256"].items())
    assert manifest["gpu_memory_mib_before"] == 0
    expected_model = [
        str(Path(config["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--checkpoint",
        "8400_masked",
        "--port",
        "9597",
        "--active-slot-mask",
    ]
    expected_job = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_tracking_matrix.py",
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--source",
        "wam",
        "--variant",
        "position_hold",
        "--checkpoint",
        "8400_masked",
    ]
    assert manifest["model_command"] == expected_model and manifest["job_command"] == expected_job
    processes = dict(coordinator=coordinator)
    owned = {}
    for role, cuda in (("model", "7"), ("matrix", "")):
        record = optional_json(directory / f"{role}_process.json", evidence)
        if record is not None:
            assert record["cuda_visible_devices"] == cuda and record["ld_library_path"] == ""
            assert record["command"] == manifest["model_command" if role == "model" else "job_command"]
            owned[role] = record
            processes[role] = process_record(record["pid"], cuda, start_ticks=record["start_ticks"])
    processes["physics"] = []
    for name in config["records"]:
        worker = optional_json(
            root / "simulation/logs/8400_masked_position_hold" / f"{name}_s42.process.json", evidence
        )
        if worker is not None:
            assert worker["command"][worker["command"].index("--source") + 1] == "wam"
            assert worker["command"][worker["command"].index("--sampling-seed") + 1] == "42"
            processes["physics"].append(process_record(worker["pid"], ""))
    info = optional_json(directory / "server_info.json", evidence)
    if info is not None:
        ck = config["checkpoints"][0]
        assert (
            Path(info["checkpoint_path"]).resolve()
            == (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
        )
        assert Path(info["config_file"]).resolve() == (Path(ck["run"]) / "config.yaml").resolve()
        assert info["port"] == 9597 and info["attention_backend"] == "sdpa" and not info["ema_weights"]
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
        assert intervention["active_slot_mask"] and not intervention["exact_published_caption"]
        assert intervention["warm_start_contract_run"] is None
        for relative in ("scripts/serve_wam_acceptance.py", "src/s2a/wam_runtime.py"):
            evidence.check(relative, intervention["code"]["files_sha256"][relative])
    complete = optional_json(directory / "complete.json", evidence)
    release = optional_json(directory / "release.json", evidence)
    error = optional_json(directory / "error.json", evidence)
    matrix_exit = optional_json(directory / "matrix_exit.json", evidence)
    released = False
    if complete is not None and release is not None:
        assert complete["complete"] and (complete["wam_trials"], complete["physics_steps"]) == (3, 1800)
        assert 45 <= complete["queries"] <= 225
        assert release["model_pid"] == owned["model"]["pid"]
        assert release["model_start_ticks"] == owned["model"]["start_ticks"]
        assert release["last_job_pid"] == owned["matrix"]["pid"]
        assert release["model_returncode"] in (0, -15) and release["last_job_returncode"] == 0
        assert (
            release["only_owned_process_groups_stopped"]
            and not release["cleanup_errors"]
            and not release["automatic_retry"]
        )
        assert matrix_exit == dict(pid=owned["matrix"]["pid"], returncode=0)
        assert len(processes["physics"]) == 3
        released = not any(processes[k]["present"] for k in ("coordinator", "model", "matrix")) and not any(
            p["present"] for p in processes["physics"]
        )
    return dict(
        status="started",
        processes=processes,
        service_info=info,
        recorded_execution_error=error,
        complete_record=complete,
        release=release,
        owned_service_released_normally=released and error is None,
        live_model_environment_authenticated=processes.get("model", {}).get("present", False),
        registered_model_cuda_visible_devices=owned.get("model", {}).get("cuda_visible_devices"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-verified", type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert args.launch.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    proof = source_proof(evidence)
    config, gate, gate_path, gate_sha, launch = preflight(args.launch, args.launch_sha256, evidence)
    reusable = {}
    if args.reuse_verified:
        previous = evidence.json(args.reuse_verified)
        assert previous["schema_version"] == SCHEMA and previous["launch_sha256"] == args.launch_sha256
        assert previous["input_sha256"][str(Path(__file__).resolve())] == sha256(__file__)
        for path, digest in previous["input_sha256"].items():
            evidence.check(path, digest)
        reusable = {str(Path(c["case"]).resolve()): c for c in previous["cases"] if c["status"] == "verified"}
    evidence.check(config["fixture"], config["fixture_sha256"])
    records = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
    root = REPO / config["output_dir"]
    service_root = (root / "intervention_services/9597").resolve()
    cases, requests = [], 0
    for name in config["records"]:
        case = root / "simulation/8400_masked_position_hold" / f"{name}_s42"
        count = len(list(case.glob("request_*.json")))
        assert count <= 75
        requests += count
        identity = dict(case=str(case), record=name, source="wam", checkpoint="8400_masked", sampling_seed=42)
        result = optional_json(case / "result.json", evidence)
        if result is None:
            cases.append(dict(identity, status="partial" if case.exists() else "missing"))
            continue
        if result["error"] is not None:
            cases.append(dict(identity, status="execution_error", error=result["error"]))
            continue
        try:
            assert 15 <= count == result["chunks"] <= 75
            cached = reusable.get(str(case.resolve()))
            if cached is not None:
                cases.append(dict(cached, reused_from=str(args.reuse_verified.resolve())))
                continue
            actual = verify_case(
                config,
                records[name],
                config["variants"][0],
                case,
                evidence,
                source="wam",
                checkpoint="8400_masked",
                full=True,
                sampling_seed=42,
                first_query_validator=first_query,
            )
            chunks = evidence.json(case / "chunks.json")
            assert len(chunks) == count and 15 <= len(chunks) <= 75
            assert all(8 <= b["step"] - a["step"] <= 40 for a, b in zip(chunks, chunks[1:]))
            actual["service_provenance"] = service_artifacts(case, chunks, service_root, evidence)
            cases.append(actual)
        except Exception as exc:
            cases.append(dict(identity, status="audit_error", error=repr(exc), traceback=traceback.format_exc()))
    assert len(cases) == 3 and requests <= 225
    resource = resources(config, gate, gate_path, gate_sha, launch, evidence)
    verified = [c for c in cases if c["status"] == "verified"]
    all_complete = len(verified) == 3
    if all_complete and resource["complete_record"] is not None:
        assert resource["complete_record"]["queries"] == requests == sum(c["chunks"] for c in verified)
        matrix = evidence.json(root / "simulation/tracking_matrix_8400_masked_position_hold.json")
        assert matrix["complete"] and len(matrix["cases"]) == 3 and not matrix["model_started"]
        assert all(c["returncode"] == 0 for c in matrix["cases"])
        assert sorted((c["record"], c["sampling_seed"]) for c in matrix["cases"]) == sorted(
            (name, 42) for name in config["records"]
        )
        snapshot_path = root / "simulation/logs/8400_masked_position_hold/execution_snapshot.json"
        evidence.check(snapshot_path, matrix["execution_snapshot_sha256"])
        snapshot = evidence.json(snapshot_path)
        assert snapshot["identity"] == ["wam", "position_hold", "8400_masked"]
        assert snapshot["config_sha256"] == evidence.hashes[str(REPO / CONFIG)]
        assert snapshot["jobs"] == [[name, 42] for name in config["records"]] and snapshot["cpu_workers"] == 1
    evidence.recheck()
    value = dict(
        schema_version=SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="only016_three_raw_old_scene_tracking_replan_development_trials",
        complete=all_complete and resource["owned_service_released_normally"],
        all_cases_independently_verified=all_complete,
        verified_cases=len(verified),
        registered_budget=dict(wam_trials=3, physics_steps=1800, maximum_h40_queries=225),
        observed_request_files=requests,
        cases=cases,
        resources=resource,
        group=dict(
            complete=all_complete,
            verified_cases=len(verified),
            expected_cases=3,
            stable_successes_among_verified=sum(c["outcome"]["stable"] for c in verified),
            ever_successes_among_verified=sum(c["outcome"]["ever"] for c in verified),
            stable_rate=sum(c["outcome"]["stable"] for c in verified) / 3 if all_complete else None,
        ),
        launch_path=str(args.launch.resolve()),
        launch_sha256=args.launch_sha256,
        source_proof=proof,
        reused_verified_cases=sum("reused_from" in c for c in cases),
        prior_oracle_referenced_not_executed=dict(
            sha256=QUALIFICATION_SHA, oracle_trials=3, oracle_excluded_from_wam_denominator=True
        ),
        original163_weight_metadata_rechecked=True,
        new_model_or_physics_by_auditor=False,
        input_sha256=evidence.hashes,
        limitations=[
            "All trials use three old scenes/init0/s42 and one frozen raw-reference policy.",
            "Pending cases are not scientific failures; partial rates are null.",
            "The same frozen seed-aware mechanics and independent causal timing replay verify every endpoint.",
            "Existing oracle qualifications and013 backend state diagnostics are prerequisites, never WAM task successes.",
            "Model SIGTERM is authenticated by the external owned release/PID evidence; no absent internal finally receipt is invented.",
            "There is no fresh performance, budget extension, per-task selection or automatic retry.",
        ],
    )
    write_new(args.output, value)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                complete=value["complete"],
                verified_cases=len(verified),
                observed_request_files=requests,
            )
        )
    )


if __name__ == "__main__":
    main()
