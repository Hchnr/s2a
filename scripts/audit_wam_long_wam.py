"""Read-only017: fixed three1200-step WAM trials and exact old600 prefixes.

No model, simulator, render or HTTP request is made. The literal1200 mechanical
checker, episode-clock checker and saved-service provenance helper stay frozen.
Pending cases have no failure/outcome and no incomplete-group success rate.
"""

import argparse
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_earlier_position import verify_first_query006
from audit_wam_long_qualification import episode_check, exact_prefix, lines
from audit_wam_position import Evidence
from audit_wam_tracking_replan import service_artifacts
from run_wam_long_service import commands, preflight, weight_stats
from run_wam_short_replan import cpu_environment, process_record, write_new
from wam_long_position_checker import source_proof, verify_case

from s2a.artifacts import REPO, sha256
from s2a.task_acceptance import task_outcome

ROOT = REPO / "outputs/wam_stability_20261021"
SCHEMA = "s2a.long-raw-readonly-wam-audit.v1"
RUNTIME_COMMIT = "d6fb4a6765a5223a2108a9f9e1bbf50c34a9dff8"
COORDINATOR_COMMIT = "4c18fa01a8fc46f9204f707a9d76a3472a4c8f5b"
OLD = ROOT / "short_replan_audit/final12_verified_with_unused_checkpoint_exception.json"
OLD_SHA = "a4560bb70cc21e204dc7bd0e35fc3aa2a7cab7365b394a6a4c1727473b60ad12"
PINNED = {
    "scripts/run_wam_long_service.py": "339ae16978041f379313c3efd3b46f581745f68453745bfa3e3b5dfe5ce27bf5",
    "scripts/wam_long_position_checker.py": "e49bae82689c8ff6c3ecbc09219ce62f6a437b651e5cfd9ca4a2ef293c603710",
    "scripts/audit_wam_long_qualification.py": "d147b2f1edaab0cddef821751352c3d5d50904686f37ad821c738de1155facfc",
    "scripts/audit_wam_tracking_replan.py": "e5aee978fea6fe08685a3154ebe84ab3be3da627339c6ad69a657d766ab45ea9",
}
QUALIFICATION_ORIGINAL = ROOT / "long_raw_preflight/qualification_full6_verified.json"
QUALIFICATION_ORIGINAL_SHA = "c86569abeb1fb7e57bfd08468cc5e02cf598ef9c51f75a0da66b54e73e95357c"
QUALIFICATION_CAUSAL = ROOT / "long_raw_implementation/endpoint599_independent_causality.json"
QUALIFICATION_CAUSAL_SHA = "05ba5ae30c9ec826faf906b621868bb7e051b8f569077f05e0a0c0f58e25e6bb"


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


def old600(case, record, old_case, evidence):
    """Exact state/command plus every old75 saved input/raw/normalized output."""
    rows, previous_rows = lines(case / "steps.jsonl", evidence), lines(old_case / "steps.jsonl", evidence)
    prefix = exact_prefix(rows, previous_rows)
    assert prefix["all600_shared_rows_exact"], prefix
    assert json.dumps(rows[:600], sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        previous_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ), "old600 canonical JSON tokens differ, including numeric types or signed zero"
    chunks, previous_chunks = evidence.json(case / "chunks.json"), evidence.json(old_case / "chunks.json")
    assert [c["step"] for c in chunks] == list(range(0, 1200, 8))
    assert [c["step"] for c in previous_chunks] == list(range(0, 600, 8))
    old_manifest = evidence.json(old_case / "manifest.json")
    assert old_manifest["record"] == record and old_manifest["source"] == "wam"
    for index in range(75):
        assert evidence.read(case / f"input_{index:04d}.png") == evidence.read(old_case / f"input_{index:04d}.png")
        request, previous_request = (evidence.json(p / f"request_{index:04d}.json") for p in (case, old_case))
        for key in (
            "model_mode",
            "seed",
            "fps",
            "num_frames",
            "action_chunk_size",
            "guidance",
            "num_steps",
            "shift",
            "prompt",
            "extra",
        ):
            assert request[key] == previous_request[key], (index, key)
        content, previous_content = (
            evidence.json(p / f"response_{index:04d}.json")["outputs"][0]["content"] for p in (case, old_case)
        )
        np.testing.assert_array_equal(content["action"], previous_content["action"])
        np.testing.assert_array_equal(
            content["action128"]["normalized_action"], previous_content["action128"]["normalized_action"]
        )
        for key in (
            "anchor_tcp",
            "anchor_opening",
            "raw128",
            "effective128",
            "raw_targets_tcp",
            "effective_targets_tcp",
            "openings",
        ):
            np.testing.assert_array_equal(chunks[index][key], previous_chunks[index][key])
        for key in (
            "physics_before_sha256",
            "physics_after_sha256",
            "physics_before_time_s",
            "physics_after_time_s",
            "image_sha256",
            "execution_endpoint_count",
            "step",
        ):
            assert chunks[index][key] == previous_chunks[index][key]
    return dict(
        **prefix,
        all600_canonical_json_tokens_exact=True,
        old_case=str(old_case),
        all75_png_bytes_exact=True,
        all75_full40x128_raw_and_normalized_exact=True,
        all75_common_conditions_anchor_and_raw_targets_exact=True,
        old600_outcome=task_outcome([r["success"] for r in previous_rows], expected_steps=600, stable_steps=40),
        old600_results_not_relabelled=True,
    )


def qualification_contract(qualified, evidence):
    assert qualified["qualification_passed"] and qualified["all_cases_complete"] and qualified["normal_process_release"]
    assert (qualified["native_trials"], qualified["oracle_trials"], qualified["physical_steps"]) == (3, 3, 7200)
    assert qualified["schema_version"] in (
        "s2a.long-qualification-readonly.v1",
        "s2a.long-qualification-physical-core.v1",
    )
    if qualified["schema_version"] == "s2a.long-qualification-physical-core.v1":
        assert qualified["original_qualification_passed"] is False
        assert qualified["original_strict_result"] == dict(
            path=str(QUALIFICATION_ORIGINAL), sha256=QUALIFICATION_ORIGINAL_SHA, qualification_passed=False
        )
        assert qualified["independent_causality_review"] == dict(
            path=str(QUALIFICATION_CAUSAL), sha256=QUALIFICATION_CAUSAL_SHA
        )
        evidence.check(QUALIFICATION_ORIGINAL, QUALIFICATION_ORIGINAL_SHA)
        evidence.check(QUALIFICATION_CAUSAL, QUALIFICATION_CAUSAL_SHA)
        assert evidence.json(QUALIFICATION_ORIGINAL)["qualification_passed"] is False
        assert qualified["new_model_requests"] == qualified["new_physics_steps"] == 0
        assert qualified["oracle_excluded_from_wam_denominator"] and qualified["no_original_artifact_modified"]
        assert set(qualified["physical_core_cases"]) == {f"task{i}_init0" for i in range(3)}
        for case in qualified["physical_core_cases"].values():
            assert case["native_qualified"] and case["oracle_physical_core_qualified"]
            assert case["original_oracle_strict_qualified"] is False and case["oracle_outcome"]["stable"]
            assert case["oracle_admitted_tracking"]["position_rmse_m"] <= 0.01
            assert case["oracle_admitted_tracking"]["so3_rmse_rad"] <= 0.05
            assert case["original_full1200_mechanical_and_query_checks_unchanged"]
            assert case["all75_available_query_endpoints_exact"]
            prefix = case["first600"]
            assert prefix["actual_state_command_motor_and_all_other_fields_exact"]
            assert prefix["all600_shared_rows_exact"] is False and prefix["only_exception_step"] == 599
            assert prefix["existing_interface_tolerance"] == 1e-10 and prefix["new_scoring_tolerance_added"] is False
            expected = {"raw_desired_tcp", "effective_desired_tcp", "raw_model_tcp_error", "effective_model_tcp_error"}
            assert set(prefix["exception_fields"]) == expected
            assert len(prefix["logging_differences"]) == 4
            assert {d["field"] for d in prefix["logging_differences"]} == expected
            assert all(
                d["step"] == 599 and 0 < d["maximum_absolute_difference"] <= 1e-10
                for d in prefix["logging_differences"]
            )


def registration(group, evidence):
    assert group["schema_version"] == "s2a.long-raw-root-launch.v1"
    assert group["cpu_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
    assert group["external_references_read_only"] and not group["automatic_retry"]
    assert group["registered_budget"] == dict(
        wam_trials=3, physics_steps=3600, maximum_h40_queries=450, fresh_initial_states=0
    )
    assert [p["partition"] for p in group["partitions"]] == [0, 1, 2]
    configs, gates, launches, stats = {}, {}, {}, {}
    for entry in group["partitions"]:
        part = entry["partition"]
        evidence.check(entry["path"], entry["sha256"])
        launch = evidence.json(entry["path"])
        assert launch["schema"] == "s2a.long-raw-partition-root-launch.v1"
        assert launch["partition"] == part and (launch["physical_gpu"], launch["port"]) == (5 + part, 9655 + part)
        assert launch["budget_git_commit"] == "9fe490a" and launch["runtime_git_commit"] == RUNTIME_COMMIT
        assert launch["model_coordinator_commit"] == COORDINATOR_COMMIT
        assert launch["registered_budget"] == dict(
            wam_trials=1, physics_steps=1200, maximum_h40_queries=150, fresh_initial_states=0
        )
        assert launch["cpu_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        assert (
            launch["gpu_memory_mib_before"] == 0
            and launch["external_references_read_only"]
            and not launch["automatic_retry"]
        )
        command = launch["command"]
        assert Path(command[1]).resolve() == REPO / "scripts/run_wam_long_service.py"
        for key, value in (
            ("--partition", str(part)),
            ("--seed", "42"),
            ("--gate", launch["prerequisite_gate_path"]),
            ("--gate-sha256", launch["prerequisite_gate_sha256"]),
        ):
            assert command[command.index(key) + 1] == value
        assert "--dry-run" not in command
        c, root, records, ck, signatures, weights = preflight(
            launch["prerequisite_gate_path"], launch["prerequisite_gate_sha256"], part
        )
        for name, digest in signatures.items():
            evidence.check(name, digest)
        gate = evidence.json(launch["prerequisite_gate_path"])
        assert c["records"] == [f"task{part}_init0"] and c["sampling_seeds"] == [42]
        assert c["online_max_steps"] == 1200 and c["max_observation_interval_steps"] == 8
        assert c["episode_horizon"] == 1300 and c["ignore_done"] is False
        assert ck["gpu"] == 5 + part and ck["port"] == 9655 + part and len(records) == 1
        qualified = evidence.json(gate["qualification"]["path"])
        qualification_contract(qualified, evidence)
        configs[part], gates[part], launches[part], stats[part] = c, gate, launch, weights
    return configs, gates, launches, stats


def resource(part, config, launch, gate, evidence):
    directory = REPO / config["output_dir"] / "orchestration"
    coordinator = process_record(launch["coordinator_pid"], "", start_ticks=launch["coordinator_start_ticks"])
    manifest = optional_json(directory / "manifest.json", evidence)
    if manifest is None:
        return dict(
            partition=part,
            processes=dict(coordinator=coordinator),
            owned_service_released_normally=False,
            status="not_started",
        )
    assert manifest["schema_version"] == "s2a.long-raw-coordinator.v1"
    assert (manifest["partition"], manifest["physical_gpu"], manifest["port"]) == (part, 5 + part, 9655 + part)
    assert (manifest["wam_trials"], manifest["physics_steps"], manifest["maximum_h40_queries"]) == (1, 1200, 150)
    assert manifest["gpu_memory_mib_before"] == 0
    assert (manifest["coordinator"]["pid"], manifest["coordinator"]["start_ticks"]) == (
        launch["coordinator_pid"],
        launch["coordinator_start_ticks"],
    )
    assert (
        manifest["source_sha256"][str(Path(launch["prerequisite_gate_path"]).resolve())]
        == launch["prerequisite_gate_sha256"]
    )
    assert all(manifest["source_sha256"][name] == digest for name, digest in gate["input_sha256"].items())
    for name, digest in manifest["source_sha256"].items():
        evidence.check(name, digest)
    model_command, job_command = commands(config, config["checkpoints"][0], part)
    assert manifest["model_command"] == model_command and manifest["job_command"] == job_command
    processes = dict(coordinator=coordinator)
    records = dict(coordinator=dict(manifest["coordinator"], command=launch["command"]))
    for role, cuda, command in (("model", str(5 + part), model_command), ("matrix", "", job_command)):
        saved = optional_json(directory / f"{role}_process.json", evidence)
        if saved is not None:
            assert (
                saved["command"] == command and saved["cuda_visible_devices"] == cuda and saved["ld_library_path"] == ""
            )
            records[role] = saved
            processes[role] = process_record(saved["pid"], cuda, start_ticks=saved["start_ticks"])
    worker = optional_json(
        REPO / config["output_dir"] / f"simulation/logs/8400_masked_position_hold/task{part}_init0_s42.process.json",
        evidence,
    )
    if worker is not None:
        assert worker["ppid"] == records["matrix"]["pid"]
        assert worker["environment"]["CUDA_VISIBLE_DEVICES"] == worker["environment"]["LD_LIBRARY_PATH"] == ""
        assert worker["environment"]["MUJOCO_GL"] == "osmesa"
        processes["physics"] = process_record(worker["pid"], "", start_ticks=worker["start_ticks"])
    info = optional_json(directory / "server_info.json", evidence)
    if info is not None:
        ck = config["checkpoints"][0]
        assert Path(info["config_file"]).resolve() == (Path(ck["run"]) / "config.yaml").resolve()
        assert (
            Path(info["checkpoint_path"]).resolve()
            == (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
        )
        assert info["port"] == 9655 + part and info["attention_backend"] == "sdpa" and not info["ema_weights"]
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
    complete = optional_json(directory / "complete.json", evidence)
    release = optional_json(directory / "release.json", evidence)
    error = optional_json(directory / "error.json", evidence)
    released = False
    if complete is not None and release is not None:
        assert complete == dict(complete=True, wam_trials=1, physics_steps=1200, queries=150, partition=part)
        assert (
            release["model_pid"] == records["model"]["pid"]
            and release["model_start_ticks"] == records["model"]["start_ticks"]
        )
        assert release["last_job_pid"] == records["matrix"]["pid"] and release["last_job_returncode"] == 0
        assert release["model_returncode"] in (0, -15) and not release["cleanup_errors"]
        assert release["only_owned_process_groups_stopped"] and not release["automatic_retry"]
        assert evidence.json(directory / "matrix_exit.json") == dict(pid=records["matrix"]["pid"], returncode=0)
        assert set(processes) == {"coordinator", "model", "matrix", "physics"}
        released = not any(p["present"] for p in processes.values())
    return dict(
        partition=part,
        status="started",
        processes=processes,
        registered_processes=records,
        physics_registration=worker,
        info=info,
        complete_record=complete,
        release=release,
        recorded_execution_error=error,
        owned_service_released_normally=released and error is None,
    )


def ready_binding(path, digest, resources, evidence):
    evidence.check(path, digest)
    snapshot = evidence.json(path)
    assert snapshot["schema_version"] == "s2a.long-raw-live-ready-snapshot.v1" and snapshot["ready"]
    assert snapshot["new_model_requests"] == snapshot["new_physics_steps"] == 0
    assert snapshot["nvml_to_local_pid_mapping_claimed"] is False
    assert [p["partition"] for p in snapshot["partitions"]] == [0, 1, 2]
    owned = [p for part in snapshot["partitions"] for p in part["local_registered_processes"]]
    assert len(owned) == len({p["pid"] for p in owned}) == 12
    for name, value in snapshot["input_sha256"].items():
        evidence.check(name, value)
    for part, saved in enumerate(snapshot["partitions"]):
        assert (saved["physical_gpu"], saved["port"]) == (5 + part, 9655 + part)
        assert saved["ready"] and saved["source_identity_verified"] and saved["actual_http_info_equals_saved"]
        assert saved["actual_info"] == resources[part]["info"]
        identities = {p["role"]: p for p in saved["local_registered_processes"]}
        assert set(identities) == {"coordinator", "model", "matrix", "physics"}
        current = resources[part]
        registered = dict(current["registered_processes"], physics=current["physics_registration"])
        for role, identity in identities.items():
            record = registered[role]
            assert (identity["pid"], identity["start_ticks"]) == (record["pid"], record["start_ticks"])
            assert identity["cwd"] == str(REPO) and identity["proc_state"] not in ("Z", "X")
            assert identity["environment"]["CUDA_VISIBLE_DEVICES"] == (str(5 + part) if role == "model" else "")
            assert identity["environment"]["LD_LIBRARY_PATH"] == ""
            assert identity["command"] == record["command"]
        assert identities["model"]["ppid"] == identities["matrix"]["ppid"] == identities["coordinator"]["pid"]
        assert identities["physics"]["ppid"] == identities["matrix"]["pid"]
        assert (
            identities["physics"]["environment"]["MUJOCO_GL"]
            == identities["physics"]["environment"]["PYOPENGL_PLATFORM"]
            == "osmesa"
        )
    return dict(
        path=str(path.resolve()),
        sha256=digest,
        all12_historical_process_identities_verified=True,
        actual_get_info_equal_saved=True,
        nvml_pid_translation_claimed=False,
    )


def final_service_budget(part, config, case, evidence):
    root = REPO / config["output_dir"]
    service = root / "intervention_services" / str(9655 + part)
    expected = {Path(p).resolve() for p in case["service_provenance"]["generated_request_directories"]}
    assert len(expected) == 150 and expected == {p.resolve() for p in service.glob("generate_*") if p.is_dir()}
    directory = root / "orchestration"
    log = evidence.read(directory / "service.log").decode()
    posts = [
        (path, int(status))
        for method, path, status in re.findall(r'"([A-Z]+) ([^ ]+) HTTP/1\.[01]" (\d{3})', log)
        if method == "POST"
    ]
    assert len(posts) == len(re.findall(r'"POST ', log)) == 150 and all(p == ("/generate", 200) for p in posts)
    model = evidence.json(directory / "model_process.json")
    assert re.findall(r"Started server process \[(\d+)\]", log) == [str(model["pid"])]
    assert log.count(f"Uvicorn running on http://127.0.0.1:{9655 + part}") == 1
    return dict(
        partition=part,
        exact_requests=150,
        exact_generation_directories=150,
        no_extra_post=True,
        finished_server_log_present=f"Finished server process [{model['pid']}]" in log,
        internal_finally_receipt_inferred=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--ready-snapshot", type=Path)
    parser.add_argument("--ready-sha256")
    parser.add_argument("--reuse-verified", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert args.launch.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    assert bool(args.ready_snapshot) == bool(args.ready_sha256)
    evidence = Evidence()
    evidence.read(__file__)
    for name, digest in PINNED.items():
        evidence.check(name, digest)
    proof = source_proof(evidence)
    evidence.check(args.launch, args.launch_sha256)
    group = evidence.json(args.launch)
    configs, gates, launches, stats = registration(group, evidence)
    evidence.check(OLD, OLD_SHA)
    previous = evidence.json(OLD)
    assert previous["all12_cases_independently_verified"] and previous["verified_physics_steps"] == 7200
    assert previous["original_coordinator_final_gate_success"] is False
    for name, digest in previous["source_before_after_hashes"].items():
        evidence.check(name, digest)
    old_cases = {c["record"]: Path(c["case"]) for c in previous["cases"] if c["profile"] == "interval8_alpha1"}
    assert set(old_cases) == {f"task{i}_init0" for i in range(3)}
    cached = {}
    if args.reuse_verified:
        prior = evidence.json(args.reuse_verified)
        assert prior["schema_version"] == SCHEMA and prior["launch_sha256"] == args.launch_sha256
        assert prior["input_sha256"][str(Path(__file__).resolve())] == sha256(__file__)
        for name, digest in prior["input_sha256"].items():
            evidence.check(name, digest)
        cached = {c["partition"]: c for c in prior["cases"] if c["status"] == "verified"}
    evidence.check(configs[0]["fixture"], configs[0]["fixture_sha256"])
    fixture = {r["name"]: r for r in evidence.json(configs[0]["fixture"])["records"]}
    cases, observed = [], 0
    for part in (0, 1, 2):
        config = configs[part]
        name = f"task{part}_init0"
        case = REPO / config["output_dir"] / f"simulation/8400_masked_position_hold/{name}_s42"
        count = len(list(case.glob("request_*.json")))
        assert count <= 150
        observed += count
        identity = dict(partition=part, case=str(case), record=name, source="wam", checkpoint="8400_masked")
        result = optional_json(case / "result.json", evidence)
        if result is None:
            cases.append(dict(identity, status="partial" if case.exists() else "missing"))
            continue
        if result["error"] is not None:
            cases.append(dict(identity, status="execution_error", error=result["error"]))
            continue
        try:
            assert count == result["chunks"] == 150
            if part in cached:
                cases.append(dict(cached[part], reused_from=str(args.reuse_verified.resolve())))
                continue
            manifest = evidence.json(case / "manifest.json")
            assert manifest["long_episode_contract"] == dict(
                episode_horizon=1300, ignore_done=False, physical_steps=1200, initialization_policy_steps_added=0
            )
            actual = verify_case(
                config,
                fixture[name],
                config["variants"][0],
                case,
                evidence,
                source="wam",
                checkpoint="8400_masked",
                full=True,
                sampling_seed=42,
                first_query_validator=first_query,
            )
            rows = lines(case / "steps.jsonl", evidence)
            clock = episode_check(case, rows, evidence)
            prefix = old600(case, fixture[name], old_cases[name], evidence)
            chunks = evidence.json(case / "chunks.json")
            service = (REPO / config["output_dir"] / "intervention_services" / str(9655 + part)).resolve()
            actual.update(
                partition=part,
                episode_clock=clock,
                first600=prefix,
                service_provenance=service_artifacts(case, chunks, service, evidence),
                first600_outcome=task_outcome([r["success"] for r in rows[:600]], expected_steps=600, stable_steps=40),
                last600_outcome=task_outcome([r["success"] for r in rows[600:]], expected_steps=600, stable_steps=40),
            )
            matrix_path = REPO / config["output_dir"] / "simulation/long_matrix_8400_masked_position_hold.json"
            matrix = evidence.json(matrix_path)
            assert matrix["complete"] and len(matrix["cases"]) == 1
            only = matrix["cases"][0]
            assert (only["record"], only["sampling_seed"], only["returncode"]) == (name, 42, 0)
            logs = REPO / config["output_dir"] / "simulation/logs/8400_masked_position_hold"
            worker = evidence.json(logs / f"{name}_s42.process.json")
            assert (worker["pid"], worker["start_ticks"], worker["command"]) == (
                only["pid"],
                only["start_ticks"],
                only["command"],
            )
            evidence.check(logs / "execution_snapshot.json", matrix["execution_snapshot_sha256"])
            snapshot = evidence.json(logs / "execution_snapshot.json")
            assert snapshot["identity"] == ["wam", "position_hold", "8400_masked"] and snapshot["cpu_workers"] == 1
            assert snapshot["model_started"] is matrix["model_started"] is False
            assert snapshot["jobs"] == [[name, 42]] and snapshot["config_sha256"] == sha256(
                REPO / f"configs/wam_long_raw_task{part}_20261021.json"
            )
            matrix_process = evidence.json(REPO / config["output_dir"] / "orchestration/matrix_process.json")
            coordinator_identity = snapshot["coordinator_identity"]
            assert snapshot["coordinator_pid"] == coordinator_identity["pid"] == matrix_process["pid"]
            assert coordinator_identity["start_ticks"] == matrix_process["start_ticks"]
            assert coordinator_identity["ppid"] == launches[part]["coordinator_pid"]
            assert worker["ppid"] == matrix_process["pid"]
            for record in (coordinator_identity, worker):
                assert record["environment"]["CUDA_VISIBLE_DEVICES"] == record["environment"]["LD_LIBRARY_PATH"] == ""
            expected_worker = [
                str(Path(config["simulation_python"]).absolute()),
                str(REPO / "scripts/run_wam_long_matrix.py"),
                "--config",
                str(REPO / f"configs/wam_long_raw_task{part}_20261021.json"),
                "--seed",
                "42",
                "--source",
                "wam",
                "--variant",
                "position_hold",
                "--record",
                name,
                "--sampling-seed",
                "42",
                "--worker-snapshot",
                str(logs / "execution_snapshot.json"),
                "--checkpoint",
                "8400_masked",
            ]
            assert worker["command"] == expected_worker
            for filename, digest in snapshot["execution_sha256"].items():
                evidence.check(filename, digest)
                assert manifest["execution_sha256"][filename] == digest
            cases.append(actual)
        except Exception as exc:
            cases.append(dict(identity, status="audit_error", error=repr(exc), traceback=traceback.format_exc()))
    resources = [resource(i, configs[i], launches[i], gates[i], evidence) for i in (0, 1, 2)]
    verified = [c for c in cases if c["status"] == "verified"]
    all_cases = len(verified) == 3
    released = all(r["owned_service_released_normally"] for r in resources)
    ready = ready_binding(args.ready_snapshot, args.ready_sha256, resources, evidence) if args.ready_snapshot else None
    if all_cases and released:
        assert ready is not None, "final needs source-bound historical live identity snapshot"
        budget = [
            final_service_budget(i, configs[i], next(c for c in verified if c["partition"] == i), evidence)
            for i in (0, 1, 2)
        ]
        assert sum(c["exact_requests"] for c in budget) == observed == 450
    else:
        budget = None
    weight_stats(configs[0]["checkpoints"][0])
    evidence.recheck()
    value = dict(
        schema_version=SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=all_cases and released and ready is not None and budget is not None,
        all_cases_independently_verified=all_cases,
        verified_cases=len(verified),
        registered_budget=dict(wam_trials=3, physics_steps=3600, maximum_h40_queries=450),
        observed_request_files=observed,
        cases=cases,
        resources=resources,
        ready_binding=ready,
        final_http_and_directory_budget=budget,
        group=dict(
            complete=all_cases,
            expected_cases=3,
            verified_cases=len(verified),
            stable_successes_among_verified=sum(c["outcome"]["stable"] for c in verified),
            ever_successes_among_verified=sum(c["outcome"]["ever"] for c in verified),
            stable_rate=sum(c["outcome"]["stable"] for c in verified) / 3 if all_cases else None,
        ),
        source_proof=proof,
        launch_path=str(args.launch.resolve()),
        launch_sha256=args.launch_sha256,
        original010_case_reference=dict(path=str(OLD), sha256=OLD_SHA, original_coordinator_final_gate_success=False),
        qualification_references={str(i): dict(gates[i]["qualification"]) for i in (0, 1, 2)},
        actual8400_weight_stat_checks=stats,
        reused_verified_cases=sum("reused_from" in c for c in cases),
        new_model_requests=0,
        new_physics_steps=0,
        input_sha256=evidence.hashes,
        limitations=[
            "Only three old-scene raw8/s42 trials; no fresh initial-state or unseen-training claim.",
            "All full1200 mechanics and episode clocks must pass alongside exact old600 rows and75 full model/image queries.",
            "The old600 and new1200 native outcomes are reported separately; no retroactive success relabeling.",
            "Pending cases are not failures and incomplete-group rates are null.",
            "Oracle/native qualifications and prior experiments are referenced, not rerun or counted as WAM successes.",
            "Final requires all450 POSTs/generation directories, three normal external releases and historical live source/CVD bindings.",
            "No NVML/local PID translation or absent internal finally receipt is invented.",
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
                observed_request_files=observed,
            )
        )
    )


if __name__ == "__main__":
    main()
