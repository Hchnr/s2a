"""Read-only019: three fixed raw8/seed43 trials, never a seed42 output regression.

The frozen014 literal checker performs every600-endpoint mechanical/query
reconstruction. This wrapper authenticates019 registration, actual live-ready
identities and complete lifecycle/budget evidence. It starts no child, HTTP
request, environment or model, and never rewrites an execution artifact.
"""

import argparse
import hashlib
import json
import re
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, process_record, write_new
from wam_position_acceptance_checker import source_proof, verify_case, verify_query

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
REGISTRATION = "8401aa3"
PLAN = "plans/v0.0.1/019_raw8_seed43_replication_20261021.md"
BUDGET = dict(wam_trials=3, physics_steps=1800, h40_queries=225, new_oracle_trials=0, fresh_initial_states=0)
OLD_AUDIT = ROOT / "short_replan_audit/final12_verified_with_unused_checkpoint_exception.json"
OLD_AUDIT_SHA = "a4560bb70cc21e204dc7bd0e35fc3aa2a7cab7365b394a6a4c1727473b60ad12"
PREREQUISITES = {
    "seed43_qualification": "9cd4cf46260eda2d87e13bd992c5ce871469d4ff2d3f38734129e2a1bd5fbba3",
    "long_raw_final": "b3910cdeb6669e5f6f74a654d44f2a214dc3b12249d33391e085efbb6a375168",
}
FIRST_CONDITIONS = (
    "model_mode",
    "fps",
    "num_frames",
    "action_chunk_size",
    "guidance",
    "num_steps",
    "shift",
    "prompt",
    "extra",
)


def optional_json(path, evidence):
    """Incomplete active JSON is pending and is not entered in the SHA ledger."""
    if not path.exists():
        return None
    try:
        json.loads(path.read_bytes())
    except json.JSONDecodeError:
        return None
    return evidence.json(path)


def registered(relative, evidence):
    expected = hashlib.sha256(
        subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
    ).hexdigest()
    evidence.check(REPO / relative, expected)
    return expected


def configurations(evidence):
    registered(PLAN, evidence)
    base = evidence.json(REPO / "configs/wam_short_replan_8_alpha1_20261021.json")
    configs = []
    for part in range(3):
        relative = f"configs/wam_raw8_seed43_task{part}_20261021.json"
        registered(relative, evidence)
        config = evidence.json(REPO / relative)
        expected = json.loads(json.dumps(base))
        expected.update(
            output_dir=f"outputs/wam_stability_20261021/raw8_seed43/task{part}",
            records=[f"task{part}_init0"],
            sampling_seeds=[43],
            notes=config["notes"],
        )
        expected["checkpoints"] = [dict(base["checkpoints"][0], gpu=4 + part, port=9744 + part)]
        assert config == expected, "019 differs from frozen raw8 outside the registered seed/resources/record"
        assert (config["seed"], config["online_max_steps"], config["max_observation_interval_steps"]) == (42, 600, 8)
        assert config["sampling_seeds"] == [43]
        evidence.check(REPO / config["fixture"], config["fixture_sha256"])
        configs.append(config)
    return configs


def compare_first_inputs(old, current):
    """Only physical initial inputs/conditions are paired; generated outputs may differ."""
    assert type(old["request"]["seed"]) is int and old["request"]["seed"] == 42
    assert type(current["request"]["seed"]) is int and current["request"]["seed"] == 43
    for key in FIRST_CONDITIONS:
        assert old["request"][key] == current["request"][key], key
    assert old["record"] == current["record"]
    assert old["controller"] == current["controller"]
    assert old["png"] == current["png"]
    for key in ("anchor_tcp", "anchor_opening"):
        np.testing.assert_array_equal(old["chunk"][key], current["chunk"][key])
    for value in (old, current):
        chunk = value["chunk"]
        assert chunk["step"] == 0 and chunk["physics_unchanged"]
        assert chunk["physics_before_sha256"] == chunk["physics_after_sha256"]
        assert chunk["physics_before_sha256"] == value["controller"]["actual_initial_physics_sha256"]
        assert chunk["physics_before_time_s"] == chunk["physics_after_time_s"]
        for key in (
            "measured_source",
            "measured_twist",
            "opening_before",
            "gripper_qpos_before",
            "motor_target_before",
        ):
            np.testing.assert_array_equal(old["row"][key], current["row"][key])
    assert old["chunk"]["physics_before_sha256"] == current["chunk"]["physics_before_sha256"]
    assert old["chunk"]["physics_before_time_s"] == current["chunk"]["physics_before_time_s"]
    return dict(
        image_png_bytes_exact=True,
        fixed_initial_anchor_exact=True,
        initial_physics_sha_exact=True,
        controller_and_motor_initialization_exact=True,
        initial_measured_state_exact=True,
        conditions_except_sampling_seed_exact=True,
        previous_sampling_seed=42,
        actual_sampling_seed=43,
        seed43_generated_output_authenticated_separately=True,
        seed43_raw_equals_seed42_required=False,
        scope="first observation only; subsequent closed-loop images/anchors may diverge; no seed42 output substitution",
    )


def first_query(config, record, variant, case, chunk, evidence, checkpoint, *, sampling_seed):
    assert sampling_seed == 43 and config["sampling_seeds"] == [43]
    verify_query(config, case, 0, chunk, evidence, checkpoint, sampling_seed=43)
    evidence.check(OLD_AUDIT, OLD_AUDIT_SHA)
    previous = evidence.json(OLD_AUDIT)
    old = next(c for c in previous["cases"] if c["profile"] == "interval8_alpha1" and c["record"] == record["name"])
    old_case = (REPO / old["case"]).resolve()
    for filename in (
        "manifest.json",
        "controller.json",
        "chunks.json",
        "request_0000.json",
        "input_0000.png",
        "steps.jsonl",
    ):
        path = old_case / filename
        evidence.check(path, previous["source_before_after_hashes"][str(path)])
    old_manifest = evidence.json(old_case / "manifest.json")
    assert old_manifest["source"] == "wam" and old_manifest["experiment"]["sampling_seeds"] == [42]
    assert old_manifest["variant"] == variant

    def inputs(folder, manifest, first):
        return dict(
            request=evidence.json(folder / "request_0000.json"),
            record=manifest["record"],
            controller=evidence.json(folder / "controller.json"),
            png=evidence.read(folder / "input_0000.png"),
            chunk=first,
            row=json.loads(evidence.read(folder / "steps.jsonl").splitlines()[0]),
        )

    result = compare_first_inputs(
        inputs(old_case, old_manifest, evidence.json(old_case / "chunks.json")[0]),
        inputs(case, evidence.json(case / "manifest.json"), chunk),
    )
    result["source_case"] = str(old_case)
    return result


def expected_commands(config, part):
    relative = f"configs/wam_raw8_seed43_task{part}_20261021.json"
    service = [
        str(Path(config["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--config",
        relative,
        "--seed",
        "42",
        "--checkpoint",
        "8400_masked",
        "--port",
        str(9744 + part),
        "--active-slot-mask",
    ]
    matrix = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_position_matrix.py",
        "--config",
        relative,
        "--seed",
        "42",
        "--source",
        "wam",
        "--variant",
        "position_hold",
        "--checkpoint",
        "8400_masked",
    ]
    return service, matrix


def check_identity_roster(part, config, launch, snapshot, manifest):
    assert snapshot["partition"] == part and snapshot["ready"] and snapshot["actual_http_info_equals_saved"]
    assert (snapshot["physical_gpu"], snapshot["port"]) == (4 + part, 9744 + part)
    entries = snapshot["local_registered_processes"]
    assert len(entries) == 4 and {p["role"] for p in entries} == {"coordinator", "model", "matrix", "physics"}
    assert len({p["pid"] for p in entries}) == 4
    owned = {p["role"]: p for p in entries}
    service, matrix = expected_commands(config, part)
    assert manifest["config"] == config and manifest["physical_gpu"] == 4 + part
    assert manifest["model_command"] == service and manifest["jobs"] == [["position_hold", matrix]]
    assert manifest["coordinator_pid"] == launch["coordinator"]["pid"] == owned["coordinator"]["pid"]
    assert launch["coordinator"]["start_ticks"] == owned["coordinator"]["start_ticks"]
    assert owned["coordinator"]["command"] == launch["command"]
    assert owned["model"]["command"] == service and owned["matrix"]["command"] == matrix
    for role, item in owned.items():
        assert type(item["pid"]) is int and item["pid"] > 0
        assert type(item["start_ticks"]) is int and item["start_ticks"] > 0
        assert item["cwd"] == str(REPO) and item["proc_state"] not in ("Z", "X")
        assert item["environment"]["CUDA_VISIBLE_DEVICES"] == (str(4 + part) if role == "model" else "")
        assert item["environment"]["LD_LIBRARY_PATH"] == ""
        if role in ("model", "matrix"):
            assert item["ppid"] == owned["coordinator"]["pid"]
    assert owned["physics"]["ppid"] == owned["matrix"]["pid"]
    assert owned["physics"]["environment"]["MUJOCO_GL"] == "osmesa"
    assert owned["physics"]["environment"]["PYOPENGL_PLATFORM"] == "osmesa"
    worker = owned["physics"]["command"]
    assert worker[0] == str((REPO / config["simulation_python"]).absolute())
    assert worker[1] == str(REPO / "scripts/run_wam_position_matrix.py")
    for flag, value in (
        ("--seed", "42"),
        ("--source", "wam"),
        ("--variant", "position_hold"),
        ("--sampling-seed", "43"),
        ("--record", f"task{part}_init0"),
        ("--checkpoint", "8400_masked"),
    ):
        assert worker[worker.index(flag) + 1] == value
    assert Path(worker[worker.index("--config") + 1]) == REPO / f"configs/wam_raw8_seed43_task{part}_20261021.json"
    return owned


def check_info(info, config, part, evidence):
    checkpoint = config["checkpoints"][0]
    assert (
        Path(info["checkpoint_path"]).resolve()
        == (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    )
    assert Path(info["config_file"]).resolve() == (Path(checkpoint["run"]) / "config.yaml").resolve()
    assert info["port"] == 9744 + part and info["attention_backend"] == "sdpa" and not info["ema_weights"]
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
    assert info["output_dir"] == str(REPO / config["output_dir"] / "intervention_services" / str(9744 + part))
    intervention = info["acceptance_intervention"]
    assert intervention["active_slot_mask"] and not intervention["exact_published_caption"]
    assert intervention["warm_start_contract_run"] is None
    for path in ("scripts/serve_wam_acceptance.py", "src/s2a/wam_runtime.py"):
        evidence.check(path, intervention["code"]["files_sha256"][path])


def preflight(launch_path, launch_sha, ready_path, ready_sha, evidence):
    configs = configurations(evidence)
    evidence.check(launch_path, launch_sha)
    launch = evidence.json(launch_path)
    assert launch["schema_version"] == "s2a.raw8-seed43-root-launch.v1" and launch["registered_budget"] == BUDGET
    assert [p["partition"] for p in launch["partitions"]] == [0, 1, 2]
    preparation = launch["preflight"]
    evidence.check(preparation["path"], preparation["sha256"])
    prepared = evidence.json(preparation["path"])
    assert prepared["schema_version"] == "s2a.raw8-seed43-preflight.v1" and prepared["complete"]
    assert prepared["registered_budget"] == BUDGET
    expected_commit = subprocess.check_output(["git", "rev-parse", REGISTRATION], cwd=REPO, text=True).strip()
    assert prepared["registration_commit"] == expected_commit
    for label, expected in PREREQUISITES.items():
        item = prepared["prerequisites"][label]
        assert item["sha256"] == expected
        evidence.check(item["path"], expected)
    assert len(prepared["prior12_owned_processes_absent"]) == 12
    assert all(not p["present"] for p in prepared["prior12_owned_processes_absent"])
    for name, digest in prepared["input_sha256"].items():
        evidence.check(name, digest)
    for part, (actual, original) in enumerate(zip(launch["partitions"], prepared["partitions"], strict=True)):
        assert original["partition"] == actual["partition"] == part
        assert original["physical_gpu"] == 4 + part and original["port"] == 9744 + part
        assert actual["command"] == original["command"]
        config_path = REPO / f"configs/wam_raw8_seed43_task{part}_20261021.json"
        assert original["config_path"] == str(config_path) and original["config_sha256"] == sha256(config_path)
    evidence.check(ready_path, ready_sha)
    ready = evidence.json(ready_path)
    assert ready["schema_version"] == "s2a.raw8-seed43-live-ready-snapshot.v1" and ready["ready"]
    assert ready["new_model_requests"] == ready["new_physics_steps"] == 0
    assert ready["local_network_operations"] == "three GET /info requests only"
    assert ready["nvml_to_local_pid_mapping_claimed"] is False
    assert ready["input_sha256"][str(Path(launch_path).resolve())] == launch_sha
    for name, digest in ready["input_sha256"].items():
        evidence.check(name, digest)
    assert [p["partition"] for p in ready["partitions"]] == [0, 1, 2]
    processes = [p for part in ready["partitions"] for p in part["local_registered_processes"]]
    assert len(processes) == len({p["pid"] for p in processes}) == 12
    return configs, launch, prepared, ready


def checkpoint_stats(config, evidence):
    """Recheck only the163 files of the actual8400, never unrelated retained weights."""
    path = REPO / "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
    evidence.check(path, "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8")
    item = next(c for c in evidence.json(path)["checkpoints"] if c["name"] == "8400_masked")
    ck = config["checkpoints"][0]
    assert Path(item["checkpoint"]).resolve() == (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
    assert len(item["files"]) == 163
    for filename, saved in item["files"].items():
        stat = Path(filename).stat()
        assert stat.st_size == saved["bytes"] and stat.st_mtime_ns == saved["mtime_ns"]
        assert len(saved["sha256"]) == 64
    return dict(
        checkpoint="8400_masked",
        files_stat_rechecked=163,
        full_weight_bytes_rehashed=False,
        unrelated_checkpoint_files_stat_checked=False,
    )


def service_artifacts(case, chunks, service, evidence):
    directories = []
    for index in range(len(chunks)):
        response = evidence.json(case / f"response_{index:04d}.json")
        directory = Path(response["args"]["output_dir"]).resolve()
        assert directory.parent == service and re.fullmatch(r"generate_[a-f0-9]{32}", directory.name)
        assert evidence.json(directory / "sample_args.json") == response["args"]
        assert evidence.json(directory / "sample_outputs.json") == response
        directories.append(str(directory))
    assert len(directories) == len(set(directories)) == 75
    assert set(map(Path, directories)) == {p.resolve() for p in service.glob("generate_*") if p.is_dir()}
    for pattern, name in (
        ("request_*.json", "request_{:04d}.json"),
        ("response_*.json", "response_{:04d}.json"),
        ("input_*.png", "input_{:04d}.png"),
    ):
        assert {p.name for p in case.glob(pattern)} == {name.format(i) for i in range(75)}
    assert {p.name for p in case.parent.iterdir() if p.is_dir()} == {case.name}
    return dict(generated_request_directories=directories, all75_saved_service_args_and_responses_exact=True)


def http_budget(text, model_pid, port):
    """Exact POST budget within the one owned, normally shut down service lifetime."""
    assert re.findall(r"Started server process \[(\d+)\]", text) == [str(model_pid)]
    assert re.findall(r"Finished server process \[(\d+)\]", text) == [str(model_pid)]
    markers = [
        f"Started server process [{model_pid}]",
        f"Uvicorn running on http://127.0.0.1:{port}",
        "Shutting down",
        "Waiting for application shutdown.",
        "Application shutdown complete.",
        f"Finished server process [{model_pid}]",
    ]
    assert all(text.count(marker) == 1 for marker in markers)
    offsets = [text.index(marker) for marker in markers]
    assert offsets == sorted(offsets)
    requests = list(re.finditer(r'"([A-Z]+) ([^ ]+) HTTP/1\.[01]" (\d{3})', text))
    posts = [r for r in requests if r[1] == "POST"]
    assert len(posts) == text.count('"POST ') == 75
    assert all((r[2], r[3]) == ("/generate", "200") for r in posts)
    assert all(offsets[1] < r.start() < offsets[2] for r in posts)
    return dict(
        successful_http_requests=75,
        no_extra_post=True,
        owned_pid_ordered_shutdown=True,
        marker_offsets_decoded_text_characters=offsets,
    )


def resources(part, config, launch, ready, evidence):
    root = REPO / config["output_dir"]
    directory = root / "orchestration/8400_masked_position_hold"
    manifest = evidence.json(directory / "manifest.json")
    config_path = REPO / f"configs/wam_raw8_seed43_task{part}_20261021.json"
    assert manifest["config_sha256"] == sha256(config_path)
    owned = check_identity_roster(part, config, launch, ready, manifest)
    model = evidence.json(directory / "model_process.json")
    assert model == dict(pid=owned["model"]["pid"], gpu=4 + part, port=9744 + part)
    matrix = evidence.json(directory / "position_hold_process.json")
    assert matrix == dict(pid=owned["matrix"]["pid"], command=owned["matrix"]["command"])
    logs = root / "simulation/logs/8400_masked_position_hold"
    worker = evidence.json(logs / f"task{part}_init0_s43.process.json")
    assert worker == dict(pid=owned["physics"]["pid"], command=owned["physics"]["command"])
    snapshot_path = logs / "execution_snapshot.json"
    snapshot = evidence.json(snapshot_path)
    assert snapshot["identity"] == ["wam", "position_hold", "8400_masked"]
    assert snapshot["jobs"] == [[f"task{part}_init0", 43]] and snapshot["cpu_workers"] == 1
    assert snapshot["coordinator_pid"] == owned["matrix"]["pid"] and snapshot["config_sha256"] == sha256(config_path)
    assert not snapshot["model_started"]
    assert (
        Path(owned["physics"]["command"][owned["physics"]["command"].index("--worker-snapshot") + 1]) == snapshot_path
    )
    for name, digest in snapshot["execution_sha256"].items():
        evidence.check(name, digest)
    info = evidence.json(directory / "server_info.json")
    assert info == ready["actual_info"]
    check_info(info, config, part, evidence)
    processes = {
        role: process_record(p["pid"], p["environment"]["CUDA_VISIBLE_DEVICES"], start_ticks=p["start_ticks"])
        for role, p in owned.items()
    }
    complete = optional_json(directory / "complete.json", evidence)
    release = optional_json(directory / "release.json", evidence)
    failure = optional_json(directory / "error.json", evidence)
    matrix_path = root / "simulation/position_matrix_8400_masked_position_hold.json"
    matrix_result = optional_json(matrix_path, evidence)
    released = False
    budget = None
    if complete is not None and release is not None and not any(p["present"] for p in processes.values()):
        assert failure is None
        assert complete == dict(complete=True, variants=["position_hold"])
        assert release == dict(model_pid=owned["model"]["pid"], model_returncode=release["model_returncode"])
        assert release["model_returncode"] in (0, -15)
        assert matrix_result is not None and matrix_result["complete"] and not matrix_result["model_started"]
        assert matrix_result["execution_snapshot_sha256"] == sha256(snapshot_path)
        assert matrix_result["cases"] == [
            dict(
                record=f"task{part}_init0",
                sampling_seed=43,
                returncode=0,
                command=owned["physics"]["command"],
                pid=owned["physics"]["pid"],
            )
        ]
        budget = http_budget(evidence.read(directory / "service.log").decode(), owned["model"]["pid"], 9744 + part)
        released = True
    return dict(
        partition=part,
        physical_gpu=4 + part,
        port=9744 + part,
        processes=processes,
        actual_ready_info_equals_saved=True,
        recorded_error=failure,
        matrix_result=matrix_result,
        complete_record=complete,
        release=release,
        final_http_budget=budget,
        owned_service_released_normally=released,
        generic_release_does_not_record_coordinator_or_worker_wait_status=True,
    )


def opening_tracking(rows):
    errors = np.array([r["opening_target"] - r["opening_after"] for r in rows])
    assert errors.shape == (600,) and np.isfinite(errors).all()
    return dict(mae=float(np.mean(np.abs(errors))), rmse=float(np.sqrt(np.mean(errors**2))))


def group_outcome(cases):
    if len(cases) != 3 or any(c["status"] != "verified" for c in cases):
        return None
    return {
        name: dict(
            successes=sum(c["result"]["outcome"][name] for c in cases),
            trials=3,
            rate=sum(c["result"]["outcome"][name] for c in cases) / 3,
        )
        for name in ("ever", "terminal", "stable")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--ready-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    assert args.launch.resolve().is_relative_to(REPO / "outputs") and args.ready.resolve().is_relative_to(
        REPO / "outputs"
    )
    evidence = Evidence()
    evidence.read(__file__)
    proof = source_proof(evidence)
    fixed014 = subprocess.check_output(
        ["git", "show", "1444f337ba7fcae83f1cba88faac4eadeface642:scripts/wam_position_acceptance_checker.py"]
    )
    assert hashlib.sha256(fixed014).hexdigest() == proof["new_checker_sha256"]
    configs, launch, prepared, ready = preflight(
        args.launch, args.launch_sha256, args.ready, args.ready_sha256, evidence
    )
    weights_before = checkpoint_stats(configs[0], evidence)
    fixture = evidence.json(REPO / configs[0]["fixture"])
    cases, resource_results, failures = [], [], []
    for part, config in enumerate(configs):
        resources_result = resources(part, config, launch["partitions"][part], ready["partitions"][part], evidence)
        resource_results.append(resources_result)
        case = REPO / config["output_dir"] / "simulation/8400_masked_position_hold" / f"task{part}_init0_s43"
        result = optional_json(case / "result.json", evidence)
        if result is None:
            cases.append(dict(partition=part, record=f"task{part}_init0", status="pending", result=None))
            continue
        try:
            record = next(r for r in fixture["records"] if r["name"] == f"task{part}_init0")
            actual = verify_case(
                config,
                record,
                config["variants"][0],
                case,
                evidence,
                source="wam",
                checkpoint="8400_masked",
                full=True,
                sampling_seed=43,
                first_query_validator=first_query,
            )
            chunks = evidence.json(case / "chunks.json")
            assert actual["steps"] == 600 and actual["chunks"] == len(chunks) == 75
            assert [c["step"] for c in chunks] == list(range(0, 600, 8))
            assert actual["raw_checkpoint_evidence"] and not actual["component_ablation"]
            service = REPO / config["output_dir"] / "intervention_services" / str(9744 + part)
            provenance = service_artifacts(case, chunks, service, evidence)
            rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
            actual["opening_tracking"] = opening_tracking(rows)
            actual["first_query019_provenance"] = actual.pop("first_query003_regression")
            actual["success_duration_s"] = actual["outcome"]["successful_steps"] / 20
            actual["trailing_success_duration_s"] = actual["outcome"]["trailing_success_steps"] / 20
            cases.append(
                dict(
                    partition=part,
                    record=record["name"],
                    status="verified",
                    result=actual,
                    service_provenance=provenance,
                )
            )
        except Exception:
            failures.append(dict(partition=part, case=str(case), error=traceback.format_exc()))
            cases.append(dict(partition=part, record=f"task{part}_init0", status="error", result=None))
    verified = [c for c in cases if c["status"] == "verified"]
    all_cases = len(verified) == 3 and not failures
    released = all(r["owned_service_released_normally"] for r in resource_results)
    complete = all_cases and released
    evidence.check(OLD_AUDIT, OLD_AUDIT_SHA)
    old = evidence.json(OLD_AUDIT)
    old_cases = [c for c in old["cases"] if c["profile"] == "interval8_alpha1"]
    assert len(old_cases) == 3
    weights_after = checkpoint_stats(configs[0], evidence)
    assert weights_before == weights_after
    evidence.recheck()
    output = dict(
        schema_version="s2a.raw8-seed43-independent-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=complete,
        all_cases_independently_verified=all_cases,
        normal_process_release=released,
        registration_commit=prepared["registration_commit"],
        registered_budget=BUDGET,
        verified_cases=len(verified),
        verified_physics_steps=sum(c["result"]["steps"] for c in verified),
        verified_queries=sum(c["result"]["chunks"] for c in verified),
        verified_http_queries=sum(
            r["final_http_budget"]["successful_http_requests"] for r in resource_results if r["final_http_budget"]
        ),
        actual_checkpoint_stat=weights_after,
        experiment_seed=42,
        sampling_seed=43,
        source_proof=proof,
        cases=cases,
        group_outcome=group_outcome(cases),
        resources=resource_results,
        errors=failures,
        seed42_comparison=dict(
            source_path=str(OLD_AUDIT),
            source_sha256=OLD_AUDIT_SHA,
            trials=3,
            cases=[dict(record=c["record"], outcome=c["outcome"], tracking=c["tracking"]) for c in old_cases],
            separate_seed_denominators=True,
            later_closed_loop_inputs_are_not_paired=True,
        ),
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_initial_states=0,
        limitations=[
            "Three old initial states and a fixed second sampling seed are development replication, not fresh or multi-initial-state acceptance.",
            "Raw43 is authenticated by its own request/response/service files; no raw43=raw42 assertion or substitution is made.",
            "014 reconstructs every600-endpoint command/motor/qpos/reference/clock and native stable predicate; oracle015 only qualifies execution and is excluded from this WAM denominator.",
            "Object contact is metrics only; absence of target finger contact cannot disprove native task success.",
            "Current opening tracks reference aperture, not force or certified grasp/release; low TCP error alone cannot uniquely attribute task failure to model or S2A.",
            "Generic service complete.json only declares its variant; independent actual case rows,75 query/service directories/HTTP200 per part and12 ready-bound PID/startticks are checked separately.",
            "Only model returncode and worker returncode are saved by the generic service/matrix; coordinator/matrix normal completion follows their complete artifact and absent owned identity, not an invented Popen wait status.",
            "Actual caption/active-mask branching is source-authenticated; this unchanged service has no complete per-query prepared-caption or final in-memory noise/poison ledger.",
        ],
    )
    write_new(args.output, output)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                complete=complete,
                verified_cases=len(verified),
                verified_queries=output["verified_queries"],
                errors=len(failures),
            )
        )
    )
    if failures or (args.full and not complete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
