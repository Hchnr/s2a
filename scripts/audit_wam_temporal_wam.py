"""Read-only020 WAM audit: causal reference ensembles are not raw checkpoint scores.

The frozen af147e1 checker independently reconstructs all600 ensemble targets,
contributors, original128 outputs and014 mechanics. This wrapper adds the
registered3-trial lifecycle, exact first-query42 regression and resource gates.
No environment/model, child process or HTTP request is started here.
"""

import argparse
import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from audit_wam_raw8_seed43 import (
    FIRST_CONDITIONS,
    OLD_AUDIT,
    OLD_AUDIT_SHA,
    checkpoint_stats,
    group_outcome,
    http_budget,
    opening_tracking,
    optional_json,
    service_artifacts,
)
from run_wam_short_replan import cpu_environment, process_record, write_new
from wam_temporal_position_checker import CONTRACT, source_proof, verify_case, verify_query

from s2a.artifacts import REPO, sha256

REGISTRATION = "7025c45"
PLAN = "plans/v0.0.1/020_causal_temporal_ensemble_20261021.md"
BUDGET = dict(wam_trials=3, physics_steps=1800, maximum_h40_queries=225, fresh_initial_states=0)
PRIOR019_SHA = "c2c09f2ea8fefb231577f282d7fe6473af12d6312f076b0fd99c083636398d21"


def registered(relative, evidence):
    digest = hashlib.sha256(
        subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
    ).hexdigest()
    evidence.check(REPO / relative, digest)
    return digest


def configurations(evidence):
    registered(PLAN, evidence)
    base = evidence.json(REPO / "configs/wam_short_replan_8_alpha1_20261021.json")
    configs = []
    for part in range(3):
        relative = f"configs/wam_temporal_equal_task{part}_20261021.json"
        registered(relative, evidence)
        config = evidence.json(REPO / relative)
        expected = json.loads(json.dumps(base))
        expected.update(
            output_dir=f"outputs/wam_stability_20261021/temporal_equal_development/task{part}",
            records=[f"task{part}_init0"],
            notes=config["notes"],
            variants=[dict(base["variants"][0], name="temporal_equal")],
            model_reference_label="causal_temporal_equal_ensemble",
            reference_ensemble=CONTRACT,
        )
        expected["checkpoints"] = [dict(base["checkpoints"][0], gpu=4 + part, port=9844 + part)]
        assert config == expected, "020 differs from frozen raw8 beyond registered ensemble/resources/record"
        assert config["sampling_seeds"] == [42]
        evidence.check(REPO / config["fixture"], config["fixture_sha256"])
        configs.append(config)
    return configs


def compare_first_inputs(old, current):
    """The first actual model forecast is exact42 regression before paths diverge."""
    assert type(old["request"]["seed"]) is int and old["request"]["seed"] == 42
    assert type(current["request"]["seed"]) is int and current["request"]["seed"] == 42
    for key in FIRST_CONDITIONS:
        assert old["request"][key] == current["request"][key], key
    assert old["record"] == current["record"] and old["controller"] == current["controller"]
    assert old["png"] == current["png"]
    assert old["variant"]["name"] == "position_hold" and current["variant"]["name"] == "temporal_equal"
    assert {k: v for k, v in old["variant"].items() if k != "name"} == {
        k: v for k, v in current["variant"].items() if k != "name"
    }
    for key in ("anchor_tcp", "anchor_opening", "raw128", "effective128"):
        np.testing.assert_array_equal(old["chunk"][key], current["chunk"][key])
    np.testing.assert_array_equal(old["normalized_action"], current["normalized_action"])
    for value in (old, current):
        chunk = value["chunk"]
        assert chunk["step"] == 0 and chunk["physics_unchanged"]
        assert chunk["physics_before_sha256"] == chunk["physics_after_sha256"]
        assert chunk["physics_before_sha256"] == value["controller"]["actual_initial_physics_sha256"]
        assert chunk["physics_before_time_s"] == chunk["physics_after_time_s"]
    for key in ("physics_before_sha256", "physics_before_time_s"):
        assert old["chunk"][key] == current["chunk"][key]
    for key in ("measured_source", "measured_twist", "opening_before", "gripper_qpos_before", "motor_target_before"):
        np.testing.assert_array_equal(old["row"][key], current["row"][key])
    return dict(
        source_sampling_seed=42,
        actual_sampling_seed=42,
        first_query_full40x128_raw_exact=True,
        first_query_full40x128_normalized_exact=True,
        first_image_png_bytes_exact=True,
        first_initial_anchor_physics_controller_motor_exact=True,
        model_conditions_exact=True,
        variant_name_only_changed=True,
        scope="first query only; later causal ensemble execution changes observations; no later raw equality assumed",
    )


def first_query(config, record, variant, case, chunk, evidence, checkpoint, *, sampling_seed):
    assert sampling_seed == 42 and config["sampling_seeds"] == [42]
    verify_query(config, case, 0, chunk, evidence, checkpoint, sampling_seed=42)
    evidence.check(OLD_AUDIT, OLD_AUDIT_SHA)
    old = evidence.json(OLD_AUDIT)
    previous = next(c for c in old["cases"] if c["profile"] == "interval8_alpha1" and c["record"] == record["name"])
    folder = (REPO / previous["case"]).resolve()
    for name in (
        "manifest.json",
        "controller.json",
        "chunks.json",
        "request_0000.json",
        "response_0000.json",
        "input_0000.png",
        "steps.jsonl",
    ):
        path = folder / name
        evidence.check(path, old["source_before_after_hashes"][str(path)])

    def inputs(path, first):
        manifest = evidence.json(path / "manifest.json")
        response = evidence.json(path / "response_0000.json")
        return dict(
            request=evidence.json(path / "request_0000.json"),
            record=manifest["record"],
            variant=manifest["variant"],
            controller=evidence.json(path / "controller.json"),
            png=evidence.read(path / "input_0000.png"),
            chunk=first,
            normalized_action=response["outputs"][0]["content"]["action128"]["normalized_action"],
            row=json.loads(evidence.read(path / "steps.jsonl").splitlines()[0]),
        )

    verdict = compare_first_inputs(inputs(folder, evidence.json(folder / "chunks.json")[0]), inputs(case, chunk))
    verdict["source_case"] = str(folder)
    return verdict


def expected_commands(config, part):
    relative = f"configs/wam_temporal_equal_task{part}_20261021.json"
    model = [
        str(Path(config["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--config",
        relative,
        "--seed",
        "42",
        "--checkpoint",
        "8400_masked",
        "--port",
        str(9844 + part),
        "--active-slot-mask",
    ]
    matrix = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_temporal_matrix.py",
        "--config",
        relative,
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


def check_gate(path, digest, part, config, evidence):
    evidence.check(path, digest)
    gate = evidence.json(path)
    assert gate["schema_version"] == "s2a.temporal-equal-launch-gate.v1" and gate["all_prerequisites_passed"]
    assert gate["partition"] == part
    for filename, expected in gate["input_sha256"].items():
        evidence.check(filename, expected)
    required = [
        PLAN,
        f"configs/wam_temporal_equal_task{part}_20261021.json",
        "scripts/run_wam_temporal_service.py",
        "scripts/capture_wam_temporal_ready.py",
        "scripts/run_wam_temporal_matrix.py",
        "scripts/serve_wam_acceptance.py",
        "src/s2a/temporal_ensemble.py",
        "src/s2a/temporal_ensemble_simulation.py",
        "src/s2a/wam_runtime.py",
    ]
    assert all(gate["input_sha256"][str(REPO / filename)] == sha256(REPO / filename) for filename in required)
    qualification = gate["qualification"]
    evidence.check(qualification["path"], qualification["sha256"])
    oracle = evidence.json(qualification["path"])
    assert oracle["schema_version"] == "s2a.temporal-equal-oracle-qualification.v1"
    assert oracle["complete"] and oracle["qualified"] and oracle["normal_process_release"] and not oracle["errors"]
    assert oracle["source_oracle_only"] and not oracle["oracle_is_wam_success_evidence"]
    assert oracle["registered_budget"] == dict(
        oracle_trials=3, physical_steps=1800, model_requests=0, fresh_initial_states=0
    )
    assert oracle["verified_oracle_cases"] == 3 and oracle["verified_physics_log_endpoints"] == 1800
    assert oracle["all1800_common_step_fields_exact_to_007"]
    previous = gate["prior_seed43"]
    assert previous["sha256"] == PRIOR019_SHA
    evidence.check(previous["path"], PRIOR019_SHA)
    prior = evidence.json(previous["path"])
    assert prior["schema_version"] == "s2a.raw8-seed43-independent-audit.v1"
    assert (
        prior["complete"]
        and prior["all_cases_independently_verified"]
        and prior["normal_process_release"]
        and not prior["errors"]
    )
    assert (
        prior["verified_cases"],
        prior["verified_physics_steps"],
        prior["verified_queries"],
        prior["verified_http_queries"],
    ) == (3, 1800, 225, 225)
    for item in (qualification, previous):
        assert gate["input_sha256"][str(Path(item["path"]).resolve())] == item["sha256"]
        for filename, expected in evidence.json(item["path"])["input_sha256"].items():
            assert gate["input_sha256"][filename] == expected
    assert config["reference_ensemble"] == CONTRACT
    return gate


def check_roster(part, config, launch, snapshot, manifest):
    assert snapshot["partition"] == manifest["partition"] == part and snapshot["ready"]
    assert snapshot["physical_gpu"] == manifest["physical_gpu"] == launch["physical_gpu"] == 4 + part
    assert snapshot["port"] == launch["port"] == 9844 + part and snapshot["actual_http_info_equals_saved"]
    values = snapshot["local_registered_processes"]
    assert len(values) == len({p["pid"] for p in values}) == 4
    assert {p["role"] for p in values} == {"coordinator", "model", "matrix", "physics"}
    roles = {p["role"]: p for p in values}
    assert manifest["config"] == config and not manifest["automatic_retry"] and manifest["external_sources_readonly"]
    assert roles["coordinator"]["pid"] == manifest["coordinator"]["pid"] == launch["coordinator"]["pid"]
    assert (
        roles["coordinator"]["start_ticks"]
        == manifest["coordinator"]["start_ticks"]
        == launch["coordinator"]["start_ticks"]
    )
    model, matrix = expected_commands(config, part)
    assert manifest["model_command"] == roles["model"]["command"] == model
    assert manifest["job_command"] == roles["matrix"]["command"] == matrix
    assert roles["coordinator"]["command"] == launch["command"]
    command = launch["command"]
    assert Path(command[0]).absolute() == REPO / ".venv/bin/python"
    assert Path(command[1]).resolve() == REPO / "scripts/run_wam_temporal_service.py"
    for flag, value in (
        ("--partition", str(part)),
        ("--seed", "42"),
        ("--gate", launch["prerequisite_gate_path"]),
        ("--gate-sha256", launch["prerequisite_gate_sha256"]),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--dry-run" not in command
    for role, identity in roles.items():
        assert type(identity["pid"]) is int and identity["pid"] > 0
        assert type(identity["start_ticks"]) is int and identity["start_ticks"] > 0
        assert identity["cwd"] == str(REPO) and identity["proc_state"] not in ("Z", "X")
        assert identity["environment"]["CUDA_VISIBLE_DEVICES"] == (str(4 + part) if role == "model" else "")
        assert identity["environment"]["LD_LIBRARY_PATH"] == ""
        if role in ("model", "matrix"):
            assert identity["ppid"] == roles["coordinator"]["pid"]
    physics = roles["physics"]
    assert physics["ppid"] == roles["matrix"]["pid"]
    assert physics["environment"]["MUJOCO_GL"] == physics["environment"]["PYOPENGL_PLATFORM"] == "osmesa"
    worker = physics["command"]
    assert worker[0] == str((REPO / config["simulation_python"]).absolute())
    assert worker[1] == str(REPO / "scripts/run_wam_temporal_matrix.py")
    for flag, value in (
        ("--seed", "42"),
        ("--source", "wam"),
        ("--variant", "temporal_equal"),
        ("--record", f"task{part}_init0"),
        ("--sampling-seed", "42"),
        ("--checkpoint", "8400_masked"),
    ):
        assert worker[worker.index(flag) + 1] == value
    assert Path(worker[worker.index("--config") + 1]) == REPO / f"configs/wam_temporal_equal_task{part}_20261021.json"
    return roles


def check_info(info, config, part, evidence):
    checkpoint = config["checkpoints"][0]
    assert (
        Path(info["checkpoint_path"]).resolve()
        == (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    )
    assert Path(info["config_file"]).resolve() == (Path(checkpoint["run"]) / "config.yaml").resolve()
    assert info["port"] == 9844 + part and info["attention_backend"] == "sdpa" and not info["ema_weights"]
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
    assert info["output_dir"] == str(REPO / config["output_dir"] / "intervention_services" / str(9844 + part))
    intervention = info["acceptance_intervention"]
    assert intervention["active_slot_mask"] and not intervention["exact_published_caption"]
    assert intervention["warm_start_contract_run"] is None
    for path in ("scripts/serve_wam_acceptance.py", "src/s2a/wam_runtime.py"):
        evidence.check(path, intervention["code"]["files_sha256"][path])


def check_release(released, roles):
    """The new coordinator actually saves these identities and wait statuses."""
    assert released["model_pid"] == roles["model"]["pid"]
    assert released["model_start_ticks"] == roles["model"]["start_ticks"]
    assert released["model_returncode"] in (0, -15)
    assert released["last_job_pid"] == roles["matrix"]["pid"] and released["last_job_returncode"] == 0
    assert not released["cleanup_errors"] and released["only_owned_process_groups_stopped"]
    assert released["automatic_retry"] is False
    assert set(released["stop_evidence"]) == {"job", "model"}
    assert all(
        item["pid"] == roles["matrix" if role == "job" else role]["pid"]
        for role, item in released["stop_evidence"].items()
    )
    return True


def ensemble_reference_usage(rows, chunks):
    """Count causal contributor endpoints, separately from the newest-query cursor."""
    assert len(rows) == 600 and [r["step"] for r in rows] == list(range(600))
    assert [c["step"] for c in chunks] == list(range(0, 600, 8))
    lengths = {c["step"]: len(c["raw128"]) for c in chunks}
    assert all(length == 40 for length in lengths.values())
    used = {q: set() for q in lengths}
    exposure = {q: 0.0 for q in lengths}
    for step, row in enumerate(rows):
        contributors = row["temporal_ensemble"]["contributors"]
        expected = {(q, step - q) for q, count in lengths.items() if q <= step < q + count}
        actual = [(p["query_step"], p["endpoint_index"]) for p in contributors]
        assert len(actual) == len(set(actual)) and set(actual) == expected
        for item in contributors:
            assert item["absolute_endpoint"] == step + 1 and item["weight"] == 1 / len(contributors)
            used[item["query_step"]].add(item["endpoint_index"])
            exposure[item["query_step"]] += item["weight"]
    queries = []
    for q, count in lengths.items():
        expected = set(range(min(count, 600 - q)))
        assert used[q] == expected
        queries.append(
            dict(
                query_step=q,
                generated_endpoints=count,
                recorded_contributor_endpoints=len(used[q]),
                unavailable_after_physical_budget=count - len(used[q]),
                equal_weight_exposure=exposure[q],
            )
        )
    generated = sum(lengths.values())
    contributing = sum(len(indices) for indices in used.values())
    assert generated == 3000 and contributing == 2920
    assert abs(sum(exposure.values()) - 600) <= 1e-10
    return dict(
        generated_endpoints=generated,
        recorded_contributor_endpoints=contributing,
        contributor_endpoint_fraction=contributing / generated,
        physical_targets=600,
        unavailable_after_physical_budget=generated - contributing,
        per_query=queries,
        scope="Recorded causal contributors, not physical steps or accuracy; identity_fast_path copies an equivalent newest reference. Latest-query discarded_on_requery_s is not total ensemble suffix discard.",
    )


def resources(part, config, launch, ready, evidence):
    root = REPO / config["output_dir"]
    directory = root / "orchestration"
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["schema_version"] == "s2a.temporal-equal-service.v1"
    assert manifest["config_sha256"] == sha256(REPO / f"configs/wam_temporal_equal_task{part}_20261021.json")
    roles = check_roster(part, config, launch, ready, manifest)
    gate_path, gate_sha = launch["prerequisite_gate_path"], launch["prerequisite_gate_sha256"]
    assert manifest["source_sha256"][str(Path(gate_path).resolve())] == gate_sha
    for filename, expected in manifest["source_sha256"].items():
        evidence.check(filename, expected)
    for role in ("model", "matrix"):
        entry = evidence.json(directory / f"{role}_process.json")
        assert entry["pid"] == roles[role]["pid"] and entry["start_ticks"] == roles[role]["start_ticks"]
        assert entry["command"] == roles[role]["command"]
        assert entry["cuda_visible_devices"] == (str(4 + part) if role == "model" else "")
        assert entry["ld_library_path"] == "" and entry["present"]
    logs = root / "simulation/logs/8400_masked_temporal_equal"
    worker = evidence.json(logs / f"task{part}_init0_s42.process.json")
    assert worker["pid"] == roles["physics"]["pid"] and worker["start_ticks"] == roles["physics"]["start_ticks"]
    assert worker["command"] == roles["physics"]["command"]
    assert worker["ppid"] == roles["physics"]["ppid"] and worker["environment"] == roles["physics"]["environment"]
    snapshot_path = logs / "execution_snapshot.json"
    assert Path(worker["command"][worker["command"].index("--worker-snapshot") + 1]) == snapshot_path
    snapshot = evidence.json(snapshot_path)
    assert snapshot["identity"] == ["wam", "temporal_equal", "8400_masked"] and not snapshot["model_started"]
    assert snapshot["coordinator_pid"] == roles["matrix"]["pid"] and snapshot["cpu_workers"] == 1
    assert snapshot["coordinator_identity"]["start_ticks"] == roles["matrix"]["start_ticks"]
    assert snapshot["jobs"] == [[f"task{part}_init0", 42]]
    assert snapshot["config_sha256"] == sha256(REPO / f"configs/wam_temporal_equal_task{part}_20261021.json")
    for filename, expected in snapshot["execution_sha256"].items():
        evidence.check(filename, expected)
    info = evidence.json(directory / "server_info.json")
    assert info == ready["actual_info"]
    check_info(info, config, part, evidence)
    processes = {
        role: process_record(p["pid"], p["environment"]["CUDA_VISIBLE_DEVICES"], start_ticks=p["start_ticks"])
        for role, p in roles.items()
    }
    completed = optional_json(directory / "complete.json", evidence)
    released = optional_json(directory / "release.json", evidence)
    error = optional_json(directory / "error.json", evidence)
    matrix = optional_json(root / "simulation/temporal_matrix_8400_masked_temporal_equal.json", evidence)
    normal, http = False, None
    if completed is not None and released is not None and not any(p["present"] for p in processes.values()):
        assert error is None
        assert completed == dict(complete=True, partition=part, wam_trials=1, physics_steps=600, queries=75)
        check_release(released, roles)
        assert matrix is not None and matrix["complete"] and not matrix["model_started"]
        assert matrix["execution_snapshot_sha256"] == sha256(snapshot_path)
        assert len(matrix["cases"]) == 1
        trial = matrix["cases"][0]
        assert trial["record"] == f"task{part}_init0" and trial["sampling_seed"] == 42 and trial["returncode"] == 0
        assert trial["pid"] == roles["physics"]["pid"] and trial["command"] == roles["physics"]["command"]
        assert trial["start_ticks"] == roles["physics"]["start_ticks"] and trial["ppid"] == roles["matrix"]["pid"]
        assert trial["environment"] == roles["physics"]["environment"]
        http = http_budget(evidence.read(directory / "service.log").decode(), roles["model"]["pid"], 9844 + part)
        normal = True
    return dict(
        partition=part,
        physical_gpu=4 + part,
        port=9844 + part,
        processes=processes,
        recorded_error=error,
        complete_record=completed,
        release=released,
        matrix_result=matrix,
        final_http_budget=http,
        owned_service_released_normally=normal,
    )


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
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    assert args.launch.resolve().is_relative_to(REPO / "outputs") and args.ready.resolve().is_relative_to(
        REPO / "outputs"
    )
    evidence = Evidence()
    evidence.read(__file__)
    for commit, relative in (
        ("5e2090c", "scripts/audit_wam_raw8_seed43.py"),
        ("af147e1", "scripts/wam_temporal_position_checker.py"),
    ):
        digest = hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO)).hexdigest()
        evidence.check(REPO / relative, digest)
    proof = source_proof(evidence)
    configs = configurations(evidence)
    evidence.check(args.launch, args.launch_sha256)
    launch = evidence.json(args.launch)
    assert launch["schema_version"] == "s2a.temporal-equal-root-launch.v1" and launch["registered_budget"] == BUDGET
    assert [p["partition"] for p in launch["partitions"]] == [0, 1, 2]
    gates = [
        check_gate(p["prerequisite_gate_path"], p["prerequisite_gate_sha256"], i, configs[i], evidence)
        for i, p in enumerate(launch["partitions"])
    ]
    evidence.check(args.ready, args.ready_sha256)
    ready = evidence.json(args.ready)
    assert ready["schema_version"] == "s2a.temporal-equal-live-ready-snapshot.v1" and ready["ready"]
    assert ready["new_model_requests"] == ready["new_physics_steps"] == 0
    assert (
        ready["local_network_operations"] == "three GET /info requests only"
        and not ready["nvml_to_local_pid_mapping_claimed"]
    )
    assert ready["input_sha256"][str(args.launch.resolve())] == args.launch_sha256
    for filename, expected in ready["input_sha256"].items():
        evidence.check(filename, expected)
    assert [p["partition"] for p in ready["partitions"]] == [0, 1, 2]
    identities = [p for part in ready["partitions"] for p in part["local_registered_processes"]]
    assert len(identities) == len({p["pid"] for p in identities}) == 12
    weights_before = checkpoint_stats(configs[0], evidence)
    fixture = evidence.json(REPO / configs[0]["fixture"])
    cases, resources_list, errors = [], [], []
    for part, config in enumerate(configs):
        resources_list.append(resources(part, config, launch["partitions"][part], ready["partitions"][part], evidence))
        case = REPO / config["output_dir"] / "simulation/8400_masked_temporal_equal" / f"task{part}_init0_s42"
        if optional_json(case / "result.json", evidence) is None:
            cases.append(dict(partition=part, record=f"task{part}_init0", status="pending", result=None))
            continue
        try:
            record = next(r for r in fixture["records"] if r["name"] == f"task{part}_init0")
            result = verify_case(
                config,
                record,
                config["variants"][0],
                case,
                evidence,
                source="wam",
                checkpoint="8400_masked",
                full=True,
                sampling_seed=42,
                first_query_validator=first_query,
            )
            assert result["steps"] == 600 and result["chunks"] == 75
            assert not result["raw_checkpoint_evidence"] and not result["component_ablation"]
            assert result["reference_intervention"] == "causal_temporal_equal_ensemble"
            replay = result["independent_temporal_ensemble_replay"]
            assert replay["passed"] and replay["verified_endpoints"] == 600 and replay["same_absolute_endpoint"]
            assert 1 <= replay["max_contributors"] <= 5
            chunks = evidence.json(case / "chunks.json")
            assert [c["step"] for c in chunks] == list(range(0, 600, 8))
            service = REPO / config["output_dir"] / "intervention_services" / str(9844 + part)
            provenance = service_artifacts(case, chunks, service, evidence)
            rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
            result["opening_tracking"] = opening_tracking(rows)
            result["ensemble_reference_usage"] = ensemble_reference_usage(rows, chunks)
            result["latest_query_clock_diagnostics"] = result.pop("governor_diagnostics")
            result["latest_query_clock_diagnostics_scope"] = (
                "Inherited014 newest-query cursor only; past references remain ensemble contributors. Do not interpret discarded_on_requery_s as ensemble suffix discard."
            )
            result["first_query020_provenance"] = result.pop("first_query003_regression")
            result["success_duration_s"] = result["outcome"]["successful_steps"] / 20
            result["trailing_success_duration_s"] = result["outcome"]["trailing_success_steps"] / 20
            cases.append(
                dict(
                    partition=part,
                    record=record["name"],
                    status="verified",
                    result=result,
                    service_provenance=provenance,
                )
            )
        except Exception:
            errors.append(dict(partition=part, case=str(case), error=traceback.format_exc()))
            cases.append(dict(partition=part, record=f"task{part}_init0", status="error", result=None))
    verified = [c for c in cases if c["status"] == "verified"]
    all_cases = len(verified) == 3 and not errors
    release = all(r["owned_service_released_normally"] for r in resources_list)
    weights_after = checkpoint_stats(configs[0], evidence)
    assert weights_before == weights_after
    evidence.recheck()
    output = dict(
        schema_version="s2a.temporal-equal-wam-independent-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=all_cases and release,
        all_cases_independently_verified=all_cases,
        normal_process_release=release,
        registered_budget=BUDGET,
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
        group_outcome=group_outcome(cases),
        resources=resources_list,
        errors=errors,
        prerequisite_gates=[
            dict(partition=i, qualification=g["qualification"], prior_seed43=g["prior_seed43"])
            for i, g in enumerate(gates)
        ],
        actual_checkpoint_stat=weights_after,
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_initial_states=0,
        limitations=[
            "020 changes the admitted reference using only already-generated same-absolute-endpoint forecasts; it is not a raw-checkpoint score.",
            "Oracle qualifications are excluded from3 WAM trials. Three old initial states/seed42 are development, not fresh acceptance.",
            "First query raw/normalized/pixels/initial input exactly regress010/42; subsequent observations may diverge under ensemble control.",
            "Lower admitted-reference error may come from changing the reference; compare latest raw/model-clock and nominal-clock tracking separately.",
            "Old generated references can be stale after state/contact changes; smoother predictions alone are not accuracy or native task success.",
            "Target contact/opening metrics do not certify force/grasp/release or override the native stable predicate.",
            "Full closure requires3x600 endpoints/225 unique responses+HTTP200 and12 ready-bound identities absent; pending rates are null.",
            "The unchanged model service has no full per-request prepared-caption/final in-memory mask ledger; no such receipt is fabricated.",
        ],
    )
    write_new(args.output, output)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                complete=output["complete"],
                verified_cases=len(verified),
                errors=len(errors),
            )
        )
    )
    if errors or (args.full and not output["complete"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
