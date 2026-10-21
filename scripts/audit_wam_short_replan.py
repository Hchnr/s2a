"""Read-only010 audit of twelve short-replan WAM development trials.

Directly reuses the frozen006 full per-step verifier. Registration, profile
identity, denominator, service provenance and query budgets belong to010 here.
No model, oracle replay or physics environment is started by this entry.
"""

import argparse
import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_earlier_position import verify_case
from audit_wam_position import Evidence
from run_wam_short_replan import (
    CHECKPOINT,
    PRIOR_GATE,
    PRIOR_GATE_SHA,
    REGISTRATION_COMMIT,
    REPO,
    commands,
    config_path,
    cpu_environment,
    preflight,
    process_record,
    sha256,
    write_new,
)

SCHEMA = "s2a.short-replan-readonly-audit.v1"
VERIFIER = "scripts/audit_wam_earlier_position.py"
VERIFIER_SHA = "35c7e332f1fdfc6ff2b5e95c0af3311bba67b2815450d75a572c1432526b1119"
COORDINATOR = "scripts/run_wam_short_replan.py"
COORDINATOR_SHA = "b78d21e9f9826c99dc2d65288e7975d8cefb0c7d31e3f3abbeab2b2ef8b65524"
LAUNCH = "outputs/wam_stability_20261021/short_replan_coordinators/launch.json"
LAUNCH_SHA = "462e28f0b484b072e7f161759195abade70419e7108fe38f8d2090415d0c616a"


def optional_json(path, evidence):
    if not path.exists():
        return None
    try:
        json.loads(path.read_bytes())
    except json.JSONDecodeError:
        return None
    return evidence.json(path)


def resource_snapshot(interval, configs, launch, evidence):
    c = configs[interval, 1]
    ck = c["checkpoints"][0]
    gpu, port = ck["gpu"], ck["port"]
    directory = REPO / c["output_dir"] / "orchestration" / f"short_replan_interval{interval}"
    manifest = optional_json(directory / "manifest.json", evidence)
    if manifest is None:
        return dict(interval=interval, status="not_started", owned_service_released_normally=False)
    assert manifest["registration_commit"] == REGISTRATION_COMMIT
    assert manifest["interval"] == interval and manifest["physical_gpu"] == gpu and manifest["port"] == port
    assert manifest["global_budget"] == dict(
        wam_trials=12, physics_steps=7200, model_queries_upper_bound=678, added_oracle_trials=0
    )
    assert manifest["configurations"] == {str(alpha): configs[interval, alpha] for alpha in (1, 0)}
    service, jobs = commands(interval, configs)
    assert manifest["service_command"] == service and manifest["jobs"] == jobs
    assert manifest["prerequisite_gate_sha256"] == PRIOR_GATE_SHA
    assert manifest["gpu_memory_mib_before"] < 500
    for filename, signature in manifest["frozen_input_sha256"].items():
        evidence.check(filename, signature)
    assert manifest["frozen_input_sha256"][str(REPO / COORDINATOR)] == COORDINATOR_SHA
    service_root = (REPO / c["output_dir"] / "intervention_services" / str(port)).resolve()
    assert manifest["service_artifact_root_for_both_profiles"] == str(service_root)
    for job in jobs:
        binding = manifest["profile_service_binding"][job["profile"]]
        assert binding == dict(
            service_config=config_path(interval, 1),
            service_config_sha256=evidence.hashes[str(REPO / config_path(interval, 1))],
            service_artifact_root=str(service_root),
            cpu_config=job["config"],
            cpu_config_sha256=evidence.hashes[str(REPO / job["config"])],
        )
    leader = manifest["coordinator"]
    registered = next(r for r in launch["records"] if r["interval"] == interval)
    assert (registered["physical_gpu"], registered["port"]) == (gpu, port)
    assert registered["coordinator_pid"] == leader["pid"]
    assert int(registered["coordinator_start_ticks"]) == leader["start_ticks"]
    assert manifest["git_head"] == launch["launch_git_commit"]
    for profile in registered["profiles"]:
        evidence.check(profile["path"], profile["sha256"])
    coordinator = process_record(leader["pid"], "", start_ticks=leader["start_ticks"])
    model = optional_json(directory / "model_process.json", evidence)
    model_process = None
    if model is not None:
        assert model["gpu"] == gpu and model["port"] == port and model["command"] == service
        assert model["cuda_visible_devices"] == str(gpu) and model["ld_library_path"] == ""
        model_process = process_record(model["pid"], str(gpu), start_ticks=model["start_ticks"])
    matrix_processes, physics_processes, exits = {}, {}, {}
    for job in jobs:
        profile = job["profile"]
        worker = optional_json(directory / f"{profile}_process.json", evidence)
        if worker is not None:
            assert worker["command"] == job["command"]
            assert worker["cuda_visible_devices"] == worker["ld_library_path"] == ""
            matrix_processes[profile] = process_record(worker["pid"], "", start_ticks=worker["start_ticks"])
        exit_record = optional_json(directory / f"{profile}_exit.json", evidence)
        if exit_record is not None:
            assert worker is not None and exit_record["pid"] == worker["pid"]
            exits[profile] = exit_record
        logs = REPO / configs[interval, job["alpha"]]["output_dir"] / "simulation/logs/8400_masked_position_hold"
        physics_processes[profile] = []
        for name in c["records"]:
            child = optional_json(logs / f"{name}_s42.process.json", evidence)
            if child is not None:
                physics_processes[profile].append(process_record(child["pid"], ""))
    info = optional_json(directory / "server_info.json", evidence)
    if info is not None:
        assert info["port"] == port and not info["ema_weights"]
        assert info["acceptance_intervention"]["active_slot_mask"]
        assert not info["acceptance_intervention"]["exact_published_caption"]
        assert info["acceptance_intervention"]["warm_start_contract_run"] is None
        assert (
            Path(info["checkpoint_path"]).resolve()
            == (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
        )
    recorded_error = optional_json(directory / "error.json", evidence)
    complete = optional_json(directory / "complete.json", evidence)
    release = optional_json(directory / "release.json", evidence)
    released = False
    if complete is not None and release is not None:
        assert complete == dict(
            complete=True,
            profiles=[j["profile"] for j in jobs],
            wam_trials=6,
            physics_steps=3600,
            queries_expected=6 * ((600 + interval - 1) // interval),
        )
        assert model is not None and release["model_pid"] == model["pid"]
        assert release["model_start_ticks"] == model["start_ticks"] and release["model_returncode"] in (0, -15)
        assert release["last_job_returncode"] == 0 and release["model_absent"]
        assert release["only_owned_process_groups_stopped"] and not release["automatic_retry"]
        assert len(matrix_processes) == len(exits) == 2
        assert all(x["returncode"] == 0 for x in exits.values())
        assert sum(len(v) for v in physics_processes.values()) == 6
        released = (
            not coordinator["present"]
            and not model_process["present"]
            and not any(x["present"] for x in matrix_processes.values())
            and not any(x["present"] for v in physics_processes.values() for x in v)
        )
    return dict(
        interval=interval,
        physical_gpu=gpu,
        service_config=config_path(interval, 1),
        service_artifact_root_for_both_profiles=str(service_root),
        coordinator=coordinator,
        model=model_process,
        matrices=matrix_processes,
        physics_workers=physics_processes,
        recorded_execution_error=recorded_error,
        owned_service_released_normally=released and recorded_error is None,
    )


def service_artifacts(case, config, service_root, evidence):
    """Bind alpha0 copies to the actual private alpha1 service response bytes."""
    generated = []
    for index in range(
        (600 + config["max_observation_interval_steps"] - 1) // config["max_observation_interval_steps"]
    ):
        response = evidence.json(case / f"response_{index:04d}.json")
        directory = Path(response["args"]["output_dir"]).resolve()
        assert directory.parent == service_root and directory.name.startswith("generate_")
        assert evidence.json(directory / "sample_args.json") == response["args"]
        assert evidence.json(directory / "sample_outputs.json") == response
        generated.append(str(directory))
    assert len(set(generated)) == len(generated)
    return dict(
        service_config_output=str(service_root),
        complete_saved_service_args_and_response_exact=True,
        generated_request_directories=generated,
        cpu_alpha_is_applied_after_raw_response=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-verified", type=Path)
    args = parser.parse_args()
    cpu_environment()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO / "outputs"):
        raise ValueError("audit output must be a new repository outputs file")
    configs, gate, signatures, stats = preflight()
    evidence, reusable, cases = Evidence(), {}, []
    for filename, signature in signatures.items():
        evidence.check(filename, signature)
    evidence.read(__file__)
    evidence.check(VERIFIER, VERIFIER_SHA)
    evidence.check(COORDINATOR, COORDINATOR_SHA)
    evidence.check(LAUNCH, LAUNCH_SHA)
    launch = evidence.json(LAUNCH)
    assert launch["budget_git_commit"] == REGISTRATION_COMMIT
    assert launch["coordinator_source_sha256"] == COORDINATOR_SHA
    assert launch["prior_gate_sha256"] == PRIOR_GATE_SHA
    assert (
        launch["registered_trials"],
        launch["registered_physics_steps"],
        launch["registered_query_upper_bound"],
        launch["added_oracle_trials"],
    ) == (12, 7200, 678, 0)
    assert sorted(r["interval"] for r in launch["records"]) == [8, 16]
    assert hashlib.sha256(subprocess.check_output(["git", "show", f"18a5a54:{VERIFIER}"])).hexdigest() == VERIFIER_SHA
    assert (
        hashlib.sha256(subprocess.check_output(["git", "show", f"68ff90e:{COORDINATOR}"])).hexdigest()
        == COORDINATOR_SHA
    )
    if args.reuse_verified:
        previous = evidence.json(args.reuse_verified)
        assert previous["schema_version"] == SCHEMA
        assert (
            previous["source_before_after_hashes"][str(Path(__file__).resolve())]
            == evidence.hashes[str(Path(__file__).resolve())]
        )
        for filename, signature in previous["source_before_after_hashes"].items():
            evidence.check(filename, signature)
        reusable = {str(Path(c["case"]).resolve()): c for c in previous["cases"] if c["status"] == "verified"}
    fixture = evidence.json(configs[8, 1]["fixture"])
    fixtures = {r["name"]: r for r in fixture["records"]}
    observed_requests = 0
    for interval in (8, 16):
        service_root = (
            REPO
            / configs[interval, 1]["output_dir"]
            / "intervention_services"
            / str(configs[interval, 1]["checkpoints"][0]["port"])
        ).resolve()
        expected_queries = (600 + interval - 1) // interval
        for alpha in (1, 0):
            config = configs[interval, alpha]
            variant = config["variants"][0]
            profile = f"interval{interval}_alpha{alpha}"
            for name in config["records"]:
                case = Path(config["output_dir"]) / "simulation/8400_masked_position_hold" / f"{name}_s42"
                requests = len(list(case.glob("request_*.json")))
                assert requests <= expected_queries
                observed_requests += requests
                identity = dict(
                    case=str(case),
                    profile=profile,
                    interval=interval,
                    model_rotation_alpha=alpha,
                    source="wam",
                    checkpoint=CHECKPOINT,
                    variant="position_hold",
                    record=name,
                )
                result = optional_json(case / "result.json", evidence)
                if result is None:
                    cases.append(dict(identity, status="partial" if case.exists() else "missing"))
                    continue
                if result["error"] is not None:
                    cases.append(dict(identity, status="execution_error", error=result["error"]))
                    continue
                try:
                    assert requests == result["chunks"] == expected_queries
                    cached = reusable.get(str(case.resolve()))
                    if cached is not None:
                        cases.append(dict(cached, reused_from=str(args.reuse_verified.resolve())))
                        continue
                    verified = verify_case(
                        config, fixtures[name], variant, case, evidence, source="wam", checkpoint=CHECKPOINT, full=True
                    )
                    chunks = evidence.json(case / "chunks.json")
                    assert [c["step"] for c in chunks] == list(range(0, 600, interval))
                    verified.update(
                        profile=profile,
                        interval=interval,
                        model_rotation_alpha=alpha,
                        expected_queries=expected_queries,
                        shared_service_provenance=service_artifacts(case, config, service_root, evidence),
                    )
                    cases.append(verified)
                except Exception as exc:
                    cases.append(
                        dict(identity, status="audit_error", error=repr(exc), traceback=traceback.format_exc())
                    )
    assert len(cases) == 12 and observed_requests <= 678
    resources = [resource_snapshot(interval, configs, launch, evidence) for interval in (8, 16)]
    groups = []
    for interval in (8, 16):
        for alpha in (1, 0):
            profile = f"interval{interval}_alpha{alpha}"
            verified = [c for c in cases if c["profile"] == profile and c["status"] == "verified"]
            complete = len(verified) == 3
            groups.append(
                dict(
                    profile=profile,
                    interval=interval,
                    model_rotation_alpha=alpha,
                    expected_cases=3,
                    verified_cases=len(verified),
                    complete=complete,
                    stable_successes_among_verified=sum(c["outcome"]["stable"] for c in verified),
                    ever_successes_among_verified=sum(c["outcome"]["ever"] for c in verified),
                    stable_rate=sum(c["outcome"]["stable"] for c in verified) / 3 if complete else None,
                    raw_checkpoint_reference=alpha == 1,
                    component_ablation_only=alpha == 0,
                    tracking_pooled_equal_length_cases={
                        ref: {
                            metric: float(np.sqrt(np.mean([c["tracking"][ref][metric] ** 2 for c in verified])))
                            for metric in ("position_rmse_m", "so3_rmse_rad")
                        }
                        for ref in verified[0]["tracking"]
                    }
                    if complete
                    else None,
                )
            )
    all_verified = all(c["status"] == "verified" for c in cases)
    evidence.recheck()
    value = dict(
        schema_version=SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="only12_frozen010_8400_short_replan_wam_development_trials",
        complete=all_verified and all(r["owned_service_released_normally"] for r in resources),
        all12_cases_independently_verified=all_verified,
        registered_budget=dict(wam_trials=12, physics_steps=7200, model_queries_upper_bound=678),
        observed_request_files=observed_requests,
        verified_physics_steps=sum(c.get("steps", 0) for c in cases if c["status"] == "verified"),
        reused_verified_cases=sum("reused_from" in c for c in cases),
        cases=cases,
        groups=groups,
        resources=resources,
        inherited_verifier=dict(
            path=VERIFIER,
            sha256=VERIFIER_SHA,
            direct_function_reuse=True,
            first_query_source="same8400 old intent_hold full raw/normalized/image/initial/anchor",
            inherited_first_query003_regression_field_is_not_010_registration=True,
        ),
        prior_gate_reference=dict(
            path=PRIOR_GATE,
            sha256=PRIOR_GATE_SHA,
            all_prior_inputs_rehashed=True,
            weight_stats=stats,
            added_oracle_trials=0,
            qualification_physics_reexecuted=0,
            oracle_excluded_from_wam_denominator=True,
            prior_stages=sorted(gate["audits"]),
        ),
        source_before_after_hashes=evidence.hashes,
        warnings=[
            "010 has exactly12 WAM development trials; existing oracle qualifications are only referenced, not rerun or counted.",
            "Interval8/16 discards generated suffixes; the frozen per-step verifier checks causal timing and separate raw/effective/admitted/nominal clocks.",
            "Both alpha profiles share an alpha1-configured private service; every saved server response/args is bound to its actual alpha1 service directory.",
            "Alpha0 changes local rotations and integrated world position; it is component diagnosis, not raw full-checkpoint success.",
            "Partial cases or pending coordinator/model/CPU-worker release prevent completion; no partial winner or fresh acceptance claim.",
            "Native success/contact/object state are post-step metrics only. No new model request, physics step, real hardware or pretraining-unseen claim.",
        ],
    )
    write_new(output, value)
    print(
        json.dumps(
            dict(
                output=str(output),
                sha256=sha256(output),
                complete=value["complete"],
                verified_cases=sum(c["status"] == "verified" for c in cases),
                observed_request_files=observed_requests,
            )
        )
    )


if __name__ == "__main__":
    main()
