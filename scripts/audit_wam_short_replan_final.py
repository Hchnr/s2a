"""Independent010 final audit with one explicit unused-checkpoint exception.

The frozen coordinator/checker and original error/release stay unchanged.
Only15400 historical weight availability may fail; all execution input SHA,
actual8400 metadata, case mechanics, queries and owned releases stay strict.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_earlier_position import verify_case
from audit_wam_position import Evidence
from audit_wam_short_replan import (
    COORDINATOR,
    COORDINATOR_SHA,
    LAUNCH,
    LAUNCH_SHA,
    VERIFIER,
    VERIFIER_SHA,
    resource_snapshot,
    service_artifacts,
)
from audit_wam_short_replan_outcomes import describe
from run_wam_short_replan import (
    CHECKPOINT,
    PRIOR_GATE,
    PRIOR_GATE_SHA,
    REGISTRATION_COMMIT,
    configurations,
    cpu_environment,
    write_new,
)

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
FAILURE = ROOT / "short_replan_audit/final_unrelated_weight_stat_failure.json"
FAILURE_SHA = "73506055b250979dbdf5a48bb13cddc7d48c0de7046d652edefc3766328d4310"
PREVIOUS = ROOT / "short_replan_audit/progress_11.json"
PREVIOUS_SHA = "3af07f8811b523b0f635d7ac6c227d6a1b30defb4fb7fdebdc61affd60949dab"
ORIGINAL_AUDITOR = "scripts/audit_wam_short_replan.py"
ORIGINAL_AUDITOR_SHA = "03e19eb1b5c4cf8bb591ecf224a80966d355d11b1352a3b687ba9a65a13a5177"
OUTCOMES = "scripts/audit_wam_short_replan_outcomes.py"
OUTCOMES_SHA = "8a2e4c68442c6dce2b2c8c77ec56bc943c353c0038560e0900fbe5b756d90cfa"


def weight_availability(gate, failure, evidence):
    checked = []
    for item in gate["weight_stat_checks"]:
        recorded = next(c for c in failure["checkpoint_stat_checks"] if c["name"] == item["name"])
        evidence.check(item["fingerprint_path"], recorded["fingerprint_sha256"])
        doc = evidence.json(item["fingerprint_path"])
        weights = next(c for c in doc["checkpoints"] if c["name"] == item["name"]) if "checkpoints" in doc else doc
        assert len(weights["files"]) == item["files_stat_rechecked"]
        missing = []
        for name, expected in weights["files"].items():
            assert len(expected["sha256"]) == 64
            try:
                current = Path(name).stat()
            except FileNotFoundError:
                missing.append(name)
                continue
            assert (current.st_size, current.st_mtime_ns) == (expected["bytes"], expected["mtime_ns"]), name
        if item["name"] == "15400_masked":
            assert missing and set(missing) == set(recorded["missing_files"])
        else:
            assert not missing
        checked.append(
            dict(
                name=item["name"],
                registered_files=len(weights["files"]),
                missing_files=missing,
                present_files_metadata_unchanged=len(weights["files"]) - len(missing),
                actual010_model=item["name"] == "8400_masked",
                weight_bytes_rehashed=False,
            )
        )
    assert {c["name"] for c in checked} == {"8400_masked", "8800_masked", "15400_masked", "15600_masked"}
    assert next(c for c in checked if c["actual010_model"])["present_files_metadata_unchanged"] == 163
    return checked


def preregistration(evidence):
    evidence.check(PRIOR_GATE, PRIOR_GATE_SHA)
    gate = evidence.json(PRIOR_GATE)
    assert gate["complete_prior_gates_passed"] and gate["all_inputs_before_after_unchanged"] and not gate["failures"]
    assert (gate["registered_trials"], gate["registered_physics_steps"], gate["registered_query_upper_bound"]) == (
        12,
        7200,
        678,
    )
    assert gate["budget_commit"] == REGISTRATION_COMMIT
    for name, digest in gate["input_sha256"].items():
        evidence.check(name, digest)
    configs = configurations(evidence.hashes)
    for item in gate["normal_service_release"]:
        directory = Path(item["root"])
        assert not (directory / "error.json").exists()
        assert evidence.json(directory / "complete.json") == item["complete"]
        assert evidence.json(directory / "release.json") == item["release"]
    assert all(not Path(f"/proc/{p['pid']}").exists() for p in gate["old_registered_processes"])
    evidence.check(COORDINATOR, COORDINATOR_SHA)
    evidence.check(VERIFIER, VERIFIER_SHA)
    evidence.check(ORIGINAL_AUDITOR, ORIGINAL_AUDITOR_SHA)
    evidence.check(OUTCOMES, OUTCOMES_SHA)
    evidence.check(LAUNCH, LAUNCH_SHA)
    launch = evidence.json(LAUNCH)
    assert launch["budget_git_commit"] == REGISTRATION_COMMIT and launch["coordinator_source_sha256"] == COORDINATOR_SHA
    assert launch["prior_gate_sha256"] == PRIOR_GATE_SHA
    assert (
        launch["registered_trials"],
        launch["registered_physics_steps"],
        launch["registered_query_upper_bound"],
        launch["added_oracle_trials"],
    ) == (12, 7200, 678, 0)
    evidence.check(FAILURE, FAILURE_SHA)
    failure = evidence.json(FAILURE)
    assert failure["actual8400_all163_metadata_unchanged"]
    assert not failure["prior_gate_input_sha_missing"] and not failure["prior_gate_input_sha_changed"]
    assert not failure["external_change_cause_established"] and not failure["actual_model_or_physics_reexecuted"]
    for name, digest in failure["source_hashes"].items():
        evidence.check(name, digest)
    trace = evidence.read(ROOT / "short_replan_coordinators/interval8.log").decode()
    assert "weight_stats(gate)" in trace and "FileNotFoundError:" in trace
    assert "iter_000015400/.save_complete" in trace and "stat = Path(filename).stat()" in trace
    return configs, gate, launch, weight_availability(gate, failure, evidence)


def release_and_budget(interval, configs, launch, cases, evidence):
    snapshot = resource_snapshot(interval, configs, launch, evidence)
    processes = [snapshot["coordinator"], snapshot["model"], *snapshot["matrices"].values()]
    processes += [p for values in snapshot["physics_workers"].values() for p in values]
    assert len(snapshot["matrices"]) == 2 and sum(map(len, snapshot["physics_workers"].values())) == 6
    assert all(not p["present"] for p in processes)
    config = configs[interval, 1]
    root = REPO / config["output_dir"]
    directory = root / "orchestration" / f"short_replan_interval{interval}"
    release = evidence.json(directory / "release.json")
    model = evidence.json(directory / "model_process.json")
    assert release["model_pid"] == model["pid"] and release["model_start_ticks"] == model["start_ticks"]
    assert release["model_returncode"] in (0, -15) and release["last_job_returncode"] == 0
    assert release["model_absent"] and release["only_owned_process_groups_stopped"] and not release["automatic_retry"]
    for alpha in (1, 0):
        profile = f"interval{interval}_alpha{alpha}"
        process = evidence.json(directory / f"{profile}_process.json")
        assert evidence.json(directory / f"{profile}_exit.json") == dict(pid=process["pid"], returncode=0)
        config_alpha = configs[interval, alpha]
        matrix = evidence.json(
            REPO / config_alpha["output_dir"] / "simulation/position_matrix_8400_masked_position_hold.json"
        )
        assert matrix["complete"] and len(matrix["cases"]) == 3 and all(c["returncode"] == 0 for c in matrix["cases"])
    if interval == 8:
        assert not (directory / "complete.json").exists()
        assert evidence.json(directory / "error.json") == dict(
            error="FileNotFoundError(2, 'No such file or directory')",
            scientific_failure_inferred=False,
            automatic_retry=False,
        )
        assert not snapshot["owned_service_released_normally"]
    else:
        assert snapshot["owned_service_released_normally"] and snapshot["recorded_execution_error"] is None
    selected = [c for c in cases if c["interval"] == interval]
    assert len(selected) == 6
    expected = [
        Path(p).resolve() for c in selected for p in c["shared_service_provenance"]["generated_request_directories"]
    ]
    service = root / "intervention_services" / str(config["checkpoints"][0]["port"])
    actual = {p.resolve() for p in service.glob("generate_*") if p.is_dir()}
    count = 450 if interval == 8 else 228
    assert len(expected) == len(set(expected)) == count and set(expected) == actual
    log = evidence.read(directory / "service.log").decode()
    posts = [
        (path, int(status))
        for method, path, status in re.findall(r'"([A-Z]+) ([^ ]+) HTTP/1\.[01]" (\d{3})', log)
        if method == "POST"
    ]
    assert len(posts) == len(re.findall(r'"POST ', log)) == count and all(p == ("/generate", 200) for p in posts)
    assert re.findall(r"Started server process \[(\d+)\]", log) == [str(model["pid"])]
    return dict(
        interval=interval,
        frozen010_resource_snapshot=snapshot,
        all_owned_processes_absent=True,
        independent_external_release_verified=True,
        model_returncode=release["model_returncode"],
        original_coordinator_final_gate_success=interval == 16,
        exact_model_queries=count,
        all_generate_directories_equal_six_case_provenance=True,
        no_extra_http_post=True,
        original_error_retained=interval == 8,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    configs, gate, launch, stats = preregistration(evidence)
    evidence.check(PREVIOUS, PREVIOUS_SHA)
    prior = evidence.json(PREVIOUS)
    assert prior["schema_version"] == "s2a.short-replan-readonly-audit.v1"
    assert prior["source_before_after_hashes"][str(REPO / ORIGINAL_AUDITOR)] == ORIGINAL_AUDITOR_SHA
    for name, digest in prior["source_before_after_hashes"].items():
        evidence.check(name, digest)
    reusable = {str(Path(c["case"]).resolve()): c for c in prior["cases"] if c["status"] == "verified"}
    assert len(reusable) == 11
    fixture = {r["name"]: r for r in evidence.json(configs[8, 1]["fixture"])["records"]}
    cases = []
    for interval in (8, 16):
        for alpha in (1, 0):
            config = configs[interval, alpha]
            for name in config["records"]:
                case = REPO / config["output_dir"] / "simulation/8400_masked_position_hold" / f"{name}_s42"
                result = evidence.json(case / "result.json")
                expected = 75 if interval == 8 else 38
                assert result["complete"] and result["error"] is None and result["steps"] == 600
                assert result["chunks"] == expected == len(list(case.glob("request_*.json")))
                cached = reusable.get(str(case.resolve()))
                if cached is not None:
                    verified = dict(cached, reused_from=str(PREVIOUS))
                else:
                    verified = verify_case(
                        config,
                        fixture[name],
                        config["variants"][0],
                        case,
                        evidence,
                        source="wam",
                        checkpoint=CHECKPOINT,
                        full=True,
                    )
                    chunks = evidence.json(case / "chunks.json")
                    assert [c["step"] for c in chunks] == list(range(0, 600, interval))
                    service_root = (
                        REPO
                        / configs[interval, 1]["output_dir"]
                        / "intervention_services"
                        / str(config["checkpoints"][0]["port"])
                    ).resolve()
                    verified.update(
                        profile=f"interval{interval}_alpha{alpha}",
                        interval=interval,
                        model_rotation_alpha=alpha,
                        expected_queries=expected,
                        shared_service_provenance=service_artifacts(case, config, service_root, evidence),
                    )
                assert (
                    verified["record"],
                    verified["interval"],
                    verified["model_rotation_alpha"],
                    verified["status"],
                ) == (name, interval, alpha, "verified")
                cases.append(verified)
    assert len(cases) == 12 and sum(c["steps"] for c in cases) == 7200 and sum(c["chunks"] for c in cases) == 678
    resources = [release_and_budget(i, configs, launch, cases, evidence) for i in (8, 16)]
    diagnostics = [describe(case, evidence) for case in cases]
    groups = []
    for interval in (8, 16):
        for alpha in (1, 0):
            selected = [c for c in cases if c["interval"] == interval and c["model_rotation_alpha"] == alpha]
            assert len(selected) == 3
            groups.append(
                dict(
                    profile=f"interval{interval}_alpha{alpha}",
                    stable_successes=sum(c["outcome"]["stable"] for c in selected),
                    ever_successes=sum(c["outcome"]["ever"] for c in selected),
                    expected_cases=3,
                    stable_rate=sum(c["outcome"]["stable"] for c in selected) / 3,
                    raw_checkpoint_reference=alpha == 1,
                    component_ablation_only=alpha == 0,
                )
            )
    evidence.recheck()
    value = dict(
        schema_version="s2a.short-replan-final-exception-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="new_readonly_final_case_and_request_verification_with_explicit_unused15400_availability_exception",
        complete=True,
        all12_cases_independently_verified=True,
        verified_physics_steps=7200,
        observed_request_files=678,
        new_readonly_audit_complete=True,
        original_coordinator_final_gate_success=False,
        original_missing_complete_json_not_recreated=True,
        all_owned_processes_released=True,
        unused_checkpoint15400_disappearance_cause="unknown",
        registered_budget=dict(wam_trials=12, physics_steps=7200, model_queries_upper_bound=678),
        cases=cases,
        groups=groups,
        resources=resources,
        outcome_diagnostics=diagnostics,
        reused_verified_cases=11,
        new_full600_step_case_replays=1,
        weight_metadata_checks=stats,
        all3477_prior_input_sha_unchanged=True,
        actual8400_all163_metadata_unchanged=True,
        original_failure_evidence=dict(path=str(FAILURE), sha256=FAILURE_SHA),
        new_model_requests=0,
        new_physics_steps=0,
        source_before_after_hashes=evidence.hashes,
        limitations=[
            "complete=true describes this new read-only audit, not the original failed coordinator final gate.",
            "The original interval8 error/release and missing complete stamp remain untouched; no inference/physics retry.",
            "Only unused15400 historical weight availability is excepted; every actual8400 metadata and execution source SHA stays strict.",
            "Frozen006 full mechanics plus frozen010 source/service helpers are reused; no dynamic patch or global mutation.",
            "All12 development cases, all678 successful POSTs and exact generation directories are included; oracles/fresh are not added.",
            "Object-center diagnostics cannot uniquely attribute failures to reference or controller.",
        ],
    )
    write_new(args.output, value)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                independent_audit_complete=True,
                verified_cases=12,
                physical_steps=7200,
                queries=678,
                original_coordinator_final_gate_success=False,
            )
        )
    )


if __name__ == "__main__":
    main()
