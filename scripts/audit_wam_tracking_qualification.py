"""Read-only016 qualification: authenticate all three oracle physical logs.

The failed preparation attempt remains separate. This checker launches no
model/simulator and compares the corrected run to the frozen40-step oracle.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence, close
from run_wam_short_replan import cpu_environment, process_record, write_new
from wam_position_acceptance_checker import verify_case as verify_original_case
from wam_tracking_replan_checker import source_proof, verify_case

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
CONFIG = "configs/tracking_replan_qualification_20261021_v2.json"
LAUNCH = ROOT / "tracking_replan_qualification_v2/orchestration/independent_launch.json"
LAUNCH_SHA = "33c4989e829cc339c790767e6bbb77fdf0b4b5a58c8b6c4c0f42ceff3d782ebe"
PREPARATION_FAILURE = ROOT / "tracking_replan_qualification/orchestration/pre_environment_failure.json"
PREPARATION_FAILURE_SHA = "9bab897ddf7d1645fd5a3697de7977f77c68980d6a00f8c0f2805cf17b5baea1"
PREVIOUS = ROOT / "position_full_complete_20261021.json"
PREVIOUS_SHA = "fe714dad35dad3d6d3ab933be00e166c94c0c54cff6611730c439e782759299e"


def lines(path, evidence):
    return [json.loads(line) for line in evidence.read(path).splitlines()]


def numeric_compare(current, previous, *, tolerance=1e-10):
    """Require identical structure; quantify every numeric leaf difference."""
    if isinstance(previous, dict):
        assert isinstance(current, dict) and set(current) == set(previous)
        return max((numeric_compare(current[k], previous[k], tolerance=tolerance) for k in previous), default=0.0)
    if isinstance(previous, list):
        assert isinstance(current, list) and len(current) == len(previous)
        return max(
            (numeric_compare(a, b, tolerance=tolerance) for a, b in zip(current, previous, strict=True)), default=0.0
        )
    if isinstance(previous, (bool, str)) or previous is None:
        assert type(current) is type(previous) and current == previous
        return 0.0
    assert isinstance(current, (int, float)) and not isinstance(current, bool)
    assert np.isfinite(current) and np.isfinite(previous)
    difference = abs(current - previous)
    assert difference <= tolerance, (current, previous, difference)
    return difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    proof = source_proof(evidence)
    evidence.check(LAUNCH, LAUNCH_SHA)
    launch = evidence.json(LAUNCH)
    assert launch["budget_git_commit"] == "9492840" and launch["runtime_git_commit"] == "68fa982"
    assert launch["budget"] == dict(oracle_trials=3, physical_steps=1800, model_requests=0, fresh_initial_states=0)
    assert launch["cuda_visible_devices"] == launch["ld_library_path"] == "" and launch["cpu_workers"] == 2
    assert launch["source"] == "oracle" and launch["checkpoint"] is None and not launch["model_started"]
    assert launch["experiment_seed"] == launch["sampling_seed"] == 42 and not launch["automatic_retry"]
    evidence.check(launch["source_review_path"], launch["source_review_sha256"])
    review = evidence.json(launch["source_review_path"])
    assert review["review_passed"] and review["independent_v2_dry_run_passed"]
    for path, digest in launch["execution_source_sha256"].items():
        evidence.check(path, digest)
    evidence.check(CONFIG, launch["config_sha256"])
    assert (
        hashlib.sha256(subprocess.check_output(["git", "show", f"68fa982:{CONFIG}"])).hexdigest()
        == launch["config_sha256"]
    )
    config = evidence.json(CONFIG)
    assert config["seed"] == 42 and config["sampling_seeds"] == [42]
    assert config["records"] == ["task0_init0", "task1_init0", "task2_init0"]
    assert config["online_max_steps"] == 600 and len(config["variants"]) == 1
    evidence.check(config["fixture"], config["fixture_sha256"])
    records = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
    variant = config["variants"][0]
    evidence.check(PREPARATION_FAILURE, PREPARATION_FAILURE_SHA)
    failure = evidence.json(PREPARATION_FAILURE)
    assert failure["failure_before_make_environment"] and failure["physical_steps"] == failure["model_requests"] == 0
    assert not failure["matrix_present"] and not any(p["present"] for p in failure["workers"])
    for path, digest in failure["input_sha256"].items():
        evidence.check(path, digest)
    evidence.check(PREVIOUS, PREVIOUS_SHA)
    previous = evidence.json(PREVIOUS)
    assert previous["whole_registered_protocol_complete"] and len(previous["cases"]) == 33
    assert previous["registered_oracle_cases"] == 9 and previous["registered_wam_cases"] == 24
    for path, digest in previous["source_before_after_hashes"].items():
        evidence.check(path, digest)
    output = REPO / config["output_dir"]
    matrix = evidence.json(output / "simulation/tracking_matrix_oracle_position_hold.json")
    assert matrix["complete"] and len(matrix["cases"]) == 3
    assert all(c["returncode"] == 0 for c in matrix["cases"])
    assert sorted((c["record"], c["sampling_seed"]) for c in matrix["cases"]) == sorted(
        (r, 42) for r in config["records"]
    )
    snapshot_path = output / "simulation/logs/oracle_position_hold/execution_snapshot.json"
    evidence.check(snapshot_path, matrix["execution_snapshot_sha256"])
    snapshot = evidence.json(snapshot_path)
    assert snapshot["identity"] == ["oracle", "position_hold", None] and snapshot["cpu_workers"] == 2
    assert snapshot["coordinator_pid"] == launch["matrix_pid"] and snapshot["config_sha256"] == launch["config_sha256"]
    processes = dict(
        matrix=process_record(launch["matrix_pid"], "", start_ticks=launch["matrix_start_ticks"]), workers=[]
    )
    verified = []
    for name in config["records"]:
        case = output / "simulation/oracle_position_hold" / f"{name}_s42"
        manifest = evidence.json(case / "manifest.json")
        assert manifest["record"] == records[name]
        assert not list(case.glob("request_*.json")) and not list(case.glob("response_*.json"))
        actual = verify_case(
            config,
            records[name],
            variant,
            case,
            evidence,
            source="oracle",
            checkpoint=None,
            full=True,
            sampling_seed=42,
            first_query_validator=None,
        )
        assert actual["outcome"]["stable"] and not actual["raw_checkpoint_evidence"]
        assert actual["tracking"]["admitted"]["position_rmse_m"] <= 0.01
        assert actual["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
        assert actual["tracking_replan_replay"]["query_reason_counts"]["tracking_limit"] == 0
        old = next(
            c
            for c in previous["cases"]
            if c["source"] == "oracle" and c["record"] == name and c["variant"] == "position_hold"
        )
        old_case = Path(old["case"])
        old_manifest = evidence.json(old_case / "manifest.json")
        old_config = old_manifest["experiment"]
        old_result = verify_original_case(
            old_config,
            old_manifest["record"],
            old_manifest["variant"],
            old_case,
            evidence,
            source="oracle",
            checkpoint=None,
            full=True,
            sampling_seed=42,
            first_query_validator=None,
        )
        assert old_result["outcome"] == old["outcome"]
        assert actual["chunks"] == old_result["chunks"] == 15
        current_rows, old_rows = lines(case / "steps.jsonl", evidence), lines(old_case / "steps.jsonl", evidence)
        shared_rows = [
            {k: v for k, v in row.items() if k not in {"query_decision", "tracking_replan"}} for row in current_rows
        ]
        maximum_step_difference = numeric_compare(shared_rows, old_rows)
        chunks, old_chunks = evidence.json(case / "chunks.json"), evidence.json(old_case / "chunks.json")
        shared_chunks = [{k: v for k, v in c.items() if k not in {"query_decision", "generation_s"}} for c in chunks]
        comparable_old_chunks = [{k: v for k, v in c.items() if k != "generation_s"} for c in old_chunks]
        maximum_chunk_difference = numeric_compare(shared_chunks, comparable_old_chunks)
        close(actual["tracking"]["admitted"]["position_rmse_m"], old_result["tracking"]["admitted"]["position_rmse_m"])
        close(actual["tracking"]["admitted"]["so3_rmse_rad"], old_result["tracking"]["admitted"]["so3_rmse_rad"])
        worker = evidence.json(output / "simulation/logs/oracle_position_hold" / f"{name}_s42.process.json")
        assert worker["pid"] == next(c["pid"] for c in matrix["cases"] if c["record"] == name)
        assert (
            "--source" in worker["command"] and worker["command"][worker["command"].index("--source") + 1] == "oracle"
        )
        processes["workers"].append(process_record(worker["pid"], ""))
        verified.append(
            dict(
                record=name,
                qualified=True,
                result=actual,
                old40_source_case=str(old_case),
                all600_shared_step_fields_exact=shared_rows == old_rows,
                all15_shared_chunk_fields_exact=shared_chunks == comparable_old_chunks,
                maximum_shared_step_absolute_difference=maximum_step_difference,
                maximum_shared_chunk_absolute_difference=maximum_chunk_difference,
                unchanged_physics_comparison_tolerance=1e-10,
                oracle_excluded_from_wam_denominator=True,
            )
        )
    assert not processes["matrix"]["present"] and not any(p["present"] for p in processes["workers"])
    evidence.recheck()
    result = dict(
        schema_version="s2a.tracking-replan-readonly-qualification.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        qualification_passed=True,
        all_cases_complete=True,
        oracle_trials=3,
        physical_steps=1800,
        complete=True,
        cases=verified,
        registered_budget=launch["budget"],
        normal_process_release=dict(
            all_registered_processes_absent=True,
            all_three_workers_returncode_zero=True,
            matrix_complete=True,
            processes=processes,
        ),
        source_proof=proof,
        preparation_failure_retained=str(PREPARATION_FAILURE),
        new_model_requests_by_auditor=0,
        new_physics_steps_by_auditor=0,
        oracle_wam_success=False,
        fresh_initial_states=0,
        input_sha256=evidence.hashes,
        limitations=[
            "Only three old-scene correct-reference trials qualify this new causal timing executor.",
            "No WAM output or fresh initial-state performance is certified.",
            "The original missing-source preparation attempt executed zero environments/physics; evidence is retained.",
            "Mechanical reconstruction and all600 causal decisions are replayed without importing production timing policy.",
        ],
    )
    write_new(args.output, result)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                qualification_passed=True,
                oracle_trials=3,
                physical_steps=1800,
            )
        )
    )


if __name__ == "__main__":
    main()
