"""Explicit020 registered-physics supplement; original strict False is retained.

The preregistered1800 common row tokens and physical gates remain exact. This
sidecar allows only the225 separately disclosed chunk metadata type pairs
float1.0 versus int1. No source/config/log is rewritten and no physics runs.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_position import Evidence
from audit_wam_temporal_qualification import BUDGET, common_row_identity
from run_wam_short_replan import cpu_environment, process_record, write_new

from s2a.artifacts import REPO, sha256

STRICT = (
    REPO / "outputs/wam_stability_20261021/temporal_equal_qualification/qualification_independent_verification.json"
)
STRICT_SHA = "44e989bb78dc4251831190ae0f5ece965492aef91d4eced9d85f76523ed7bec6"
AUDITOR = REPO / "scripts/audit_wam_temporal_qualification.py"
AUDITOR_SHA = "5b46fa14800e987508604e1b50f810ec73b5a6333fe5f40aa91caad4d34a190c"
CHECKER = REPO / "scripts/wam_temporal_position_checker.py"
CHECKER_SHA = "5725fe14d4e2527e2a3f28e5acd9da3c043b614258a56363f003bc72f951a57d"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def chunk_type_supplement(current, previous):
    """No numeric normalization: identify the sole allowed type/value pair."""
    assert len(current) == len(previous) == 75
    differences = []
    for index, (new, old) in enumerate(zip(current, previous, strict=True)):
        assert set(new) == set(old)
        assert "generation_s" in new and "model_rotation_alpha" in new
        assert type(new["model_rotation_alpha"]) is float and new["model_rotation_alpha"] == 1.0
        assert type(old["model_rotation_alpha"]) is int and old["model_rotation_alpha"] == 1
        new_fields = {key: value for key, value in new.items() if key not in ("generation_s", "model_rotation_alpha")}
        old_fields = {key: value for key, value in old.items() if key not in ("generation_s", "model_rotation_alpha")}
        assert canonical(new_fields) == canonical(old_fields), index
        differences.append(
            dict(
                chunk=index,
                physical_query_step=new["step"],
                path=f"chunks[{index}].model_rotation_alpha",
                current_json_type="float",
                current_value=1.0,
                previous_json_type="int",
                previous_value=1,
                all_other_nonwall_fields_canonical_exact=True,
            )
        )
    return differences


def rows(path, evidence):
    return [json.loads(line) for line in evidence.read(path).splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert Path.cwd().resolve() == REPO
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(STRICT, STRICT_SHA)
    evidence.check(AUDITOR, AUDITOR_SHA)
    evidence.check(CHECKER, CHECKER_SHA)
    original = evidence.json(STRICT)
    assert original["schema_version"] == "s2a.temporal-equal-oracle-qualification.v1"
    assert original["complete"] and original["qualified"] is False and original["errors"] == []
    assert original["normal_process_release"] and original["all1800_common_step_fields_exact_to_007"]
    assert original["verified_oracle_cases"] == 3 and original["verified_physics_log_endpoints"] == 1800
    assert original["registered_budget"] == BUDGET and original["source_oracle_only"]
    assert not original["oracle_is_wam_success_evidence"]
    for path, signature in original["input_sha256"].items():
        evidence.check(path, signature)
    launch = evidence.json(original["launch_path"])
    evidence.check(original["launch_path"], original["launch_sha256"])
    evidence.check(launch["config_path"], launch["config_sha256"])
    config = evidence.json(launch["config_path"])
    assert type(config["variants"][0]["model_rotation_alpha"]) is float
    assert config["variants"][0]["model_rotation_alpha"] == 1.0
    statuses = [
        dict(role="matrix", **process_record(launch["matrix_pid"], "", start_ticks=launch["matrix_start_ticks"]))
    ]
    root = REPO / config["output_dir"]
    logs = root / "simulation/logs/oracle_temporal_equal"
    assert evidence.json(launch["matrix_exit_path"]) == dict(
        pid=launch["matrix_pid"], start_ticks=launch["matrix_start_ticks"], returncode=0
    )
    cases, all_differences = [], []
    assert [case["record"] for case in original["cases"]] == [f"task{i}_init0" for i in range(3)]
    for old_case in original["cases"]:
        assert old_case["qualified"] is False
        assert old_case["all75_common_chunk_fields_exact_excluding_wall_generation_time"] is False
        actual = old_case["result"]
        assert actual["status"] == "verified" and actual["steps"] == 600 and actual["chunks"] == 75
        assert actual["outcome"]["stable"] and not actual["raw_checkpoint_evidence"]
        assert actual["tracking"]["admitted"]["position_rmse_m"] <= 0.01
        assert actual["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
        assert actual["independent_temporal_ensemble_replay"]["passed"]
        assert actual["independent_temporal_ensemble_replay"]["verified_endpoints"] == 600
        assert actual["independent_temporal_ensemble_replay"]["identity_endpoint_count"] == 600
        folder, previous = Path(actual["case"]), Path(old_case["old007_case"])
        current_rows, old_rows = rows(folder / "steps.jsonl", evidence), rows(previous / "steps.jsonl", evidence)
        common = common_row_identity(current_rows, old_rows)
        assert common["all600_common_step_fields_canonical_exact"]
        assert common == old_case["common_rows"]
        metadata = chunk_type_supplement(evidence.json(folder / "chunks.json"), evidence.json(previous / "chunks.json"))
        old_config = evidence.json(previous / "manifest.json")["experiment"]
        old_variant = next(item for item in old_config["variants"] if item["name"] == "position_hold")
        assert type(old_variant["model_rotation_alpha"]) is int and old_variant["model_rotation_alpha"] == 1
        worker = evidence.json(logs / f"{old_case['record']}_s42.process.json")
        statuses.append(dict(role="physics", **process_record(worker["pid"], "", start_ticks=worker["start_ticks"])))
        all_differences.extend(dict(record=old_case["record"], **item) for item in metadata)
        cases.append(
            dict(
                record=old_case["record"],
                registered_physical_qualified=True,
                original_strict_case_qualified=False,
                all_other_chunk_tokens_exact=True,
                strict_chunk_token_identity_passed=False,
                common_rows=common,
                alpha_metadata_differences=metadata,
                result=actual,
                old007_case=str(previous),
                original_case_reference=dict(
                    artifact_path=str(STRICT), artifact_sha256=STRICT_SHA, record=old_case["record"]
                ),
                oracle_excluded_from_wam_denominator=True,
            )
        )
    assert len(all_differences) == 225 and len(statuses) == 4 and not any(item["present"] for item in statuses)
    evidence.recheck()
    output = dict(
        schema_version="s2a.temporal-equal-oracle-qualification.v1",
        audit_variant="registered_physics_with_alpha_metadata_supplement",
        qualification_scope="registered_physical_criteria",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        qualified=True,
        original_strict_qualified=False,
        additional_chunk_token_exact=False,
        original_strict_artifact=dict(path=str(STRICT), sha256=STRICT_SHA, qualified=False),
        verified_oracle_cases=3,
        verified_physics_log_endpoints=1800,
        normal_process_release=True,
        all1800_common_step_fields_exact_to_007=True,
        all225_other_chunk_fields_canonical_exact=True,
        registered_budget=BUDGET,
        cases=cases,
        alpha_metadata_differences=all_differences,
        errors=[],
        processes=statuses,
        source_proof=original["source_proof"],
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        oracle_is_wam_success_evidence=False,
        source_oracle_only=True,
        no_original_source_configuration_log_or_strict_result_modified=True,
        no_numeric_normalization_or_new_tolerance=True,
        launch_path=original["launch_path"],
        launch_sha256=original["launch_sha256"],
        limitations=[
            "qualified refers only to the preregistered physical qualification; the original extra full-chunk token test remains false.",
            "The sole explicit exception is chunk model_rotation_alpha float1.0 versus int1; every other nonwall chunk token and every old physical row remains canonical exact.",
            "No WAM task success, new physical execution, fit or automatic deployment is established by this read-only supplement.",
        ],
    )
    write_new(args.output, output)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                qualified=True,
                original_strict_qualified=False,
                metadata_type_pairs=len(all_differences),
            )
        )
    )


if __name__ == "__main__":
    main()
