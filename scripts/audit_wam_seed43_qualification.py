"""Read-only015: authenticate three real seed43 oracle logs and pair with007.

No model/simulator is launched. This qualifies execution-log seed handling,
without claiming a seed43 WAM request/response or any fresh performance.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, process_record, write_new
from wam_position_acceptance_checker import source_proof, verify_case

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
CONFIG = "configs/seed43_execution_qualification_20261021.json"
LAUNCH = ROOT / "seed43_execution_qualification/orchestration/root_launch.json"
LAUNCH_SHA = "04a756adfd1761f76bace8a239821a4c7d8785b26b34d56f661cb379269e9340"
QUALIFICATION = ROOT / "short_replan_qualification/qualification_independent_verification.json"
QUALIFICATION_SHA = "64f0eda45d3e5495dc12124795047beade66c6a3489c6b1f8061a637af9dc4ad"
REGRESSION = ROOT / "seed_aware_checker/seed42_regression.json"
REGRESSION_SHA = "b90d0915f3880fe71ac8d4f3b79afcae18720ca569d270849ab5158712d85f7e"


def rows(path, evidence):
    return [json.loads(line) for line in evidence.read(path).splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    proof = source_proof(evidence)
    evidence.check(REGRESSION, REGRESSION_SHA)
    regression = evidence.json(REGRESSION)
    assert regression["complete"] and regression["all_case_fields_exact"] and regression["old_log_cases_verified"] == 27
    assert proof["new_checker_sha256"] == regression["source_proof"]["new_checker_sha256"]
    evidence.check(LAUNCH, LAUNCH_SHA)
    launch = evidence.json(LAUNCH)
    assert launch["budget_git_commit"] == "a1c55a3" and launch["experiment_seed"] == 42
    assert launch["environment_log_sampling_seed"] == 43
    assert launch["budget"] == dict(oracle_trials=3, physics_steps=1800, model_requests=0, fresh_initial_states=0)
    assert launch["cuda_visible_devices"] == launch["ld_library_path"] == ""
    for path, digest in launch["execution_source_sha256"].items():
        evidence.check(path, digest)
    evidence.check(CONFIG, launch["config_sha256"])
    frozen = subprocess.check_output(["git", "show", f"a1c55a3:{CONFIG}"])
    assert hashlib.sha256(frozen).hexdigest() == launch["config_sha256"]
    config = evidence.json(CONFIG)
    assert config["seed"] == 42 and config["sampling_seeds"] == [43]
    assert config["records"] == ["task0_init0", "task1_init0", "task2_init0"]
    assert config["fixture"] == "outputs/wam_checkpoint_acceptance_20261019v2/simulation/fixtures.json"
    assert config["max_observation_interval_steps"] == 8 and len(config["variants"]) == 1
    variant = config["variants"][0]
    assert variant == dict(
        name="position_hold", rotation="response", gripper="position_hold", governor=False, model_rotation_alpha=1
    )
    evidence.check(QUALIFICATION, QUALIFICATION_SHA)
    old = evidence.json(QUALIFICATION)
    assert (
        old["qualified"] and old["complete"] and old["verified_cases"] == 12 and old["verified_physics_steps"] == 7200
    )
    for path, digest in old["input_sha256"].items():
        evidence.check(path, digest)
    output = REPO / config["output_dir"]
    matrix = evidence.json(output / "simulation/position_matrix_oracle_position_hold.json")
    assert matrix["complete"] and len(matrix["cases"]) == 3
    assert all(c["returncode"] == 0 for c in matrix["cases"])
    assert sorted(c["record"] for c in matrix["cases"]) == sorted(config["records"])
    processes = dict(
        matrix=process_record(launch["matrix_pid"], "", start_ticks=int(launch["matrix_start_ticks"])), workers=[]
    )
    verified = []
    for name in config["records"]:
        case = output / "simulation/oracle_position_hold" / f"{name}_s43"
        manifest = evidence.json(case / "manifest.json")
        assert manifest["experiment"] == config and manifest["source"] == "oracle"
        assert not list(case.glob("request_*.json")) and not list(case.glob("response_*.json"))
        actual = verify_case(
            config,
            manifest["record"],
            variant,
            case,
            evidence,
            source="oracle",
            checkpoint=None,
            full=True,
            sampling_seed=43,
            first_query_validator=None,
        )
        assert actual["chunks"] == 75 and actual["outcome"]["stable"] and not actual["raw_checkpoint_evidence"]
        assert (
            actual["tracking"]["admitted"]["position_rmse_m"] <= config["qualification"]["position_rmse_max_m"] == 0.01
        )
        assert (
            actual["tracking"]["admitted"]["so3_rmse_rad"] <= config["qualification"]["rotation_rmse_max_rad"] == 0.05
        )
        previous = next(
            c
            for c in old["cases"]
            if c["record"] == name and c["variant"] == "position_hold" and c["max_observation_interval_steps"] == 8
        )
        previous_case = Path(previous["case"])
        previous_manifest = evidence.json(previous_case / "manifest.json")
        previous_config = previous_manifest["experiment"]
        omitted = {"output_dir", "notes", "sampling_seeds", "variants", "qualification"}
        assert {k: v for k, v in config.items() if k not in omitted} == {
            k: v for k, v in previous_config.items() if k not in omitted
        }
        assert previous_config["sampling_seeds"] == [42] and variant == previous_config["variants"][0]
        old_result = verify_case(
            previous_config,
            previous_manifest["record"],
            variant,
            previous_case,
            evidence,
            source="oracle",
            checkpoint=None,
            full=True,
            sampling_seed=42,
            first_query_validator=None,
        )
        assert old_result["tracking"] == previous["tracking"] and old_result["outcome"] == previous["outcome"]
        assert {k: v for k, v in actual.items() if k != "case"} == {k: v for k, v in old_result.items() if k != "case"}
        current_rows = rows(case / "steps.jsonl", evidence)
        previous_rows = rows(previous_case / "steps.jsonl", evidence)
        assert len(current_rows) == len(previous_rows) == 600 and current_rows == previous_rows
        chunks = evidence.json(case / "chunks.json")
        old_chunks = evidence.json(previous_case / "chunks.json")
        assert len(chunks) == len(old_chunks) == 75
        for current, prior in zip(chunks, old_chunks, strict=True):
            assert {k: v for k, v in current.items() if k != "generation_s"} == {
                k: v for k, v in prior.items() if k != "generation_s"
            }
        worker = evidence.json(output / "simulation/logs/oracle_position_hold" / f"{name}_s43.process.json")
        processes["workers"].append(process_record(worker["pid"], ""))
        verified.append(
            dict(
                record=name,
                sampling_seed=43,
                source="oracle",
                qualified=True,
                result=actual,
                same_fixture_seed42_source_case=str(previous_case),
                all600_step_fields_exact_to_seed42=True,
                all75_chunk_fields_exact_except_wall_generation_s=True,
                all_result_fields_exact_except_case_path=True,
                oracle_excluded_from_wam_denominator=True,
            )
        )
    assert not processes["matrix"]["present"] and not any(p["present"] for p in processes["workers"])
    evidence.recheck()
    result = dict(
        schema_version="s2a.seed43-oracle-readonly-qualification.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        qualified=True,
        cases=verified,
        verified_oracle_cases=3,
        verified_physics_log_endpoints=1800,
        registered015_budget=launch["budget"],
        new_model_requests=0,
        new_physics_steps=0,
        fresh_initial_states=0,
        all1800_step_fields_exact_to_corresponding_seed42=True,
        all225_chunk_fields_exact_except_wall_generation_s=True,
        source_proof=proof,
        processes=processes,
        seed43_wam_request_response_verified=False,
        input_sha256=evidence.hashes,
        limitations=[
            "015 executed three new oracle trials under its separate registered1800-step CPU budget; this auditor executes none.",
            "Seed43 environment/log handling is physically qualified with correct references. There is no seed43 WAM request/response qualification.",
            "Reference42 and43 logs agree on every step field; only chunk wall-generation duration is excluded from chunk comparison.",
            "No fresh candidate or success denominator is opened; oracle results never count as WAM capability.",
        ],
    )
    write_new(args.output, result)
    print(
        json.dumps(
            dict(path=str(args.output), sha256=sha256(args.output), complete=True, qualified=True, oracle_cases=3)
        )
    )


if __name__ == "__main__":
    main()
