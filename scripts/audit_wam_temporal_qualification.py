"""Read-only020 oracle qualification: literal mechanics plus exact007 replay.

No child, simulator, model, request or fresh performance is started. The one
new row diagnostic is audited separately, and every previous physical/control
field must retain canonical JSON identity. No floating tolerance replaces that
qualification requirement.
"""

import argparse
import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, process_record, write_new
from wam_temporal_position_checker import source_proof, verify_case

from s2a.artifacts import REPO, sha256

CONFIG = "configs/temporal_equal_qualification_20261021.json"
PLAN = "plans/v0.0.1/020_causal_temporal_ensemble_20261021.md"
REGISTRATION = "7025c45"
PRIOR = REPO / "outputs/wam_stability_20261021/short_replan_qualification/qualification_independent_verification.json"
PRIOR_SHA = "64f0eda45d3e5495dc12124795047beade66c6a3489c6b1f8061a637af9dc4ad"
BUDGET = dict(oracle_trials=3, physical_steps=1800, model_requests=0, fresh_initial_states=0)
NEW_RUNTIME = (
    "src/s2a/temporal_ensemble.py",
    "src/s2a/temporal_ensemble_simulation.py",
    "scripts/run_wam_temporal_matrix.py",
)
NEW_AUDIT = ("scripts/wam_temporal_position_checker.py", "scripts/audit_wam_temporal_qualification.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def common_row_identity(current, previous):
    """Only the expressly new provenance field can be excluded from identity."""
    assert len(current) == len(previous) == 600
    for row, old in zip(current, previous, strict=True):
        assert set(row) == set(old) | {"temporal_ensemble"}
        assert "temporal_ensemble" not in old
    common = [{key: value for key, value in row.items() if key != "temporal_ensemble"} for row in current]
    different = [
        step for step, (row, old) in enumerate(zip(common, previous, strict=True)) if canonical(row) != canonical(old)
    ]
    return dict(
        all600_common_step_fields_canonical_exact=not different,
        different_step_count=len(different),
        first_different_steps=different[:10],
        excluded_new_diagnostic_fields=["temporal_ensemble"],
        numeric_tolerance_used_for_identity=False,
    )


def matrix_command():
    return [
        str(REPO / ".venv/bin/python"),
        str(REPO / "scripts/run_wam_temporal_matrix.py"),
        "--config",
        str(REPO / CONFIG),
        "--seed",
        "42",
        "--source",
        "oracle",
        "--variant",
        "temporal_equal",
        "--workers",
        "2",
    ]


def launch_check(path, digest, evidence):
    evidence.check(path, digest)
    launch = evidence.json(path)
    assert launch["schema_version"] == "s2a.temporal-equal-oracle-root-launch.v1"
    assert (
        launch["registration_commit"]
        == subprocess.check_output(["git", "rev-parse", REGISTRATION], cwd=REPO, text=True).strip()
    )
    assert launch["registered_budget"] == BUDGET
    assert launch["source"] == "oracle" and launch["checkpoint"] is None
    assert launch["experiment_seed"] == launch["sampling_seed"] == 42
    assert launch["cuda_visible_devices"] == launch["ld_library_path"] == ""
    assert launch["cpu_workers"] == 2 and not launch["model_started"] and not launch["automatic_retry"]
    assert launch["command"] == matrix_command()
    assert launch["config_path"] == str(REPO / CONFIG)
    evidence.check(REPO / CONFIG, launch["config_sha256"])
    for relative in (CONFIG, PLAN):
        registered = subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
        evidence.check(REPO / relative, hashlib.sha256(registered).hexdigest())
    evidence.check(launch["source_review_path"], launch["source_review_sha256"])
    review = evidence.json(launch["source_review_path"])
    assert review["review_passed"]
    for source, signature in review["input_sha256"].items():
        evidence.check(source, signature)
    for relative in NEW_RUNTIME + NEW_AUDIT:
        assert str(REPO / relative) in review["input_sha256"]
    for relative in NEW_RUNTIME:
        assert str(REPO / relative) in launch["execution_source_sha256"]
    for source, signature in launch["execution_source_sha256"].items():
        assert source not in review["input_sha256"] or review["input_sha256"][source] == signature
        evidence.check(source, signature)
    runtime_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", launch["runtime_git_commit"]], cwd=REPO, text=True
    ).strip()
    for relative in NEW_RUNTIME:
        registered = subprocess.check_output(["git", "show", f"{runtime_commit}:{relative}"], cwd=REPO)
        assert hashlib.sha256(registered).hexdigest() == launch["execution_source_sha256"][str(REPO / relative)]
    assert type(launch["matrix_pid"]) is int and launch["matrix_pid"] > 0
    assert type(launch["matrix_start_ticks"]) is int and launch["matrix_start_ticks"] > 0
    exit_path = Path(launch["matrix_exit_path"]).resolve()
    assert exit_path.is_relative_to(REPO / "outputs")
    return launch


def rows(path, evidence):
    return [json.loads(line) for line in evidence.read(path).splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert Path.cwd().resolve() == REPO
    assert args.launch.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    proof = source_proof(evidence)
    launch = launch_check(args.launch, args.launch_sha256, evidence)
    config = evidence.json(REPO / CONFIG)
    assert config["seed"] == 42 and config["sampling_seeds"] == [42]
    assert config["records"] == [f"task{i}_init0" for i in range(3)]
    assert config["qualification"] == dict(
        expected_cases=3, expected_physics_steps=1800, position_rmse_max_m=0.01, rotation_rmse_max_rad=0.05
    )
    assert config["variants"] == [
        dict(
            name="temporal_equal",
            rotation="response",
            gripper="position_hold",
            governor=False,
            model_rotation_alpha=1.0,
        )
    ]
    evidence.check(config["fixture"], config["fixture_sha256"])
    records = {record["name"]: record for record in evidence.json(config["fixture"])["records"]}
    evidence.check(PRIOR, PRIOR_SHA)
    prior = evidence.json(PRIOR)
    assert (
        prior["complete"]
        and prior["qualified"]
        and prior["verified_cases"] == 12
        and prior["verified_physics_steps"] == 7200
    )
    for path, signature in prior["input_sha256"].items():
        evidence.check(path, signature)
    old_cases = {
        case["record"]: case
        for case in prior["cases"]
        if case["max_observation_interval_steps"] == 8 and case["variant"] == "position_hold"
    }
    assert set(old_cases) == set(config["records"])
    root = REPO / config["output_dir"]
    matrix = evidence.json(root / "simulation/temporal_matrix_oracle_temporal_equal.json")
    assert matrix["complete"] and not matrix["model_started"] and len(matrix["cases"]) == 3
    assert sorted((case["record"], case["sampling_seed"]) for case in matrix["cases"]) == [
        (name, 42) for name in sorted(config["records"])
    ]
    assert all(case["returncode"] == 0 for case in matrix["cases"])
    logs = root / "simulation/logs/oracle_temporal_equal"
    snapshot_path = logs / "execution_snapshot.json"
    evidence.check(snapshot_path, matrix["execution_snapshot_sha256"])
    snapshot = evidence.json(snapshot_path)
    assert snapshot["identity"] == ["oracle", "temporal_equal", None] and not snapshot["model_started"]
    assert snapshot["jobs"] == [[name, 42] for name in config["records"]] and snapshot["cpu_workers"] == 2
    assert snapshot["config_sha256"] == launch["config_sha256"]
    coordinator = snapshot["coordinator_identity"]
    assert snapshot["coordinator_pid"] == coordinator["pid"] == launch["matrix_pid"]
    assert coordinator["start_ticks"] == launch["matrix_start_ticks"]
    assert coordinator["environment"]["CUDA_VISIBLE_DEVICES"] == coordinator["environment"]["LD_LIBRARY_PATH"] == ""
    assert evidence.json(launch["matrix_exit_path"]) == dict(
        pid=launch["matrix_pid"], start_ticks=launch["matrix_start_ticks"], returncode=0
    )
    for path, signature in snapshot["execution_sha256"].items():
        evidence.check(path, signature)
        assert path in launch["execution_source_sha256"] and launch["execution_source_sha256"][path] == signature
    processes = [
        dict(role="matrix", **process_record(launch["matrix_pid"], "", start_ticks=launch["matrix_start_ticks"]))
    ]
    verified, errors = [], []
    for name in config["records"]:
        folder = root / "simulation/oracle_temporal_equal" / f"{name}_s42"
        try:
            assert (
                not list(folder.glob("request_*.json"))
                and not list(folder.glob("response_*.json"))
                and not list(folder.glob("input_*.png"))
            )
            actual = verify_case(
                config,
                records[name],
                config["variants"][0],
                folder,
                evidence,
                source="oracle",
                checkpoint=None,
                full=True,
                sampling_seed=42,
                first_query_validator=None,
            )
            assert actual["steps"] == 600 and actual["chunks"] == 75 and not actual["raw_checkpoint_evidence"]
            assert actual["oracle_native_source_outcome"]["stable"]
            current_rows, previous_rows = (
                rows(folder / "steps.jsonl", evidence),
                rows(Path(old_cases[name]["case"]) / "steps.jsonl", evidence),
            )
            identity = common_row_identity(current_rows, previous_rows)
            chunks = evidence.json(folder / "chunks.json")
            previous_chunks = evidence.json(Path(old_cases[name]["case"]) / "chunks.json")
            common_chunks = [{k: v for k, v in chunk.items() if k != "generation_s"} for chunk in chunks]
            old_chunks = [{k: v for k, v in chunk.items() if k != "generation_s"} for chunk in previous_chunks]
            chunk_identity = canonical(common_chunks) == canonical(old_chunks)
            worker = evidence.json(logs / f"{name}_s42.process.json")
            matrix_case = next(case for case in matrix["cases"] if case["record"] == name)
            assert matrix_case == dict(record=name, sampling_seed=42, returncode=0, **worker)
            assert worker["ppid"] == launch["matrix_pid"]
            environment = worker["environment"]
            assert environment["CUDA_VISIBLE_DEVICES"] == environment["LD_LIBRARY_PATH"] == ""
            assert environment["MUJOCO_GL"] == environment["PYOPENGL_PLATFORM"] == "osmesa"
            expected_worker = [
                str((REPO / config["simulation_python"]).absolute()),
                str(REPO / "scripts/run_wam_temporal_matrix.py"),
                "--config",
                str(REPO / CONFIG),
                "--seed",
                "42",
                "--source",
                "oracle",
                "--variant",
                "temporal_equal",
                "--record",
                name,
                "--sampling-seed",
                "42",
                "--worker-snapshot",
                str(snapshot_path),
            ]
            assert worker["command"] == expected_worker
            processes.append(
                dict(role="physics", **process_record(worker["pid"], "", start_ticks=worker["start_ticks"]))
            )
            qualified = (
                actual["outcome"]["stable"]
                and actual["tracking"]["admitted"]["position_rmse_m"] <= 0.01
                and actual["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
                and identity["all600_common_step_fields_canonical_exact"]
                and chunk_identity
                and actual["independent_temporal_ensemble_replay"]["passed"]
            )
            verified.append(
                dict(
                    record=name,
                    qualified=bool(qualified),
                    result=actual,
                    old007_case=str(old_cases[name]["case"]),
                    common_rows=identity,
                    all75_common_chunk_fields_exact_excluding_wall_generation_time=chunk_identity,
                    oracle_excluded_from_wam_denominator=True,
                )
            )
        except Exception:
            errors.append(dict(record=name, case=str(folder), error=traceback.format_exc()))
    all_cases = len(verified) == 3 and not errors
    released = len(processes) == 4 and not any(item["present"] for item in processes)
    exact = all_cases and all(case["common_rows"]["all600_common_step_fields_canonical_exact"] for case in verified)
    qualified = all_cases and released and all(case["qualified"] for case in verified)
    evidence.recheck()
    result = dict(
        schema_version="s2a.temporal-equal-oracle-qualification.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=all_cases,
        qualified=qualified,
        verified_oracle_cases=len(verified),
        verified_physics_log_endpoints=sum(case["result"]["steps"] for case in verified),
        normal_process_release=released,
        all1800_common_step_fields_exact_to_007=exact,
        cases=verified,
        errors=errors,
        processes=processes,
        registered_budget=BUDGET,
        source_proof=proof,
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        oracle_is_wam_success_evidence=False,
        source_oracle_only=True,
        launch_path=str(args.launch.resolve()),
        launch_sha256=args.launch_sha256,
        limitations=[
            "Correct-reference old-scene execution qualification only; no WAM task capability or fresh acceptance.",
            "Exactly the old007 interval8/v3 physical and diagnostic rows must repeat; no numeric tolerance replaces canonical identity.",
            "Only the new perstep temporal_ensemble provenance is excluded from old-field identity and is independently rebuilt.",
            "No tuning, automatic retry, model, new physics or future input is introduced by this audit.",
        ],
    )
    write_new(args.output, result)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                complete=all_cases,
                qualified=qualified,
                verified_oracle_cases=len(verified),
                normal_process_release=released,
            )
        )
    )
    if not qualified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
