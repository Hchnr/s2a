"""Static narrow017 wrapper fork: canonical Git identity, frozen helpers reused.

Only launch_check/main are explicitly copied from fd518ed. Every physics,
metric, reference, clock and prefix checker is imported from its unchanged
SHA-pinned file. No dynamic patch, exec, environment or model construction.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_long_qualification import (
    CONFIG,
    OLD_ORACLE_CONFIG,
    PLAN,
    RECORDS,
    REPO,
    RUNTIME_COMMIT,
    Evidence,
    cpu_environment,
    episode_check,
    factory_proof,
    lines,
    native_check,
    oracle_prefix,
    process_record,
    runtime_proof,
    sha256,
    source_proof,
    verify_case,
    write_new,
)

ORIGINAL_CHECKER = REPO / "scripts/audit_wam_long_qualification.py"
ORIGINAL_CHECKER_SHA = "d147b2f1edaab0cddef821751352c3d5d50904686f37ad821c738de1155facfc"


def canonical_runtime_commit(value):
    """Resolve a saved Git identifier to the single registered full commit."""
    if not isinstance(value, str) or not value:
        raise ValueError("missing runtime commit identity")
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{value}^{{commit}}"],
        cwd=REPO,
        text=True,
    ).strip()
    if resolved != RUNTIME_COMMIT:
        raise ValueError("runtime commit resolves to a different registered source")
    return resolved


def launch_check(path, digest, source, config, evidence):
    path = path.resolve()
    assert path.is_relative_to(REPO / "outputs")
    evidence.check(path, digest)
    launch = evidence.json(path)
    assert launch["budget_git_commit"] == "9fe490a"
    assert canonical_runtime_commit(launch["runtime_git_commit"]) == RUNTIME_COMMIT
    subprocess.run(["git", "merge-base", "--is-ancestor", launch["runtime_git_commit"], "HEAD"], check=True)
    assert launch["config_sha256"] == sha256(CONFIG)
    assert launch["budget"] == {
        f"{source}_trials": 3,
        "physical_steps": 3600,
        "model_requests": 0,
        "fresh_initial_states": 0,
    }
    assert launch["cuda_visible_devices"] == launch["ld_library_path"] == ""
    assert launch["model_started"] is False and launch["automatic_retry"] is False
    command = launch["command"]
    assert Path(command[0]).absolute() == (REPO / ".venv/bin/python").absolute()
    assert Path(command[1]).resolve() == REPO / "scripts/run_wam_long_matrix.py"
    for key, expected in (
        ("--config", str(CONFIG)),
        ("--source", source),
        ("--variant", "position_hold"),
        ("--seed", "42"),
        ("--workers", "2"),
    ):
        actual = command[command.index(key) + 1]
        assert Path(actual).resolve() == CONFIG if key == "--config" else actual == expected
    assert not any(k in command for k in ("--job-indices", "--record", "--checkpoint"))
    evidence.check(launch["source_review_path"], launch["source_review_sha256"])
    review = evidence.json(launch["source_review_path"])
    assert review["source_review_passed"] and review["new_model_requests"] == review["new_physics_steps"] == 0
    for name, h in review["input_sha256"].items():
        evidence.check(name, h)
    for name, h in launch["execution_source_sha256"].items():
        evidence.check(name, h)
    for relative in (
        "src/s2a/long_environment.py",
        "src/s2a/long_simulation.py",
        "scripts/run_wam_long_matrix.py",
        str(PLAN.relative_to(REPO)),
    ):
        assert str(REPO / relative) in launch["execution_source_sha256"]
    root = REPO / config["output_dir"]
    label = "native" if source == "native" else "oracle_position_hold"
    matrix = evidence.json(root / f"simulation/long_matrix_{label}.json")
    snapshot_path = root / f"simulation/logs/{label}/execution_snapshot.json"
    evidence.check(snapshot_path, matrix["execution_snapshot_sha256"])
    snapshot = evidence.json(snapshot_path)
    assert matrix["complete"] and len(matrix["cases"]) == 3 and all(c["returncode"] == 0 for c in matrix["cases"])
    assert snapshot["identity"] == [source, "position_hold", None]
    assert snapshot["coordinator_pid"] == launch["matrix_pid"] and snapshot["cpu_workers"] == 2
    coordinator_identity = snapshot["coordinator_identity"]
    assert (
        coordinator_identity["pid"] == launch["matrix_pid"]
        and coordinator_identity["start_ticks"] == launch["matrix_start_ticks"]
    )
    assert (
        coordinator_identity["environment"]["CUDA_VISIBLE_DEVICES"]
        == coordinator_identity["environment"]["LD_LIBRARY_PATH"]
        == ""
    )
    assert snapshot["config_sha256"] == launch["config_sha256"] and snapshot["model_started"] is False
    assert snapshot["jobs"] == [[r, 42] for r in RECORDS]
    for name, h in snapshot["execution_sha256"].items():
        evidence.check(name, h)
    processes = [
        dict(role="matrix", **process_record(launch["matrix_pid"], "", start_ticks=launch["matrix_start_ticks"]))
    ]
    assert sorted((c["record"], c["sampling_seed"]) for c in matrix["cases"]) == [(r, 42) for r in RECORDS]
    for case in matrix["cases"]:
        worker = evidence.json(root / f"simulation/logs/{label}/{case['record']}_s42.process.json")
        assert worker["pid"] == case["pid"] and worker["command"] == case["command"]
        assert worker["environment"]["CUDA_VISIBLE_DEVICES"] == worker["environment"]["LD_LIBRARY_PATH"] == ""
        assert worker["ppid"] == launch["matrix_pid"]
        assert worker["start_ticks"] == case["start_ticks"]
        processes.append(
            dict(role=case["record"], **process_record(worker["pid"], "", start_ticks=worker["start_ticks"]))
        )
    assert not any(p["present"] for p in processes)
    return dict(
        launch_path=str(path),
        launch_sha256=digest,
        processes=processes,
        normal_process_release=True,
        case_paths={r: str(root / "simulation" / label / f"{r}_s42") for r in RECORDS},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("review", "native", "full"), required=True)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--launch-sha256")
    parser.add_argument("--oracle-launch", type=Path)
    parser.add_argument("--oracle-launch-sha256")
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(ORIGINAL_CHECKER, ORIGINAL_CHECKER_SHA)
    for relative in (
        "scripts/verify_wam_acceptance.py",
        "scripts/run_wam_short_replan.py",
        "src/s2a/task_acceptance.py",
        "src/s2a/artifacts.py",
    ):
        evidence.read(REPO / relative)
    proof = dict(factory=factory_proof(evidence), runtime=runtime_proof(evidence), mechanics=source_proof(evidence))
    config = evidence.json(CONFIG)
    evidence.read(PLAN)
    assert config["records"] == RECORDS and config["sampling_seeds"] == [42]
    assert config["online_max_steps"] == 1200 and config["episode_horizon"] == 1300 and config["ignore_done"] is False
    registered = subprocess.check_output(["git", "show", f"9fe490a:{CONFIG.relative_to(REPO)}"])
    assert hashlib.sha256(registered).hexdigest() == sha256(CONFIG)
    evidence.check(config["fixture"], config["fixture_sha256"])
    records = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
    if args.stage == "review":
        evidence.recheck()
        write_new(
            args.output,
            dict(
                source_review_passed=True,
                source_proof=proof,
                input_sha256=evidence.hashes,
                new_model_requests=0,
                new_physics_steps=0,
            ),
        )
        return
    assert args.launch and args.launch_sha256
    release = {"native": launch_check(args.launch, args.launch_sha256, "native", config, evidence)}
    results = {
        r: dict(native=native_check(Path(release["native"]["case_paths"][r]), config, records[r], evidence))
        for r in RECORDS
    }
    native_pass = all(row["native"]["qualified"] for row in results.values())
    assert native_pass, "017 native is incomplete/unstable or common600 prefix differs"
    if args.stage == "full":
        assert args.oracle_launch and args.oracle_launch_sha256
        release["oracle"] = launch_check(args.oracle_launch, args.oracle_launch_sha256, "oracle", config, evidence)
        previous_config = evidence.json(OLD_ORACLE_CONFIG)
        for name in RECORDS:
            case = Path(release["oracle"]["case_paths"][name])
            result = verify_case(
                config,
                records[name],
                config["variants"][0],
                case,
                evidence,
                source="oracle",
                checkpoint=None,
                full=True,
                sampling_seed=42,
                first_query_validator=None,
            )
            rows = lines(case / "steps.jsonl", evidence)
            episode = episode_check(case, rows, evidence)
            prefix = oracle_prefix(
                case, REPO / previous_config["output_dir"] / f"simulation/oracle_position_hold/{name}_s42", evidence
            )
            assert not list(case.glob("request_*.json")) and not list(case.glob("response_*.json"))
            qualified = bool(
                result["outcome"]["stable"]
                and result["tracking"]["admitted"]["position_rmse_m"] <= 0.01
                and result["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
                and prefix["all600_shared_rows_exact"]
            )
            results[name]["oracle"] = dict(
                case=str(case),
                qualified=qualified,
                result=result,
                episode_clock=episode,
                first600=prefix,
                excluded_from_wam_denominator=True,
            )
    passed = native_pass and args.stage == "full" and all(row["oracle"]["qualified"] for row in results.values())
    evidence.recheck()
    summary = dict(
        schema_version="s2a.long-qualification-readonly.v1",
        original_checker_sha256=ORIGINAL_CHECKER_SHA,
        canonical_runtime_git_commit=RUNTIME_COMMIT,
        created_utc=datetime.now(timezone.utc).isoformat(),
        stage=args.stage,
        native_qualification_passed=native_pass,
        qualification_passed=passed,
        all_cases_complete=True,
        native_trials=3,
        oracle_trials=3 if args.stage == "full" else 0,
        physical_steps=7200 if args.stage == "full" else 3600,
        map=results,
        normal_process_release=all(r["normal_process_release"] for r in release.values()),
        process_release=release,
        source_proof=proof,
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        oracle_excluded_from_wam_denominator=True,
    )
    write_new(args.output, summary)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                native_qualification_passed=native_pass,
                qualification_passed=passed,
                physical_steps=summary["physical_steps"],
            )
        )
    )


if __name__ == "__main__":
    main()
