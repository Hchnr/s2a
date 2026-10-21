"""Read-only017 native3 then oracle3 qualification; never constructs an env."""

import argparse
import ast
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence, close, error
from run_wam_short_replan import cpu_environment, process_record, write_new
from verify_wam_acceptance import verify_case as verify_native_generic
from wam_long_position_checker import source_proof, verify_case

from s2a.artifacts import REPO, sha256
from s2a.task_acceptance import task_outcome

CONFIG = REPO / "configs/long_raw_qualification_20261021.json"
PLAN = REPO / "plans/v0.0.1/017_long_raw_development_20261021.md"
OLD_NATIVE = REPO / "outputs/wam_checkpoint_acceptance_20261019v2/simulation/native"
OLD_ORACLE_CONFIG = REPO / "configs/short_replan_qualification_8_20261021.json"
RECORDS = [f"task{i}_init0" for i in range(3)]
RUNTIME_COMMIT = "d6fb4a6765a5223a2108a9f9e1bbf50c34a9dff8"
CPU_REVIEW = REPO / "outputs/wam_stability_20261021/long_raw_implementation/cpu_source_contract.json"
CPU_REVIEW_SHA = "0ccb6f4ae90d9ff8ee6d8de2b13473c3e940ef7e1b5c102de3fde2c0c71ab9fc"


def lines(path, evidence):
    return [json.loads(row) for row in evidence.read(path).splitlines()]


def factory_proof(evidence):
    """Remove only validation and explicit horizon keywords, then compare AST."""
    old_path, new_path = REPO / "src/s2a/libero_simulation.py", REPO / "src/s2a/long_environment.py"

    def function(path):
        tree = ast.parse(evidence.read(path))
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_environment")

    original, new = function(old_path), copy.deepcopy(function(new_path))
    assert isinstance(new.body[0], ast.Expr)
    assert ast.unparse(new.body[0]) == "validate_episode_settings(config)"
    new.body.pop(0)
    call = next(
        n
        for n in ast.walk(new)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "env_class"
    )
    added = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg in ("horizon", "ignore_done")}
    assert added == dict(horizon="config['episode_horizon']", ignore_done="config['ignore_done']")
    call.keywords = [k for k in call.keywords if k.arg not in added]
    assert ast.dump(new, include_attributes=False) == ast.dump(original, include_attributes=False)
    return dict(
        factory_reset_restore_ast_exact=True,
        only_changes=["validate explicit017 settings", "pass horizon1300/ignore_doneFalse"],
        initialization_policy_steps_added=0,
    )


def runtime_proof(evidence):
    """Independently compare the complete physical try bodies, not just a claim."""
    evidence.check(CPU_REVIEW, CPU_REVIEW_SHA)
    receipt = evidence.json(CPU_REVIEW)
    assert receipt["model_started"] is receipt["physics_started"] is False
    for path, digest in receipt["raw_source_sha256"].items():
        evidence.check(path, digest)

    def function(relative, name):
        tree = ast.parse(evidence.read(REPO / relative))
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    for old_file, old_name, new_name, omit_clock in (
        ("src/s2a/checkpoint_simulation.py", "run_case", "_run_native_case", False),
        ("src/s2a/stability_position_simulation.py", "run_position_case", "run_long_position_case", True),
    ):
        old = next(n for n in function(old_file, old_name).body if isinstance(n, ast.Try)).body
        new = next(n for n in function("src/s2a/long_simulation.py", new_name).body if isinstance(n, ast.Try)).body
        if omit_clock:
            clocks = [
                n
                for n in new
                if isinstance(n, ast.Assign) and ast.unparse(n) == "initial_episode = episode_snapshot(env)"
            ]
            assert len(clocks) == 1
            new = [n for n in new if n is not clocks[0]]
        assert ast.dump(ast.Module(body=old, type_ignores=[]), include_attributes=False) == ast.dump(
            ast.Module(body=new, type_ignores=[]), include_attributes=False
        )
    for relative in (
        "src/s2a/long_environment.py",
        "src/s2a/long_simulation.py",
        "scripts/run_wam_long_matrix.py",
        "tests/test_long_environment.py",
    ):
        registered = subprocess.check_output(["git", "show", f"{RUNTIME_COMMIT}:{relative}"])
        assert hashlib.sha256(registered).hexdigest() == sha256(REPO / relative)
        evidence.read(REPO / relative)
    return dict(
        native_physical_try_body_ast_exact=True,
        oracle_wam_physical_try_body_ast_exact_except_readonly_clock=True,
        runtime_commit=RUNTIME_COMMIT,
        worker_pid_startticks_environment_snapshot=True,
        actual_first600_and1200_physics_pending_until_executed=True,
    )


def episode_check(case, rows, evidence):
    """Recompute recorded clocks without calling the production contract helper."""
    clock = evidence.json(case / "episode_contract.json")
    assert clock["schema_version"] == "s2a.long-episode-contract.v1"
    assert clock["effective_horizon"] == 1300 and clock["ignore_done"] is False
    assert clock["physical_steps"] == clock["observed_policy_steps"] == len(rows) == 1200
    assert clock["initial_timestep"] == 0 and clock["final_timestep"] == 1200
    assert clock["scored_physical_step_budget"] == 1200
    assert clock["initialization_policy_steps_added"] == 0 and clock["initialization_excluded_from_scoring"] is True
    assert clock["initial_internal_done"] is clock["final_internal_done"] is False
    assert clock["clock_consistent"] is True
    assert clock["control_freq"] == 20
    close(clock["control_timestep_s"], 0.05, tolerance=1e-12)
    close(clock["model_timestep_s"], 0.002, tolerance=1e-12)
    close([r["actual_dt"] for r in rows], np.full(1200, 0.05))
    close(clock["final_sim_time_s"] - clock["initial_sim_time_s"], 60, tolerance=1e-9)
    close(sum(r["actual_dt"] for r in rows), 60, tolerance=1e-9)
    for label, timestep, done in (("initial_observation", 0, False), ("final_observation", 1200, False)):
        observation = clock[label]
        assert observation["effective_horizon"] == 1300 and observation["ignore_done"] is False
        assert observation["internal_timestep"] == timestep and observation["internal_done"] is done
        assert observation["control_freq"] == 20
        close(observation["control_timestep_s"], 0.05, tolerance=1e-12)
        close(observation["model_timestep_s"], 0.002, tolerance=1e-12)
        close(
            observation["sim_time_s"],
            clock["initial_sim_time_s" if timestep == 0 else "final_sim_time_s"],
            tolerance=1e-12,
        )
    return dict(
        physical_steps=1200,
        actual_elapsed_s=clock["final_sim_time_s"] - clock["initial_sim_time_s"],
        initial_policy_timestep=0,
        final_policy_timestep=1200,
        effective_horizon=1300,
        ignore_done=False,
    )


def numeric_max_difference(a, b):
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a) == set(b)
        return max((numeric_max_difference(a[k], b[k]) for k in b), default=0.0)
    if isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b)
        return max((numeric_max_difference(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    if isinstance(b, (str, bool)) or b is None:
        assert type(a) is type(b) and a == b
        return 0.0
    assert type(a) in (float, int) and type(b) in (float, int) and np.isfinite(a) and np.isfinite(b)
    return abs(a - b)


def exact_prefix(current, previous):
    assert len(previous) == 600 and len(current) == 1200
    maximum = numeric_max_difference(current[:600], previous)
    exact = current[:600] == previous
    return dict(
        all600_shared_rows_exact=exact,
        maximum_absolute_difference=maximum,
        within_existing_tolerance_1e10=maximum <= 1e-10,
    )


def native_check(case, config, record, evidence):
    verified = verify_native_generic(case / "result.json", config, {record["name"]: record}, {})
    for path, digest in verified["files"].items():
        evidence.check(path, digest)
    result, manifest = evidence.json(case / "result.json"), evidence.json(case / "manifest.json")
    rows = lines(case / "steps.jsonl", evidence)
    assert result["complete"] and result["execution_inputs_unchanged"] and result["source"] == "native"
    assert manifest["schema_version"] == "s2a.long-native-execution.v1" and manifest["experiment"] == config
    assert manifest["long_episode_contract"] == dict(
        episode_horizon=1300, ignore_done=False, physical_steps=1200, initialization_policy_steps_added=0
    )
    assert not list(case.glob("request_*.json")) and not list(case.glob("response_*.json"))
    for path, digest in manifest["execution_sha256"].items():
        evidence.check(path, digest)
    assert set(manifest["simulator_source_sha256"]) <= set(manifest["execution_sha256"])
    evidence.check(case / "model.xml", manifest["initial_model_xml_sha256"])
    assert manifest["actual_initial_physics_sha256"] == record["initial_state_sha256"]
    assert (
        hashlib.sha256(np.asarray(record["initial_state"], float).tobytes()).hexdigest()
        == record["initial_state_sha256"]
    )
    evidence.check(record["expert_path"], record["expert_sha256"])
    commands = np.load(record["expert_path"])
    assert commands.ndim == 2 and commands.shape[1] == 7 and len(commands) and np.isfinite(commands).all()
    for i, row in enumerate(rows):
        expected = commands[i] if i < len(commands) else np.r_[np.zeros(6), commands[-1, -1]]
        np.testing.assert_array_equal(row["command"], expected)
        np.testing.assert_array_equal(row["desired_source"], row["measured_source"])
        close(row["error"], error(np.asarray(row["desired_source"]), np.asarray(row["after_source"])))
        for field in ("measured_source", "after_source"):
            pose = np.asarray(row[field])
            assert pose.shape == (4, 4) and np.isfinite(pose).all()
            close(pose[3], [0, 0, 0, 1], tolerance=1e-12)
            close(pose[:3, :3].T @ pose[:3, :3], np.eye(3), tolerance=1e-8)
            close(np.linalg.det(pose[:3, :3]), 1, tolerance=1e-8)
        if i:
            assert row["opening_target"] == rows[i - 1]["opening_after"]
            np.testing.assert_array_equal(row["measured_source"], rows[i - 1]["after_source"])
    errors = np.asarray([r["error"] for r in rows])
    close(result["step_motion_position_rms_m"], np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))), tolerance=1e-12)
    close(result["step_motion_rotation_rms_rad"], np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1))), tolerance=1e-12)
    close(
        result["opening_step_mae"],
        np.mean([abs(r["opening_after"] - r["opening_target"]) for r in rows]),
        tolerance=1e-12,
    )
    close(result["physical_elapsed_s"], sum(r["actual_dt"] for r in rows), tolerance=1e-12)
    evidence.read(case / "reference.npz")
    with np.load(case / "reference.npz") as reference:
        np.testing.assert_array_equal(reference["poses"], [r["after_source"] for r in rows])
        np.testing.assert_array_equal(reference["openings"], [r["opening_after"] for r in rows])
        assert reference["poses"].shape == (1200, 4, 4) and reference["openings"].shape == (1200,)
    old = OLD_NATIVE / f"{record['name']}_s42"
    old_rows = lines(old / "steps.jsonl", evidence)
    old_manifest = evidence.json(old / "manifest.json")
    assert old_manifest["record"] == record
    prefix = exact_prefix(rows, old_rows)
    outcome = task_outcome([r["success"] for r in rows], expected_steps=1200, stable_steps=40)
    assert outcome == result["task_outcome"] and result["stable_success"] == outcome["stable"]
    clock = episode_check(case, rows, evidence)
    return dict(
        case=str(case),
        source="native",
        outcome=outcome,
        qualified=outcome["stable"] and prefix["all600_shared_rows_exact"],
        first600=prefix,
        episode_clock=clock,
        expert_commands=len(commands),
        terminal_command_hold_steps=max(0, 1200 - len(commands)),
        reference_is_all1200_actual_native_endpoints=True,
        excluded_from_wam_denominator=True,
    )


def oracle_prefix(case, old, evidence):
    prefix = exact_prefix(lines(case / "steps.jsonl", evidence), lines(old / "steps.jsonl", evidence))
    current, previous = evidence.json(case / "chunks.json"), evidence.json(old / "chunks.json")
    assert [c["step"] for c in previous] == list(range(0, 600, 8))
    assert [c["step"] for c in current] == list(range(0, 1200, 8))
    variable_arrays = {"raw128", "effective128", "raw_targets_tcp", "effective_targets_tcp", "openings"}
    truncated = []
    for new, old_chunk in zip(current[:75], previous, strict=True):
        assert set(new) == set(old_chunk)
        old_count = len(old_chunk["raw128"])
        for key in variable_arrays:
            assert new[key][:old_count] == old_chunk[key], (new["step"], key)
        for key in set(old_chunk) - variable_arrays - {"execution_endpoint_count", "generation_s"}:
            assert new[key] == old_chunk[key], (new["step"], key)
        assert old_chunk["execution_endpoint_count"] == old_count
        assert new["execution_endpoint_count"] == len(new["raw128"])
        if old_count != len(new["raw128"]):
            truncated.append(
                dict(step=new["step"], old_available_endpoints=old_count, new_available_endpoints=len(new["raw128"]))
            )
    assert truncated == [
        dict(step=i, old_available_endpoints=n, new_available_endpoints=40)
        for i, n in ((568, 32), (576, 24), (584, 16), (592, 8))
    ]
    return dict(
        **prefix,
        all75_common_query_fields_and_available_endpoints_exact=True,
        unavailable_old_future_suffixes_excluded=truncated,
    )


def launch_check(path, digest, source, config, evidence):
    path = path.resolve()
    assert path.is_relative_to(REPO / "outputs")
    evidence.check(path, digest)
    launch = evidence.json(path)
    assert launch["budget_git_commit"] == "9fe490a"
    assert launch["runtime_git_commit"] == RUNTIME_COMMIT
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
