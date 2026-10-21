"""Narrow017 sidecar: exact physical prefix, explicitly retained logging differences.

The original full6 audit stays False. This sidecar does not rerun or alter its
1200-step mechanical scoring. It qualifies the plan's actual state/command
prefix only after identifying the four diagnostic endpoint interpolation fields.
"""

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_long_qualification_plan import (
    CONFIG,
    OLD_ORACLE_CONFIG,
    RECORDS,
    Evidence,
    cpu_environment,
    launch_check,
)
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO, sha256

ORIGINAL = REPO / "outputs/wam_stability_20261021/long_raw_preflight/qualification_full6_verified.json"
ORIGINAL_SHA = "c86569abeb1fb7e57bfd08468cc5e02cf598ef9c51f75a0da66b54e73e95357c"
CAUSAL_REVIEW = REPO / "outputs/wam_stability_20261021/long_raw_implementation/endpoint599_independent_causality.json"
CAUSAL_REVIEW_SHA = "05ba5ae30c9ec826faf906b621868bb7e051b8f569077f05e0a0c0f58e25e6bb"
DIAGNOSTICS = {"raw_desired_tcp", "effective_desired_tcp", "raw_model_tcp_error", "effective_model_tcp_error"}


def exact(a, b):
    """Compare canonical JSON tokens, including float signs and numeric types."""
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(b, sort_keys=True, separators=(",", ":"))


def physical_prefix(current, previous):
    assert len(current) == 1200 and len(previous) == 600
    differences = []
    for step, (new, old) in enumerate(zip(current[:600], previous, strict=True)):
        assert new["step"] == old["step"] == step and set(new) == set(old)
        changed = {key for key in new if not exact(new[key], old[key])}
        if changed:
            assert step == 599 and changed == DIAGNOSTICS, (step, changed)
            for key in sorted(changed):
                a, b = np.asarray(new[key], dtype=float), np.asarray(old[key], dtype=float)
                assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                difference = float(np.max(np.abs(a - b)))
                assert 0 < difference <= 1e-10, (key, difference)
                differences.append(dict(step=step, field=key, maximum_absolute_difference=difference))
        assert all(exact(new[key], old[key]) for key in new if key not in DIAGNOSTICS)
    assert len(differences) == 4, "the exact known terminal logging difference must be present"
    return dict(
        actual_state_command_motor_and_all_other_fields_exact=True,
        all600_shared_rows_exact=False,
        exception_fields=sorted(DIAGNOSTICS),
        only_exception_step=599,
        logging_differences=differences,
        existing_interface_tolerance=1e-10,
        new_scoring_tolerance_added=False,
    )


def sample_diagnostic(chunk, kind, model_time, dt):
    """Reconstruct with SciPy primitives, without the production sampler helper."""
    poses = np.asarray(chunk[f"{kind}_targets_tcp"], dtype=float)
    cursor = float(np.clip(model_time / dt, 0, len(poses)))
    index = min(int(cursor), len(poses) - 1)
    fraction = cursor - index
    first = np.asarray(chunk["anchor_tcp"], dtype=float) if index == 0 else poses[index - 1]
    last = poses[index]
    pose = first.copy()
    pose[:3, 3] += fraction * (last[:3, 3] - first[:3, 3])
    relative = Rotation.from_matrix(last[:3, :3] @ first[:3, :3].T).as_rotvec()
    pose[:3, :3] = Rotation.from_rotvec(fraction * relative).as_matrix() @ first[:3, :3]
    return pose, dict(targets=len(poses), cursor=cursor, index=index, fraction=fraction)


def diagnostic_reconstruction(new_row, old_row, new_chunk, old_chunk, h, dt):
    assert new_chunk["step"] == old_chunk["step"] == 592
    assert new_row["step"] == old_row["step"] == 599
    assert new_row["model_time_s"] == old_row["model_time_s"] == 8 * dt
    branches = {}
    for label, row, chunk, expected in (
        ("old", old_row, old_chunk, dict(targets=8, cursor=8.0, index=7, fraction=1.0)),
        ("new", new_row, new_chunk, dict(targets=40, cursor=8.0, index=8, fraction=0.0)),
    ):
        actual_tcp = np.asarray(row["after_source"], dtype=float) @ h
        residuals = {}
        for kind in ("raw", "effective"):
            pose, branch = sample_diagnostic(chunk, kind, row["model_time_s"], dt)
            assert branch == expected
            saved_pose = np.asarray(row[f"{kind}_desired_tcp"], dtype=float)
            np.testing.assert_allclose(pose, saved_pose, atol=1e-10, rtol=0)
            rotation = Rotation.from_matrix(pose[:3, :3] @ actual_tcp[:3, :3].T).as_rotvec()
            error = np.concatenate((pose[:3, 3] - actual_tcp[:3, 3], rotation))
            saved_error = np.asarray(row[f"{kind}_model_tcp_error"], dtype=float)
            np.testing.assert_allclose(error, saved_error, atol=1e-10, rtol=0)
            residuals[kind] = dict(
                maximum_pose_absolute_difference=float(np.max(np.abs(pose - saved_pose))),
                maximum_error_absolute_difference=float(np.max(np.abs(error - saved_error))),
                exactly_equal=bool(np.array_equal(pose, saved_pose) and np.array_equal(error, saved_error)),
            )
        branches[label] = dict(
            **expected,
            all_four_logged_values_reconstructed_within_existing_tolerance=True,
            existing_interface_tolerance=1e-10,
            actual_reconstruction_residuals=residuals,
        )
    return branches


def diagnostic_only_source(path):
    """Guard the actual source's diagnostic consumers and direct control inputs."""
    tree = ast.parse(path.read_text())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    loads = {}
    for name, pose_key, error_key in (
        ("raw_desired", "raw_desired_tcp", "raw_model_tcp_error"),
        ("effective_desired", "effective_desired_tcp", "effective_model_tcp_error"),
    ):
        found = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
                continue
            parent = parents[node]
            if isinstance(parent, ast.keyword):
                assert parent.arg == pose_key
            else:
                assert isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                assert parent.func.id == "pose_error" and parent.args[0] is node
                assert isinstance(parents[parent], ast.keyword) and parents[parent].arg == error_key
            found.append(node.lineno)
        assert len(found) == 2
        loads[name] = found
    text = path.read_text()
    for required in (
        "desired_tcp, desired_opening = targets[index], float(openings[index])",
        "desired = desired_tcp @ h_inv",
        "converted = converter.convert(measured, desired, twist)",
        'command = np.r_[converted["command"], grip_command]',
        "env.step(command)",
    ):
        assert required in text
    return dict(
        diagnostic_consumers_only=loads,
        control_uses_actual_targets_index=True,
        source_evidence=[
            "src/s2a/long_simulation.py:331",
            "src/s2a/long_simulation.py:350",
            "src/s2a/long_simulation.py:356",
            "src/s2a/long_simulation.py:367",
            "src/s2a/stability_recovery_simulation.py:98",
            "src/s2a/reference_governor.py:38",
        ],
    )


def read_rows(path, evidence):
    return [json.loads(line) for line in evidence.read(path).decode().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(ORIGINAL, ORIGINAL_SHA)
    original = evidence.json(ORIGINAL)
    assert original["qualification_passed"] is False and original["native_qualification_passed"] is True
    assert original["all_cases_complete"] and original["normal_process_release"]
    assert (original["native_trials"], original["oracle_trials"], original["physical_steps"]) == (3, 3, 7200)
    for name, digest in original["input_sha256"].items():
        evidence.check(name, digest)
    evidence.check(CAUSAL_REVIEW, CAUSAL_REVIEW_SHA)
    causal = evidence.json(CAUSAL_REVIEW)
    assert (
        causal["physical_core_consistency_proven_for_original600"]
        and causal["raw_effective_diagnostics_not_control_inputs"]
    )
    assert causal["unchanged_existing_numeric_tolerance"] == 1e-10
    for name, digest in causal["input_sha256"].items():
        evidence.check(name, digest)
    for relative in (
        "src/s2a/long_simulation.py",
        "src/s2a/stability_recovery_simulation.py",
        "src/s2a/reference_governor.py",
    ):
        evidence.read(REPO / relative)
    source_proof = diagnostic_only_source(REPO / "src/s2a/long_simulation.py")
    config = evidence.json(CONFIG)
    old_root = REPO / evidence.json(OLD_ORACLE_CONFIG)["output_dir"]
    release = {}
    for source in ("native", "oracle"):
        previous = original["process_release"][source]
        release[source] = launch_check(
            Path(previous["launch_path"]), previous["launch_sha256"], source, config, evidence
        )
    cases = {}
    for name in RECORDS:
        prior = original["map"][name]
        assert prior["native"]["qualified"] and prior["native"]["first600"]["all600_shared_rows_exact"]
        oracle = prior["oracle"]
        assert oracle["qualified"] is False
        assert oracle["result"]["outcome"]["stable"]
        tracking = oracle["result"]["tracking"]["admitted"]
        assert tracking["position_rmse_m"] <= 0.01 and tracking["so3_rmse_rad"] <= 0.05
        assert oracle["first600"]["all75_common_query_fields_and_available_endpoints_exact"]
        current = Path(oracle["case"])
        old = old_root / f"simulation/oracle_position_hold/{name}_s42"
        rows, old_rows = read_rows(current / "steps.jsonl", evidence), read_rows(old / "steps.jsonl", evidence)
        prefix = physical_prefix(rows, old_rows)
        chunks, old_chunks = evidence.json(current / "chunks.json"), evidence.json(old / "chunks.json")
        assert len(chunks) == 150 and len(old_chunks) == 75
        manifest = evidence.json(current / "manifest.json")
        tool = evidence.json(manifest["runtime_config"]["tool_contract"])
        h = np.asarray(tool["source_from_tcp"], dtype=float)
        reconstruction = diagnostic_reconstruction(
            rows[599], old_rows[599], chunks[74], old_chunks[74], h, 1 / config["control_fps"]
        )
        cases[name] = dict(
            native_qualified=True,
            oracle_physical_core_qualified=True,
            original_oracle_strict_qualified=False,
            oracle_outcome=oracle["result"]["outcome"],
            oracle_admitted_tracking=tracking,
            first600=prefix,
            independent_endpoint_diagnostic_reconstruction=reconstruction,
            original_full1200_mechanical_and_query_checks_unchanged=True,
            all75_available_query_endpoints_exact=True,
        )
    evidence.recheck()
    result = dict(
        schema_version="s2a.long-qualification-physical-core.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        qualification_passed=True,
        qualification_gate_scope="full1200 mechanics/task/thresholds unchanged; first600 actual state/command/motor exact; four terminal diagnostic interpolation fields explicitly separate",
        original_qualification_passed=False,
        original_strict_result=dict(path=str(ORIGINAL), sha256=ORIGINAL_SHA, qualification_passed=False),
        independent_causality_review=dict(path=str(CAUSAL_REVIEW), sha256=CAUSAL_REVIEW_SHA),
        all_cases_complete=True,
        native_trials=3,
        oracle_trials=3,
        physical_steps=7200,
        physical_core_cases=cases,
        normal_process_release=all(item["normal_process_release"] for item in release.values()),
        process_release=release,
        source_proof=source_proof,
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        oracle_excluded_from_wam_denominator=True,
        no_original_artifact_modified=True,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output), qualification_passed=True)))


if __name__ == "__main__":
    main()
