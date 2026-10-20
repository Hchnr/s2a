"""Bind completed017 qualification and scoped prior releases; launch nothing."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_wam_long_service import config_path, preflight
from run_wam_short_replan import cpu_environment, verify_hashes, write_new

from s2a.artifacts import REPO, sha256
from s2a.long_simulation import execution_inputs

BASE = REPO / "outputs/wam_stability_20261021"
PRIOR = BASE / "long_raw_preflight/prior_services_verified.json"
PRIOR_SHA = "698d0171234814b787148ddde8b1abaad574de3caacb9f5de9109362d2c54ef7"


def prepare(qualification_path, qualification_sha, output):
    """Write immutable gate files only after all original inputs are rechecked."""
    qualification_path, output = Path(qualification_path).resolve(), Path(output).resolve()
    if (
        not qualification_path.is_relative_to(REPO / "outputs")
        or not output.is_relative_to(REPO / "outputs")
        or output == REPO / "outputs"
        or output.exists()
    ):
        raise ValueError("requires an existing repository qualification and a fresh outputs directory")
    if sha256(qualification_path) != qualification_sha or sha256(PRIOR) != PRIOR_SHA:
        raise ValueError("explicit qualification/prior-release fingerprints differ")
    qualification = json.loads(qualification_path.read_text())
    prior = json.loads(PRIOR.read_text())
    if (
        not qualification["qualification_passed"]
        or not qualification["all_cases_complete"]
        or qualification["native_trials"] != 3
        or qualification["oracle_trials"] != 3
        or qualification["physical_steps"] != 7200
        or not prior["prior_scopes_passed"]
    ):
        raise ValueError("all six actual1200-step qualifications and registered prior scopes are required")
    signatures = {}

    def include(filename, digest=None):
        name = str(Path(filename).resolve())
        expected = sha256(name) if digest is None else digest
        if name in signatures and signatures[name] != expected:
            raise ValueError(f"conflicting evidence fingerprints: {name}")
        signatures[name] = expected

    for item in (qualification, prior):
        for filename, digest in item["input_sha256"].items():
            include(filename, digest)
    include(qualification_path, qualification_sha)
    include(PRIOR, PRIOR_SHA)
    for relative in (
        "plans/v0.0.1/017_long_raw_development_20261021.md",
        "scripts/prepare_wam_long_launch.py",
        "scripts/run_wam_long_service.py",
        "scripts/run_wam_short_replan.py",
        "scripts/run_wam_long_matrix.py",
        "scripts/serve_wam_acceptance.py",
        "src/s2a/wam_runtime.py",
        "src/s2a/long_environment.py",
        "src/s2a/long_simulation.py",
        "scripts/wam_long_position_checker.py",
        "outputs/wam_independent_20261020/checkpoint_fingerprints.json",
    ):
        include(REPO / relative)
    for partition in (0, 1, 2):
        path = REPO / config_path(partition)
        config = json.loads(path.read_text())
        for filename, digest in execution_inputs(config, path).items():
            include(filename, digest)
    verify_hashes(signatures)
    output.mkdir(parents=True, exist_ok=False)
    gates = []
    for partition in (0, 1, 2):
        path = output / f"partition{partition}.json"
        gate = dict(
            schema_version="s2a.long-raw-launch-gate.v1",
            created_utc=datetime.now(timezone.utc).isoformat(),
            all_prerequisites_passed=True,
            qualification=dict(path=str(qualification_path), sha256=qualification_sha),
            prior_release=prior["partitions"][str(partition)],
            input_sha256=signatures,
            budget_git_commit="9fe490a",
            registered_partition_budget=dict(wam_trials=1, physics_steps=1200, h40_queries=150),
            resource_availability_must_be_rechecked_at_launch=True,
            model_started=False,
            physics_started=False,
            fresh_initial_states=0,
        )
        write_new(path, gate)
        # Reuse the actual strict launcher parser/gates; no child or GPU is started.
        _, _, records, checkpoint, checked, weights = preflight(path, sha256(path), partition)
        gates.append(
            dict(
                partition=partition,
                gate_path=str(path),
                gate_sha256=sha256(path),
                old_record=records[0]["name"],
                physical_gpu=checkpoint["gpu"],
                port=checkpoint["port"],
                checked_input_count=len(checked),
                weight_stat_checks=weights,
                actual_preflight_passed=True,
            )
        )
    verify_hashes(signatures)
    receipt = output / "prepared.json"
    write_new(
        receipt,
        dict(
            schema_version="s2a.long-raw-prepared-launches.v1",
            complete=True,
            gates=gates,
            input_sha256=signatures,
            new_model_requests=0,
            new_physics_steps=0,
            actual_gpu_availability_not_reserved=True,
            new_outputs_refuse_overwrite=True,
        ),
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    receipt = prepare(args.qualification, args.qualification_sha256, args.output)
    print(
        json.dumps(dict(prepared_path=str(receipt), sha256=sha256(receipt), model_started=False, physics_started=False))
    )


if __name__ == "__main__":
    main()
