"""Prepare019's fixed second-seed launch; no model, HTTP or physics."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from run_wam_long_service import weight_stats
from run_wam_position_matrix import protocol
from run_wam_short_replan import cpu_environment, process_record, verify_hashes, write_new

from s2a.artifacts import REPO, sha256
from s2a.stability_position_simulation import execution_inputs

ROOT = REPO / "outputs/wam_stability_20261021"
REGISTRATION = "8401aa3"
PLAN = "plans/v0.0.1/019_raw8_seed43_replication_20261021.md"
PREREQUISITES = {
    "seed43_qualification": (
        ROOT / "seed43_execution_qualification/qualification_independent_verification.json",
        "9cd4cf46260eda2d87e13bd992c5ce871469d4ff2d3f38734129e2a1bd5fbba3",
    ),
    "long_raw_final": (
        ROOT / "long_raw_audit/complete3_verified.json",
        "b3910cdeb6669e5f6f74a654d44f2a214dc3b12249d33391e085efbb6a375168",
    ),
}


def registered(path):
    expected = hashlib.sha256(subprocess.check_output(["git", "show", f"{REGISTRATION}:{path}"], cwd=REPO)).hexdigest()
    if sha256(REPO / path) != expected:
        raise ValueError(f"019 registration changed: {path}")
    return expected


def merge(signatures, additions):
    """Retain every prerequisite digest; current imports cannot replace it."""
    conflicts = [name for name, digest in additions.items() if name in signatures and signatures[name] != digest]
    if conflicts:
        raise ValueError(f"019 immutable input conflict: {conflicts}")
    signatures.update(additions)


def unused_root(root):
    """The frozen generic --dry-run creates an empty root via load_config."""
    return not root.exists() or root.is_dir() and not any(root.iterdir())


def prepare():
    cpu_environment()
    if Path.cwd().resolve() != REPO:
        raise ValueError("run from S2A")
    signatures = {str(REPO / PLAN): registered(PLAN)}
    prerequisites = {}
    for label, (path, expected) in PREREQUISITES.items():
        if sha256(path) != expected:
            raise ValueError(f"019 prerequisite changed: {label}")
        item = json.loads(path.read_text())
        if not item["complete"]:
            raise ValueError(f"019 prerequisite incomplete: {label}")
        if label == "seed43_qualification":
            assert item["qualified"] and item["verified_oracle_cases"] == 3
            assert item["verified_physics_log_endpoints"] == 1800
            assert item["all1800_step_fields_exact_to_corresponding_seed42"]
        else:
            assert item["all_cases_independently_verified"] and item["verified_cases"] == 3
            assert sum(x["exact_requests"] for x in item["final_http_and_directory_budget"]) == 450
            assert all(x["owned_service_released_normally"] for x in item["resources"])
        merge(signatures, item["input_sha256"])
        merge(signatures, {str(path): expected})
        prerequisites[label] = dict(path=str(path), sha256=expected)
    ready_path = ROOT / "long_raw_review/actual_ready_resource_snapshot.json"
    ready = json.loads(ready_path.read_text())
    absent = []
    for part in ready["partitions"]:
        for identity in part["local_registered_processes"]:
            state = process_record(
                identity["pid"], identity["environment"]["CUDA_VISIBLE_DEVICES"], start_ticks=identity["start_ticks"]
            )
            if state["present"]:
                raise ValueError("017 process still present;019 must not overlap it")
            absent.append(dict(pid=identity["pid"], start_ticks=identity["start_ticks"], present=False))
    if len(absent) != 12:
        raise ValueError("017 historical identity roster incomplete")
    base = json.loads((REPO / "configs/wam_short_replan_8_alpha1_20261021.json").read_text())
    partitions = []
    for task in range(3):
        name = f"configs/wam_raw8_seed43_task{task}_20261021.json"
        merge(signatures, {str(REPO / name): registered(name)})
        config, root, records, ck = protocol(name, source="wam", variant_name="position_hold", checkpoint="8400_masked")
        expected = json.loads(json.dumps(base))
        expected.update(
            output_dir=f"outputs/wam_stability_20261021/raw8_seed43/task{task}",
            records=[f"task{task}_init0"],
            sampling_seeds=[43],
            notes=config["notes"],
        )
        expected["checkpoints"] = [dict(base["checkpoints"][0], gpu=4 + task, port=9744 + task)]
        if config != expected or len(records) != 1 or not unused_root(root):
            raise ValueError("019 must retain registered raw8 protocol and a fresh output")
        merge(signatures, execution_inputs(config, name))
        stats = weight_stats(ck)
        command = [
            str(REPO / ".venv/bin/python"),
            "scripts/run_wam_stability_service.py",
            "--config",
            name,
            "--seed",
            "42",
            "--checkpoint",
            "8400_masked",
            "--variants",
            "position_hold",
            "--fingerprints",
            "outputs/wam_independent_20261020/checkpoint_fingerprints.json",
            "--matrix-entry",
            "scripts/run_wam_position_matrix.py",
        ]
        partitions.append(
            dict(
                partition=task,
                physical_gpu=ck["gpu"],
                port=ck["port"],
                config_path=str(REPO / name),
                config_sha256=sha256(REPO / name),
                output_dir=str(root),
                preexisting_empty_output_root=root.exists(),
                command=command,
                actual_checkpoint_stat=stats,
            )
        )
    for name in (
        __file__,
        "scripts/capture_wam_raw8_seed43_ready.py",
        "scripts/run_wam_stability_service.py",
        "scripts/run_wam_position_matrix.py",
        "scripts/serve_wam_acceptance.py",
        "src/s2a/wam_runtime.py",
    ):
        path = Path(name).resolve()
        merge(signatures, {str(path): sha256(path)})
    verify_hashes(signatures)
    return dict(
        schema_version="s2a.raw8-seed43-preflight.v1",
        complete=True,
        registration_commit=subprocess.check_output(["git", "rev-parse", REGISTRATION], cwd=REPO, text=True).strip(),
        code_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        registered_budget=dict(
            wam_trials=3, physics_steps=1800, h40_queries=225, new_oracle_trials=0, fresh_initial_states=0
        ),
        prerequisites=prerequisites,
        prior12_owned_processes_absent=absent,
        partitions=partitions,
        input_sha256=signatures,
        new_model_requests=0,
        new_physics_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    if not path.is_relative_to(REPO / "outputs") or path.exists():
        raise ValueError("prepare into a new repository output file")
    result = prepare()
    write_new(path, result)
    print(json.dumps(dict(path=str(path), sha256=sha256(path), complete=True)))


if __name__ == "__main__":
    main()
