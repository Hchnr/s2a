"""Admit one isolated recovery of the interrupted component study."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.simulation.runtime import bind_causal_event_bridge_factorial as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_factorial_recovery_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_factorial_recovery_20261111.md"
INCIDENT = REPO / "reports/tracking/joint_control/causal_event_factorial_interruption_20261111.md"
RUNNER = REPO / "scripts/simulation/runtime/run_causal_event_bridge_factorial.py"
ORIGINAL_QA = {
    "path": str(REPO / "outputs/qa/causal_event_bridge_factorial_20261110/result.json"),
    "sha256": "83ea7f96173bd439dc386b83ece84d5899f749f99702738649270bd043af1e3f",
}
ORIGINAL_DRY = {
    "path": str(REPO / "outputs/causal_event_bridge_factorial_dry_20261110/result.json"),
    "sha256": "8859115f2b04f649fde0ae65eb3a01c7e8bcc3cc448b5782a0edbb166d0d490b",
}
MODULE = "scripts.simulation.runtime.run_causal_event_bridge_factorial"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            INCIDENT,
            REPO / "tests/research/native_runtime/test_causal_event_factorial_recovery.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-factorial-recovery.v1",
            "original_head": "9dfde8667ce629a4c3ce3cc553f2c55eefba1a9b",
            "recovery_mode": "restart_from_original_initial_state",
            "prior_registration": {
                "path": "outputs/causal_event_bridge_factorial_20261110/producer/registration.json",
                "sha256": "861acede26517b592e335a3324027d2f9f19b12a3a67b308f66d9edb4006676f",
            },
            "prior_references": {
                "path": "outputs/causal_event_bridge_factorial_20261110/producer/references.json",
                "sha256": "952417fb54b96acd926a7b49c522a84ec3a45f5b7cbe902993660a8930f458c8",
            },
            "prior_worker_logs": {
                "bridge_pose_old_opening": {
                    "path": "outputs/causal_event_bridge_factorial_20261110/producer/bridge_pose_old_opening/worker.log",
                    "sha256": "55ea575e5adf4c8d6a199700ec11fbef4fbc3253792ed0edd3e7e254d75d9d68",
                },
                "old_pose_bridge_opening": {
                    "path": "outputs/causal_event_bridge_factorial_20261110/producer/old_pose_bridge_opening/worker.log",
                    "sha256": "1a870779e34211eb6fd4198c5185d3650ff70c184b70d1421121ea1b112f9aa7",
                },
            },
            "prior_recorded_min_control_steps_per_branch": 64,
            "prior_final_exit_and_cost_known": False,
            "new_attempts": 1,
            "automatic_retry": False,
            "world_model_calls": 0,
            "fresh_cases": 0,
        },
        "Exact registered isolated recovery and prior evidence",
    )
    return ref, config


def absolute_ref(ref):
    return {"path": str(REPO / ref["path"]), "sha256": ref["sha256"]}


def original_process_ids(proc_root=Path("/proc")):
    """Find live original module invocations, excluding isolated recoveries."""
    result = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if MODULE in args and "--recovery" not in args and ("--producer" in args or "--worker" in args):
            result.append(int(proc.name))
    return sorted(result)


def last_progress(path, mode):
    records = []
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))
    prefix.require(bool(records), "Recorded original progress required")
    last = records[-1]
    prefix.require(
        last["phase"] == "producer"
        and last["branch"] == mode
        and last["control_completed"] == 64
        and last["new_commands"] == 0
        and last["native_steps"] == {"attempted": 1600, "completed": 1600},
        "Original last recorded lower-bound physical cost",
    )
    return last


def verify_prior_sources(registration, original_head):
    """Retain original source identity while permitting the new routing code."""
    runner_ref = None
    for ref in registration["source_refs"]:
        if ref["path"] == str(RUNNER):
            prefix.require(runner_ref is None, "One original driver source reference")
            runner_ref = ref
        else:
            prefix.checked(ref)
    prefix.require(runner_ref is not None, "Original driver source reference is present")
    original = subprocess.check_output(
        ["git", "show", f"{original_head}:{RUNNER.relative_to(REPO)}"], cwd=REPO, timeout=30
    )
    prefix.require(hashlib.sha256(original).hexdigest() == runner_ref["sha256"], "Archived original driver bytes")
    return runner_ref


def admit(inputs):
    """Validate old evidence and reference components without candidate policies."""
    config_ref, config = read_config()
    prefix.require(not original_process_ids(), "The interrupted original owner and workers must be absent")
    old_root = REPO / "outputs/causal_event_bridge_factorial_20261110/producer"
    prefix.require(
        not (old_root / "result.json").exists() and not (old_root / "workers.json").exists(),
        "The interrupted original must not have acquired a final lifecycle receipt",
    )
    registration_ref, reference_ref = map(absolute_ref, (config["prior_registration"], config["prior_references"]))
    registration, document = prefix.read(registration_ref), prefix.read(reference_ref)
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == binding.input_receipt(inputs)
        and registration["reference_ref"] == reference_ref
        and registration["producer_branch_refs"] is None,
        "Original registration binds the same inputs and reference artifact",
    )
    runner_ref = verify_prior_sources(registration, config["original_head"])
    audits = {mode: binding.verify_reference(inputs, mode, document["references"][mode]) for mode in binding.MODES}
    prefix.require(
        document["config_ref"] == inputs["config_ref"]
        and document["case_id"] == inputs["config"]["case_id"]
        and document["source_sha256"] == inputs["source_sha256"]
        and audits == document["independent_component_checks"],
        "Independently reconstruct the original saved component interventions",
    )
    logs, progress = {}, {}
    for mode in binding.MODES:
        logs[mode] = absolute_ref(config["prior_worker_logs"][mode])
        progress[mode] = last_progress(prefix.checked(logs[mode]), mode)
    qa, dry = prefix.read(ORIGINAL_QA), prefix.read(ORIGINAL_DRY)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and dry["complete"] is dry["passed"] is True
        and dry["model_initializations"] == dry["physics_steps"] == dry["world_model_calls"] == dry["fresh_cases"] == 0,
        "Original CPU QA and zero-model dry are complete",
    )
    return {
        "config_ref": config_ref,
        "original_head": config["original_head"],
        "prior_registration_ref": registration_ref,
        "prior_reference_ref": reference_ref,
        "prior_worker_log_refs": logs,
        "prior_last_progress": progress,
        "prior_final_exit_and_cost_known": False,
        "original_runner_ref": runner_ref,
        "original_qa_ref": ORIGINAL_QA,
        "original_dry_ref": ORIGINAL_DRY,
        "independent_component_checks": audits,
    }, document
