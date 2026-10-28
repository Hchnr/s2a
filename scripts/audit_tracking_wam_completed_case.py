"""Audit a closed600-step case while the fixed twelve-case producer continues.

This does not certify the complete service denoise ledger or development gate.
"""

import argparse
import ctypes
import json
from pathlib import Path

from audit_wam_position import Evidence as PhysicsEvidence
from coupled_preview_wam_measured_audit import verify_measured_endpoints
from tracking_preview_wam_acceptance_checker_review import verify_case
from tracking_wam_review_source_binding import verify_source_binding
from verify_tracking_preview_wam_development_review import (
    current_query_evidence_scope,
    label_current_mechanical_evidence,
    normal_owned,
    outcome,
)
from verify_wam_clean_training_capture import Evidence

from s2a.artifacts import REPO
from s2a.clean_training_runtime import read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF
from s2a.terminal_development_policy import case_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(12), required=True)
    args = parser.parse_args()
    strict_native_environment()
    root = REPO / "outputs/tracking_preview_wam_development_v3_20261027"
    case = case_order()[args.case_index]
    config_ref = reference(root / f"development/configs/{case['case_id']}.json")
    config = read(config_ref)
    worker_ref = reference(REPO / config["output_dir"] / "worker_result.json")
    worker = read(worker_ref)
    require(
        worker["complete"] is worker["passed"] is True
        and worker["physics_steps"] == 600
        and worker["actual_requests"] == 75,
        "Closed600/75 case required",
    )
    require(worker["case"] == case and worker["config_ref"] == config_ref, "Actual case/config binding")
    output = REPO / f"outputs/tracking_wam_case_review_20261027/case{args.case_index:02d}/verified.json"
    require(not output.exists(), "No overwrite")
    directory = root / f"development/orchestration/{case['case_id']}"
    command = {
        key: reference(directory / name)
        for key, name in [("launch", "launch.json"), ("exit", "exit.json"), ("release", "ownership/release.json")]
    }
    ev = Evidence()
    launch = normal_owned(ev, command, "")
    require(
        launch["command"][1] == "scripts/collect_tracking_preview_wam_development_v3.py", "Actual collector command"
    )
    argv = launch["command"]
    require(
        argv[argv.index("--case-config") + 1] == config_ref["path"]
        and argv[argv.index("--case-config-sha256") + 1] == config_ref["sha256"],
        "Pre-execution config path/SHA binding",
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    physical = PhysicsEvidence()
    path = Path(worker["physical_case_dir"])
    manifest = physical.json(path / "manifest.json")
    prepared = read(worker["prepared_ref"])
    source_binding = verify_source_binding(ev, physical, prepared, worker, config_ref, command)
    ctypes.CDLL(manifest["runtime_config"]["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
    fixture = physical.json(config["fixture"])
    record = next(r for r in fixture["records"] if r["name"] == case["record_name"])
    result = dict(
        complete=False,
        passed=False,
        worker_ref=worker_ref,
        config_ref=config_ref,
        owned_command=command,
        source_refs=[
            reference(Path(__file__)),
            reference(REPO / "scripts/tracking_preview_wam_acceptance_checker_review.py"),
            reference(REPO / "scripts/tracking_preview_arm_log_replay.py"),
            reference(REPO / "scripts/tracking_wam_review_source_binding.py"),
            reference(REPO / "scripts/verify_tracking_preview_wam_development_review.py"),
        ],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    try:
        mechanical = verify_case(
            config,
            record,
            config["variants"][0],
            path,
            physical,
            source="wam",
            checkpoint="8400_masked",
            full=True,
            sampling_seed=case["sampling_seed"],
            first_query_validator=current_query_evidence_scope,
            oracle_reference=None,
        )
        mechanical = label_current_mechanical_evidence(mechanical)
        measured = verify_measured_endpoints(physical, config, worker, path)
        rows = [json.loads(line) for line in physical.read(path / "steps.jsonl").splitlines()]
        task = outcome([r["success"] for r in rows])
        require(task == mechanical["outcome"], "Independent600/tail40 outcome")
        physical.recheck()
        ev.finish()
        result.update(
            complete=True,
            passed=True,
            case=case,
            mechanical=mechanical,
            measured_endpoint_audit=measured,
            outcome=task,
            evidence_sha256=physical.hashes,
            source_binding=source_binding,
            source_binding_scope="Prepared immutable source map plus actual launch config SHA and runtime manifest; no separate producer execution_snapshot_ref was emitted",
            scope="One closed native case/request/label/image audit; complete900-request service numerical ledger and development qualification still pending",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(output, result)


if __name__ == "__main__":
    main()
