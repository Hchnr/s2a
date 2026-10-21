"""Final-only016 service budget and owned release sidecar; no execution."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, process_record, write_new

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
READY = ROOT / "tracking_service_review/actual_ready_resource_snapshot.json"
READY_SHA = "be69ea89b57844749603485e985d73c319fdf55145d1aff09b372a9233520012"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    assert args.audit.resolve().is_relative_to(REPO / "outputs")
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(args.audit, args.audit_sha256)
    audit = evidence.json(args.audit)
    assert audit["schema_version"] == "s2a.tracking-replan-readonly-wam-audit.v1"
    assert audit["complete"] and audit["all_cases_independently_verified"] and audit["verified_cases"] == 3
    assert audit["resources"]["owned_service_released_normally"]
    for filename, digest in audit["input_sha256"].items():
        evidence.check(filename, digest)
    evidence.check(READY, READY_SHA)
    ready = evidence.json(READY)
    assert ready["ready"] and ready["source_identity_verified"] and ready["actual_http_info_equals_saved"]
    assert ready["physical_gpu"] == 7 and ready["port"] == 9597
    for filename, digest in ready["source_sha256"].items():
        evidence.check(filename, digest)
    directory = ROOT / "tracking_replan_wam/orchestration"
    model = evidence.json(directory / "model_process.json")
    matrix = evidence.json(directory / "matrix_process.json")
    manifest = evidence.json(directory / "manifest.json")
    assert ready["actual_info"] == evidence.json(directory / "server_info.json")
    historical = {p["role"]: p for p in ready["local_registered_processes"]}
    assert set(historical) == {"coordinator", "model", "matrix", "physics_task0_init0_s42.process"}
    for role, registered, cuda in (
        ("coordinator", manifest["coordinator"], ""),
        ("model", model, "7"),
        ("matrix", matrix, ""),
    ):
        saved = historical[role]
        assert (saved["pid"], saved["start_ticks"]) == (registered["pid"], registered["start_ticks"])
        assert saved["environment"]["CUDA_VISIBLE_DEVICES"] == cuda
        assert saved["environment"]["LD_LIBRARY_PATH"] == "" and Path(saved["cwd"]) == REPO
        assert saved["proc_state"] not in ("Z", "X")
        assert not process_record(saved["pid"], cuda, start_ticks=saved["start_ticks"])["present"]
    assert historical["model"]["ppid"] == historical["matrix"]["ppid"] == historical["coordinator"]["pid"]
    physics = historical["physics_task0_init0_s42.process"]
    assert physics["ppid"] == matrix["pid"]
    assert physics["environment"]["CUDA_VISIBLE_DEVICES"] == physics["environment"]["LD_LIBRARY_PATH"] == ""
    assert physics["environment"]["MUJOCO_GL"] == physics["environment"]["PYOPENGL_PLATFORM"] == "osmesa"
    assert not process_record(physics["pid"], "", start_ticks=physics["start_ticks"])["present"]
    service = ROOT / "tracking_replan_wam/intervention_services/9597"
    expected = [
        Path(p).resolve() for c in audit["cases"] for p in c["service_provenance"]["generated_request_directories"]
    ]
    assert all(c["status"] == "verified" for c in audit["cases"])
    queries = sum(c["chunks"] for c in audit["cases"])
    assert len(expected) == len(set(expected)) == queries == audit["observed_request_files"]
    assert 45 <= queries <= 225
    actual = {p.resolve() for p in service.glob("generate_*") if p.is_dir()}
    assert actual == set(expected), "unregistered/missing model generation directory"
    for path in actual:
        assert evidence.json(path / "sample_args.json")["output_dir"] == str(path)
        evidence.read(path / "sample_outputs.json")
    log = evidence.read(directory / "service.log").decode()
    http = re.findall(r'"([A-Z]+) ([^ ]+) HTTP/1\.[01]" (\d{3})', log)
    posts = [(path, int(status)) for method, path, status in http if method == "POST"]
    assert len(posts) == queries and all(p == ("/generate", 200) for p in posts)
    assert len(re.findall(r'"POST ', log)) == queries, "unparsed/extra POST attempt"
    started = re.findall(r"Started server process \[(\d+)\]", log)
    assert started == [str(model["pid"])]
    assert log.count("Application startup complete.") == 1
    assert log.count("Uvicorn running on http://127.0.0.1:9597") == 1
    release = evidence.json(directory / "release.json")
    assert release["model_returncode"] in (0, -15) and release["last_job_returncode"] == 0
    assert not release["cleanup_errors"] and release["only_owned_process_groups_stopped"]
    evidence.recheck()
    result = dict(
        schema_version="s2a.tracking-replan-final-service-budget-sidecar.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        actual_generation_directories_equal_all_three_case_provenance=True,
        all_http_post_attempts_are_exact_registered_successful_generation_count=True,
        actual_h40_queries=queries,
        registered_maximum_h40_queries=225,
        extra_model_queries_detected=False,
        historical_ready_source_bound_cuda7_and_parentage_verified=True,
        registered_processes_absent=True,
        external_owned_release_normal=True,
        startup_process_pid=int(started[0]),
        finished_server_log_present=f"Finished server process [{model['pid']}]" in log,
        application_shutdown_log_present="Application shutdown complete." in log,
        internal_service_finally_receipt_inferred=False,
        nvml_to_local_pid_mapping_claimed=False,
        source_audit_sha256=args.audit_sha256,
        new_model_requests=0,
        new_physics_steps=0,
        input_sha256=evidence.hashes,
        limitations=[
            "The existing external release and historical live/proc identity prove owned lifecycle; absent internal finally/shutdown logs are not fabricated.",
            "NVML and local/proc PIDs can differ; this sidecar authenticates live CVD7 without guessing their translation.",
            "This supplements the frozen full case auditor and does not change any task outcome or denominator.",
        ],
    )
    write_new(args.output, result)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), complete=True, queries=queries)))


if __name__ == "__main__":
    main()
