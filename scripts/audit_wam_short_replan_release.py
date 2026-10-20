"""Final-only010 service budget: all678 requests and both owned releases."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, write_new

from s2a.artifacts import REPO, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(args.audit, args.audit_sha256)
    audit = evidence.json(args.audit)
    assert audit["schema_version"] == "s2a.short-replan-readonly-audit.v1" and audit["complete"]
    assert audit["all12_cases_independently_verified"] and audit["verified_physics_steps"] == 7200
    assert len(audit["cases"]) == 12 and audit["observed_request_files"] == 678
    for path, digest in audit["source_before_after_hashes"].items():
        evidence.check(path, digest)
    records = []
    for interval in (8, 16):
        resource = next(r for r in audit["resources"] if r["interval"] == interval)
        assert resource["owned_service_released_normally"] and resource["recorded_execution_error"] is None
        config = evidence.json(f"configs/wam_short_replan_{interval}_alpha1_20261021.json")
        root = REPO / config["output_dir"]
        port = config["checkpoints"][0]["port"]
        service = (root / "intervention_services" / str(port)).resolve()
        directory = root / "orchestration" / f"short_replan_interval{interval}"
        cases = [c for c in audit["cases"] if c["interval"] == interval]
        assert len(cases) == 6 and all(c["status"] == "verified" for c in cases)
        expected = [
            Path(p).resolve() for c in cases for p in c["shared_service_provenance"]["generated_request_directories"]
        ]
        actual = {p.resolve() for p in service.glob("generate_*") if p.is_dir()}
        count = 450 if interval == 8 else 228
        assert len(expected) == len(set(expected)) == len(actual) == count
        assert actual == set(expected), "extra/missing generation not in registered six-case source set"
        log = evidence.read(directory / "service.log").decode()
        requests = re.findall(r'"([A-Z]+) ([^ ]+) HTTP/1\.[01]" (\d{3})', log)
        posts = [(path, int(status)) for method, path, status in requests if method == "POST"]
        assert len(posts) == len(re.findall(r'"POST ', log)) == count
        assert all(p == ("/generate", 200) for p in posts)
        model = evidence.json(directory / "model_process.json")
        release = evidence.json(directory / "release.json")
        assert model["cuda_visible_devices"] == str(4 if interval == 8 else 5)
        assert model["ld_library_path"] == "" and release["model_pid"] == model["pid"]
        assert release["model_start_ticks"] == model["start_ticks"]
        assert release["model_returncode"] in (0, -15) and release["last_job_returncode"] == 0
        assert release["only_owned_process_groups_stopped"] and release["model_absent"]
        assert re.findall(r"Started server process \[(\d+)\]", log) == [str(model["pid"])]
        assert log.count(f"Uvicorn running on http://127.0.0.1:{port}") == 1
        records.append(
            dict(
                interval=interval,
                exact_generation_directories=count,
                exact_successful_post_requests=count,
                only_registered_requests=True,
                model_pid=model["pid"],
                model_returncode=release["model_returncode"],
                external_release_normal=True,
                finished_server_log_present=f"Finished server process [{model['pid']}]" in log,
                application_shutdown_log_present="Application shutdown complete." in log,
            )
        )
    evidence.recheck()
    value = dict(
        schema_version="s2a.short-replan-final-service-budget-sidecar.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        source_audit_sha256=args.audit_sha256,
        exact_total_model_requests=678,
        extra_model_queries_detected=False,
        records=records,
        new_model_requests=0,
        new_physics_steps=0,
        input_sha256=evidence.hashes,
        internal_finally_receipts_inferred=False,
        limitations=[
            "Both alpha profiles share each interval's alpha1 private service; all six cases are included.",
            "This adds request/lifecycle evidence without changing any native outcome or counting existing oracles.",
        ],
    )
    write_new(args.output, value)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), complete=True, exact_requests=678)))


if __name__ == "__main__":
    main()
