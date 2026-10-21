"""Describe all three completed016 cases using the frozen010 geometry helper."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_wam_position import Evidence
from audit_wam_short_replan_outcomes import describe
from run_wam_short_replan import cpu_environment, write_new

from s2a.artifacts import REPO, sha256

HELPER = "scripts/audit_wam_short_replan_outcomes.py"
HELPER_SHA = "8a2e4c68442c6dce2b2c8c77ec56bc943c353c0038560e0900fbe5b756d90cfa"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--release-audit", type=Path, required=True)
    parser.add_argument("--release-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(HELPER, HELPER_SHA)
    evidence.check(args.audit, args.audit_sha256)
    evidence.check(args.release_audit, args.release_audit_sha256)
    audit, release = evidence.json(args.audit), evidence.json(args.release_audit)
    assert audit["schema_version"] == "s2a.tracking-replan-readonly-wam-audit.v1" and audit["complete"]
    assert audit["all_cases_independently_verified"] and len(audit["cases"]) == 3
    assert release["schema_version"] == "s2a.tracking-replan-final-service-budget-sidecar.v1"
    assert release["complete"] and release["source_audit_sha256"] == args.audit_sha256
    for snapshot in (audit, release):
        for path, digest in snapshot["input_sha256"].items():
            evidence.check(path, digest)
    cases = []
    for case in audit["cases"]:
        assert case["status"] == "verified" and case["raw_checkpoint_evidence"]
        descriptive = describe(dict(case, profile="tracking_min8_max40_raw_alpha1"), evidence)
        descriptive["causal_query_diagnostics"] = case["tracking_replan_replay"]
        descriptive["native_object_contact_and_max_height"] = case["object_metrics"]
        cases.append(descriptive)
    evidence.recheck()
    result = dict(
        schema_version="s2a.tracking-replan-final-outcome-diagnostics.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        source_audit_sha256=args.audit_sha256,
        cases=cases,
        verified_steps=1800,
        new_model_requests=0,
        new_physics_steps=0,
        input_sha256=evidence.hashes,
        limitations=[
            "All three016 old-scene trials are included; no extra simulator/render/predicate evaluation is performed.",
            "Object center XY/contact/height are post-step diagnostics, never policy inputs or new acceptance thresholds.",
            "Execution errors and model reference offsets can coexist; these diagnostics do not uniquely attribute failure.",
            "Existing oracle and010 results are not added to016 success counts.",
        ],
    )
    write_new(args.output, result)
    print(json.dumps(dict(path=str(args.output), sha256=sha256(args.output), complete=True, cases=3)))


if __name__ == "__main__":
    main()
