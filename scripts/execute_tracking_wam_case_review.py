"""Own a closed case's independent read-only mechanical review and actual wait."""

import argparse
import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage
from s2a.terminal_development_policy import case_order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-index", type=int, required=True, choices=range(12))
    args = p.parse_args()
    case = case_order()[args.case_index]
    prior_ref = reference(REPO / "outputs/tracking_wam_independent_review_20261027/registration.json")
    prior = read(prior_ref)
    worker_ref = reference(
        REPO
        / f"outputs/tracking_preview_wam_development_v3_20261027/development/cases/{case['case_id']}/worker_result.json"
    )
    worker = read(worker_ref)
    require(worker["complete"] is worker["passed"] is True, "Closed production case before review")
    root = REPO / f"outputs/tracking_wam_independent_review_20261027/case{args.case_index:02d}_execution"
    require(not root.exists(), "No overwrite or automatic retry")
    sources = dict(prior["source_sha256"])
    for ref in [reference(Path(__file__)), worker_ref]:
        sources[ref["path"]] = ref["sha256"]
    reg = publish(
        root / "registration.json",
        dict(
            prior_review_registration_ref=prior_ref,
            worker_ref=worker_ref,
            source_sha256=sources,
            case=case,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        stage = owned_stage(
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                "scripts/audit_tracking_wam_completed_case.py",
                "--case-index",
                str(args.case_index),
            ],
            environment=cpu_environment(),
            directory=root / "mechanical",
            timeout_s=1800,
            source_sha256=sources,
        )
        ref = reference(REPO / f"outputs/tracking_wam_case_review_20261027/case{args.case_index:02d}/verified.json")
        verified = read(ref)
        require(verified["complete"] is verified["passed"] is True, "Independent mechanical case checks")
        result.update(complete=True, passed=True, stage_ref=stage, verified_ref=ref, task_outcome=verified["outcome"])
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, mechanical_passed=True, task_outcome=result["task_outcome"])), flush=True)


if __name__ == "__main__":
    main()
