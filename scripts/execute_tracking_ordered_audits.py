"""Own explicit batches of closed ordered-candidate input and image audits."""

import argparse
import json
import re
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-indices", type=int, nargs="+", required=True, choices=range(18))
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    require(
        re.fullmatch(r"[a-z0-9_]+", args.run_id) and len(set(args.case_indices)) == len(args.case_indices),
        "Unique cases and simple run identifier",
    )
    root = REPO / f"outputs/tracking_ordered_audits_{args.run_id}"
    require(not root.exists(), "No overwrite or automatic retry")
    producer_root = REPO / "outputs/tracking_correct_reference_ordered_execution_20261027"
    producer_ref = reference(producer_root / "registration.json")
    sources = dict(read(producer_ref)["source_sha256"])
    kinds = ["robot_inputs", "labels_images"]
    for path in [Path(__file__), *[REPO / f"scripts/audit_tracking_ordered_{kind}.py" for kind in kinds]]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    closed = [reference(producer_root / f"case{i:02d}/closed.json") for i in args.case_indices]
    for ref in closed:
        item = read(ref)
        v = read(item["verified_ref"])
        require(v["complete"] is v["passed"] is True, "Complete native replay")
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            producer_registration_ref=producer_ref,
            closed_case_refs=closed,
            case_indices=args.case_indices,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg, stages=[])
    try:
        for index in args.case_indices:
            for kind in kinds:
                output = REPO / f"outputs/tracking_ordered_{kind}_20261027/case{index:02d}/verified.json"
                if output.exists():
                    ref = reference(output)
                    verified = read(ref)
                    require(verified["complete"] is verified["passed"] is True, "Previously completed audit")
                    require(
                        verified["result_ref"]
                        == reference(
                            REPO
                            / f"outputs/tracking_correct_reference_ordered_20261027/case{index:02d}_candidate/result.json"
                        ),
                        "Reused result identity",
                    )
                    result["stages"].append(dict(case_index=index, kind=kind, reused=True, verified_ref=ref))
                else:
                    stage = owned_stage(
                        [
                            str(REPO / "outputs/libero_env311/bin/python"),
                            f"scripts/audit_tracking_ordered_{kind}.py",
                            "--case-index",
                            str(index),
                        ],
                        environment=cpu_environment(),
                        directory=root / f"case{index:02d}" / kind,
                        timeout_s=600,
                        source_sha256=sources,
                    )
                    ref = reference(output)
                    verified = read(ref)
                    require(verified["complete"] is verified["passed"] is True, "Complete audit")
                    result["stages"].append(
                        dict(case_index=index, kind=kind, reused=False, stage_ref=stage, verified_ref=ref)
                    )
            print(json.dumps(dict(completed_case=index)), flush=True)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
