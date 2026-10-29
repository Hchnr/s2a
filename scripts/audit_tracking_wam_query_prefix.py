"""Read-only request/image/response checks of a completed live query prefix."""

import argparse
import json
from pathlib import Path

from audit_wam_position import Evidence
from tracking_preview_wam_acceptance_checker_review import verify_query

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.terminal_development_policy import case_order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-index", type=int, choices=range(12), required=True)
    p.add_argument("--queries", type=int, required=True)
    args = p.parse_args()
    require(1 <= args.queries <= 75, "Positive bounded query prefix")
    root = REPO / f"outputs/tracking_wam_query_review_20261027/case{args.case_index:02d}_first{args.queries:02d}"
    require(not root.exists(), "No overwrite")
    run = REPO / "outputs/tracking_preview_wam_development_v3_20261027"
    registration_ref = reference(run / "registration.json")
    reg = read(registration_ref)
    original = REPO / "scripts/tracking_preview_wam_acceptance_checker_v3.py"
    checked(dict(path=str(original), sha256=reg["source_sha256"][str(original)]))
    case = case_order()[args.case_index]
    config_ref = reference(run / f"development/configs/{case['case_id']}.json")
    config = read(config_ref)
    require(config["wam_development"]["case"] == case, "Actual case config identity")
    directory = (
        REPO
        / config["output_dir"]
        / f"simulation/8400_masked_position_hold/{case['record_name']}_s{case['sampling_seed']}"
    )
    current = json.loads((directory / "chunks.json").read_text())
    require(len(current) >= args.queries, "Only completed chunks can be checked")
    chunks = current[: args.queries]
    ev = Evidence()
    for index, chunk in enumerate(chunks):
        require(chunk["step"] == index * 8 and chunk["physics_unchanged"] is True, "Original causal schedule")
        require(chunk["physics_before_sha256"] == chunk["physics_after_sha256"], "Physical clock held during query")
        verify_query(config, directory, index, chunk, ev, "8400_masked", sampling_seed=case["sampling_seed"])
    ev.recheck()
    snapshot_ref = publish(root / "observed_chunks.json", dict(chunks=chunks))
    ref = publish(
        root / "verified.json",
        dict(
            complete=True,
            passed=True,
            case=case,
            registration_ref=registration_ref,
            config_ref=config_ref,
            prefix_snapshot_ref=snapshot_ref,
            queries=args.queries,
            source_refs=[
                reference(Path(__file__)),
                reference(REPO / "scripts/tracking_preview_wam_acceptance_checker_review.py"),
                reference(original),
            ],
            evidence_sha256=ev.hashes,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Saved request/pixel/response prefix only; not full native state, task or whole-run qualification",
        ),
    )
    print(json.dumps(ref), flush=True)


if __name__ == "__main__":
    main()
