"""Independent051 metadata proposal verification, never model/physics execution."""

import argparse
import json
from pathlib import Path

from s2a.development_repair_admission import PREFIX, validate
from s2a.durable_stage_supervisor import REPO, publish, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--proposal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref = dict(path=str(args.proposal.resolve()), sha256=args.proposal_sha256)
    validate(ref)
    print(
        json.dumps(
            publish(
                args.output,
                dict(
                    schema_version=PREFIX + "verification.v1",
                    complete=True,
                    passed=True,
                    eligible_for_execution=True,
                    proposal_ref=ref,
                    verifier_ref=reference(REPO / "scripts/verify_wam_clean_development_repair.py"),
                    actual_model_calls=0,
                    actual_physics_steps=0,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
