"""Independent046 metadata admission; no producer/model or label imports."""

import argparse
import json
from pathlib import Path

from s2a.clean_training_d1_admission import verify_proposal
from s2a.durable_stage_supervisor import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--proposal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify_proposal(dict(path=str(args.proposal.resolve()), sha256=args.proposal_sha256))
    print(json.dumps(publish(args.output, value)), flush=True)


if __name__ == "__main__":
    main()
