"""096 run unchanged numerical checks in a separate owned evidence directory."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("holdout", "anchors"))
    args = p.parse_args()
    if args.stage == "holdout":
        import check_causal_local_prequential as check
    else:
        import check_causal_local_anchors as check
    check.ROOT = Path("outputs/causal_local_response_20261025/owned/results")
    check.main()


if __name__ == "__main__":
    main()
