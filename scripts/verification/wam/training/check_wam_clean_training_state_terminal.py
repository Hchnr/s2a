"""Read-only joint048 terminal qualification for the subsequent development stage."""

import argparse
from pathlib import Path

from s2a.research.wam.training.state_terminal import qualify


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(dict(path=str(args.registration.resolve()), sha256=args.registration_sha256))))


if __name__ == "__main__":
    main()
