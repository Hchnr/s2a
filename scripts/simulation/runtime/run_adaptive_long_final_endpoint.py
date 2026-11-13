"""Run the registered CPU-only final-endpoint replay once."""

import argparse

from scripts.simulation.runtime import adaptive_long_final_endpoint_owner as owner
from scripts.simulation.runtime import adaptive_long_final_endpoint_worker as worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--phase", choices=("checker",))
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(owner.dry_run(), flush=True)
    elif args.phase:
        owner.execute()
    else:
        worker.run_worker(source_refs=owner.source_refs)


if __name__ == "__main__":
    main()
