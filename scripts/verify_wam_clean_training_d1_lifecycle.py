"""050 independent read-only D1 lifecycle addendum CLI; no new math/model."""

import argparse
import json
import os
from pathlib import Path

from s2a.clean_training_d1_lifecycle_addendum import ROOT, verify
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, publish, require


def main():
    require(Path.cwd().resolve() == REPO, "050 must run from S2A")
    check_cpu_environment(os.environ)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    args = p.parse_args()
    value = verify(dict(path=str(args.manifest.resolve()), sha256=args.manifest_sha256))
    print(json.dumps(publish(ROOT / "result.json", value)), flush=True)


if __name__ == "__main__":
    main()
