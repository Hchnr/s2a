"""Own and reap the four preregistered native open-loop diagnostics."""

import json
import os
import subprocess
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.native_environment import strict_native_environment


def main():
    strict_native_environment()
    root = REPO / "outputs/long_joint_physical_20261027"
    root.mkdir(exist_ok=False)
    producer = read(reference(REPO / "outputs/long_joint_preview_probe_20261027/verified.json"))
    sources = dict(read(producer["registration_ref"])["source_sha256"])
    for path in (Path(__file__), REPO / "scripts/run_long_joint_physical.py"):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    registration_ref = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            schedule=[dict(case=i, horizon=h) for i in (0, 1) for h in (8, 16)],
            physics_budget=272,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    receipts = []
    for case in (0, 1):
        for horizon in (8, 16):
            for path, digest in sources.items():
                checked(dict(path=path, sha256=digest))
            name = f"case{case:02d}_h{horizon}"
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            with (root / f"{name}.log").open("w") as log:
                child = subprocess.Popen(
                    [
                        str(REPO / "outputs/libero_env311/bin/python"),
                        str(REPO / "scripts/run_long_joint_physical.py"),
                        "--case-index",
                        str(case),
                        "--horizon",
                        str(horizon),
                    ],
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                code = child.wait()
            receipt = dict(case=case, horizon=horizon, pid=child.pid, wait_returncode=code, reaped=True)
            receipts.append(receipt)
            publish(root / "receipts.json", receipts)
            require(code == 0, f"{name} failed; preserve output and do not automatically retry")
            print(json.dumps(receipt), flush=True)
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    print(
        json.dumps(
            publish(
                root / "completed.json",
                dict(
                    complete=True,
                    passed=True,
                    registration_ref=registration_ref,
                    receipts=receipts,
                    control_qualified=False,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
