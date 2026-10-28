"""Describe actual paired prefixes without claiming independent replay or causality."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, reference, require


def main():
    count = 56
    root = REPO / "outputs/tracking_wam_v5_initial_prefix_20261028"
    root.mkdir(exist_ok=False)
    rows, refs = [], []
    for version, date in ((4, "20261027"), (5, "20261028")):
        source = REPO / f"outputs/tracking_preview_wam_development_v{version}_{date}/development/cases"
        path = source / "task0_init0_s42_raw/simulation/8400_masked_position_hold/task0_init0_s42/steps.jsonl"
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        complete = [line for line in lines if line.endswith(b"\n")]
        require(len(complete) >= count, "All56 complete periods required")
        prefix = b"".join(complete[:count])
        snapshot = root / f"v{version}_prefix.jsonl"
        with snapshot.open("xb") as stream:
            stream.write(prefix)
        rows.append([json.loads(line) for line in complete[:count]])
        refs.append(
            dict(
                source_path=str(path),
                snapshot_ref=reference(snapshot),
                prefix_sha256=hashlib.sha256(prefix).hexdigest(),
            )
        )
    fields = {}
    for key in ("command", "measured_source", "after_source", "desired_tcp", "opening_before", "opening_after"):
        left, right = [np.asarray([r[key] for r in group], dtype=float) for group in rows]
        require(
            left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(), "Finite paired shapes"
        )
        changed = np.any((left != right).reshape(count, -1), axis=1)
        fields[key] = dict(
            maximum_absolute_difference=float(np.max(abs(left - right))),
            differing_periods=int(changed.sum()),
            first_difference=None if not changed.any() else int(np.flatnonzero(changed)[0]),
        )
    result = dict(
        complete=True,
        passed=True,
        periods=count,
        source_ref=reference(Path(__file__)),
        inputs=refs,
        fields=fields,
        success_periods=[sum(r["success"] for r in group) for group in rows],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        scope="Saved same-case first56 descriptors; no independent replay, policy-effect attribution or task qualification",
    )
    print(json.dumps(publish(root / "comparison.json", result)), flush=True)


if __name__ == "__main__":
    main()
