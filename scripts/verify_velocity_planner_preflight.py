"""Independently verify every080 preflight candidate minimum and selected rollout."""

import hashlib
import json
from pathlib import Path

import numpy as np
from velocity_plan_independent_audit import audit_plan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("outputs/velocity_model_planner_preflight_20261025")
    path = root / "preflight.json"
    preflight = json.loads(path.read_text())
    refs = dict(preflight["source_sha256"])
    refs[str(path)] = digest(path)
    refs[__file__] = digest(__file__)
    refs["scripts/velocity_plan_independent_audit.py"] = digest("scripts/velocity_plan_independent_audit.py")
    assert all(digest(p) == sha for p, sha in refs.items())
    params = json.loads(Path("outputs/velocity_residual_response_20261025/parameters.json").read_text())
    with np.load("outputs/velocity_residual_response_20261025/fit_cache.npz", allow_pickle=False) as cache:
        bound = 1.5 * max(
            float(np.max(np.abs(cache["raw"][:, 3]))), float(np.max(np.abs(cache["raw"][:, 3] + cache["target"][:, 1])))
        )
    assert bound == preflight["velocity_envelope"]
    results = []
    for entry in preflight["queries"]:
        results.append(
            audit_plan(
                params,
                bound,
                entry["current_opening"],
                entry["current_motor"],
                entry["current_period_velocity"],
                entry["generated_reference"],
                entry["plan"],
            )
        )
    assert all(digest(p) == sha for p, sha in refs.items())
    output = dict(
        numerically_verified=True,
        verified_queries=len(results),
        full_scalar_search=True,
        velocity_envelope=bound,
        results=results,
        source_sha256=refs,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
        physical_qualification=False,
    )
    with (root / "independent_verified.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(
        "independent complete scalar search verified",
        len(results),
        "queries;near ties",
        sum(row["numerical_near_tie"] for row in results),
    )


if __name__ == "__main__":
    main()
