"""087 reconstruct native forward-response inputs and separate next-state labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import sha256
from s2a.native_arm_response_data import reconstruct_native_arm_data

CONFIG = Path("configs/native_arm_response_data_20261025.json")
ROOT = Path("outputs/native_arm_response_data_20261025")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-index", type=int, choices=range(30))
    args = parser.parse_args()
    e = Evidence()
    for path in (__file__, "src/s2a/native_arm_response_data.py", "src/s2a/xml_site_kinematics.py"):
        e.read(path)
    config = e.json(CONFIG)
    for key in ("roster_ref", "prior_closure_ref", "paired_analysis_ref"):
        ref = config[key]
        e.check(ref["path"], ref["sha256"])
    closure = e.json(config["prior_closure_ref"]["path"])
    assert (
        closure["complete"]
        and closure["data_qualified"]
        and closure["verified_cases"] == 42
        and closure["owner_normal_wait_verified"]
    )
    assert len(config["trajectories"]) == 30
    assert sum(x["split"] == "fit" for x in config["trajectories"]) == 24
    assert sum(x["split"] == "holdout" for x in config["trajectories"]) == 6
    indices = range(30) if args.check_index is None else [args.check_index]
    if args.check_index is None:
        ROOT.mkdir(exist_ok=True)
        assert not list(ROOT.glob("trajectory*.npz"))
        assert not (ROOT / "failure.json").exists() and not (ROOT / "reconstruction_result.json").exists()
    entries = []
    try:
        for i in indices:
            entry = config["trajectories"][i]
            for key in ("ref", "result_ref", "goal_log_ref"):
                ref = entry[key]
                e.check(ref["path"], ref["sha256"])
            result = e.json(entry["result_ref"]["path"])
            goals = e.json(entry["goal_log_ref"]["path"])
            assert result["complete"] and result["passed"] and result["native_quality_passed"]
            with np.load(entry["ref"]["path"], allow_pickle=False) as source:
                arrays, metrics = reconstruct_native_arm_data(result, goals, source)
            if args.check_index is None:
                path = ROOT / f"trajectory{i:02d}.npz"
                with path.open("xb") as stream:
                    np.savez(stream, **arrays)
                cache_ref = dict(path=str(path.resolve()), sha256=sha256(path))
            else:
                cache_ref = None
            entries.append(
                dict(
                    index=i,
                    split=entry["split"],
                    task=entry["task"],
                    source=entry["ref"],
                    cache_ref=cache_ref,
                    metrics=metrics,
                )
            )
            print("reconstructed", i, entry["split"], metrics["rows"], "rows", flush=True)
        e.recheck()
    except BaseException as error:
        if args.check_index is None:
            (ROOT / "failure.json").write_text(
                json.dumps(dict(complete=False, error=repr(error), completed=entries, physics_steps=0), indent=2) + "\n"
            )
        raise
    if args.check_index is not None:
        return
    result = dict(
        complete=True,
        reconstruction_passed=True,
        independent_data_qualification=False,
        trajectories=entries,
        rows_by_split={s: sum(x["metrics"]["rows"] for x in entries if x["split"] == s) for s in ("fit", "holdout")},
        source_sha256=e.hashes,
        physics_steps=0,
        new_fits=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "reconstruction_result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
