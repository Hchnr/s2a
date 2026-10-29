"""Decompose completed paired state errors without model or simulator calls."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.score.read_bytes()
    score = json.loads(payload)
    if not score["complete"] or not score["passed"]:
        raise ValueError("Completed scoring receipt required")
    pairs = {}
    for row in score["state_rows"]:
        key = row["name"].replace("_" + row["variant"] + "_", "_PAIRED_")
        variants = pairs.setdefault(key, {})
        if row["variant"] in variants:
            raise ValueError("Duplicate pair member")
        variants[row["variant"]] = row
    if len(pairs) != 48 or any(set(v) != {"raw", "probe052"} for v in pairs.values()):
        raise ValueError("Expected exactly 48 complete pairs")
    groups = {}
    for pair in pairs.values():
        raw = pair["raw"]
        for variant in pair.values():
            for component in ("position", "rotation", "opening"):
                values = variant["errors"]["wam"][component]
                if len(values) != 40 or any(not math.isfinite(v) or (component != "opening" and v < 0) for v in values):
                    raise ValueError("Invalid endpoint errors")
        for group in ("all", f"task/{raw['task_index']}", f"seed/{raw['sampling_seed']}", raw["episode"]):
            for horizon in (8, 16, 40):
                for component in ("position", "rotation", "opening"):
                    entry = groups.setdefault(
                        f"{group}/h{horizon}/{component}", {"raw": [], "probe052": [], "improved_pairs": 0}
                    )
                    means = {}
                    for variant in ("raw", "probe052"):
                        values = pair[variant]["errors"]["wam"][component][:horizon]
                        transformed = [abs(v) for v in values] if component == "opening" else [v * v for v in values]
                        entry[variant].extend(transformed)
                        means[variant] = sum(transformed) / horizon
                    entry["improved_pairs"] += means["probe052"] < means["raw"]
    for key, entry in groups.items():
        entry["pairs"] = len(entry["raw"]) // int(key.split("/h")[-1].split("/")[0])
        for variant in ("raw", "probe052"):
            mean = sum(entry[variant]) / len(entry[variant])
            entry[variant] = mean if key.endswith("/opening") else math.sqrt(mean)
        entry["relative_change"] = entry["probe052"] / entry["raw"] - 1
    if args.score.read_bytes() != payload:
        raise ValueError("Input changed")
    result = {
        "complete": True,
        "score_sha256": hashlib.sha256(payload).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pairs": len(pairs),
        "groups": groups,
        "boundary": "Generated-reference error magnitudes; no causal mechanism, tracking or task-success claim",
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
