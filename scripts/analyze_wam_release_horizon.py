"""Compare executed aperture prefixes with already generated, unused suffixes."""

import argparse
import hashlib
import json
from pathlib import Path


def reference(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def analyze(path):
    chunks = json.loads(path.read_text())
    steps_path = path.with_name("steps.jsonl")
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    records = []
    for i, chunk in enumerate(chunks):
        openings = chunk["openings"]
        declared_count = chunk["execution_endpoint_count"]
        start = chunk["step"]
        stop = chunks[i + 1]["step"] if i + 1 < len(chunks) else len(steps)
        executed = steps[start:stop]
        count = len(executed)
        if not 0 < count < len(openings) or declared_count != len(openings):
            raise ValueError("Expected complete executed prefix and unused generated suffix")
        if any(row["chunk_index"] != i for row in executed):
            raise ValueError("Chunk/endpoint alignment changed")
        prefix, suffix = openings[:count], openings[count:]
        records.append(
            {
                "chunk": i,
                "step": start,
                "anchor_opening": chunk["anchor_opening"],
                "executed_prefix_count": count,
                "declared_execution_endpoint_count": declared_count,
                "generated_count": len(openings),
                "prefix_max": max(prefix),
                "suffix_max": max(suffix),
                "suffix_minus_prefix_max": max(suffix) - max(prefix),
                "first_generated_endpoint_ge_0_9": next((j for j, x in enumerate(openings) if x >= 0.9), None),
                "actual_executed_opening_max": max(row["opening_target"] for row in executed),
            }
        )
    summaries = {}
    for label, rows in [("all", records), ("last5_chunks", records[-5:])]:
        summaries[label] = {
            "chunks": len(rows),
            "suffix_only_ge_0_9": sum(r["prefix_max"] < 0.9 <= r["suffix_max"] for r in rows),
            "suffix_rise_ge_0_04": sum(r["suffix_minus_prefix_max"] >= 0.04 for r in rows),
            "max_generated_opening": max(max(r["prefix_max"], r["suffix_max"]) for r in rows),
        }
    return {
        "case": path.parents[3].name,
        "chunks_ref": reference(path),
        "steps_ref": reference(steps_path),
        "summary": summaries,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.cases.glob("*/simulation/*/*/chunks.json"))
    if len(paths) != 12:
        raise ValueError("Requires the complete fixed 12-case development comparison")
    result = {
        "scope": "Offline generated-prefix/suffix description; thresholds are diagnostic, not acceptance or online triggers",
        "script_ref": reference(Path(__file__)),
        "cases": [analyze(path) for path in paths],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for case in result["cases"]:
        print(case["case"], case["summary"])


if __name__ == "__main__":
    main()
