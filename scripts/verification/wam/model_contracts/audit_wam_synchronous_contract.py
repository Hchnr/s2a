"""Audit saved normalization/feature contracts and teacher error by noise step.

This consumes existing receipts, not raw training rows or future trajectories.
It does not repeat the independent numerical verification of model captures.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    consumed = {}

    def read(path, expected=None):
        path = Path(path)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"SHA mismatch: {path}")
        if str(path) in consumed and consumed[str(path)] != digest:
            raise ValueError(f"Input changed: {path}")
        consumed[str(path)] = digest
        return json.loads(payload)

    def checked(ref):
        return read(ref["path"], ref["sha256"])

    probe, teacher, sampling = read(args.probe), read(args.teacher), read(args.sampling)
    if not all(x["complete"] and x["passed"] for x in (probe, teacher, sampling)):
        raise ValueError("Completed inputs required")
    if teacher["probe_ref"]["sha256"] != consumed[str(args.probe)]:
        raise ValueError("Teacher comparison uses another probe")
    if sampling["readout_ref"] != probe["params_ref"]:
        raise ValueError("Sampling uses another head")
    params = checked(probe["params_ref"])
    fit = checked(probe["previous_fit_ref"])
    expected_columns = [f"hidden_{i:04d}" for i in range(1280)] + [f"c0_{i:02d}" for i in range(10)]
    if fit["fit_metadata"]["feature_columns"] != expected_columns:
        raise ValueError("Training feature order differs")
    if len(params["center"]) != 1290 or len(params["coefficients"]) != 1291:
        raise ValueError("Head feature dimensions differ")
    checked(sampling["normalization_ref"])
    rounds = []
    for ref in fit["collection_refs"]:
        collection = checked(ref)
        checked(collection["normalization_ref"])
        if collection["normalization_ref"]["sha256"] != sampling["normalization_ref"]["sha256"]:
            raise ValueError("Train/deployment normalization differs")
        rounds.append(collection["round_name"])
    nodes = 0
    for generation in sampling["generations"]:
        ledger = checked(generation["compact_ledger_ref"])
        if not ledger["complete"] or ledger["nodes"] != 35:
            raise ValueError("Incomplete generation ledger")
        for index, ref in enumerate(ledger["points"]):
            point = checked(ref)
            metadata = point["metadata"]
            if metadata["feature_terms"] != [] or metadata["step_index"] != index:
                raise ValueError("Deployment feature contract differs")
            nodes += 1
    if len(sampling["generations"]) != 95 or nodes != 3325:
        raise ValueError("055 continuation denominator differs")
    steps = {}
    for index in range(35):
        entry = teacher["training"][f"probe/step/{index}"]
        if entry["rows"] != 24000:
            raise ValueError("Teacher step denominator differs")
        base, head = entry["base_mse_active10"], entry["head_mse_active10"]
        steps[str(index)] = {
            component: 1 - sum(head[start:stop]) / sum(base[start:stop])
            for component, start, stop in (("xyz", 0, 3), ("rot6d", 3, 9), ("opening", 9, 10))
        }
    for path, digest in consumed.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input changed: {path}")
    result = {
        "complete": True,
        "passed": True,
        "normalization_sha256": sampling["normalization_ref"]["sha256"],
        "training_rounds": rounds,
        "deployment_nodes_checked": nodes,
        "feature_columns": expected_columns,
        "teacher_relative_mse_reduction_by_step": steps,
        "all_steps_all_components_improved": all(v > 0 for row in steps.values() for v in row.values()),
        "input_sha256": consumed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary": "Receipt/metadata audit and cached teacher summary; not a new raw numerical or causal proof",
        "new_fits": 0,
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
