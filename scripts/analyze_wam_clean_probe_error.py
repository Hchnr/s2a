"""Compare old/new heads on fixed cached rows; this is not new-policy sampling."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_wam_clean_capacity import checked
from analyze_wam_clean_residual_error import add, bucket, finish
from analyze_wam_release_horizon import reference

from s2a.clean_training_data import normalize_active10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(os.environ.get(k) != "" for k in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]):
        raise ValueError("CPU environment required")
    consumed = {str(args.probe): reference(args.probe)["sha256"]}
    produced = json.loads(args.probe.read_text())
    new = json.loads(checked(produced["params_ref"], consumed).read_text())
    fit = json.loads(checked(produced["previous_fit_ref"], consumed).read_text())
    old = json.loads(checked(fit["params_ref"], consumed).read_text())
    prepared = json.loads(checked(fit["prepared_ref"], consumed).read_text())
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    heads = {}
    for label, params in [("old", old), ("probe", new)]:
        heads[label] = (np.asarray(params["coefficients"]), np.asarray(params["center"]), params["scale"])
    labels, stats = {}, defaultdict(bucket)
    requests = 0
    import hashlib

    digests = [hashlib.sha256(), hashlib.sha256()]
    for cref in fit["collection_refs"]:
        collection = json.loads(checked(cref, consumed).read_text())
        normal = json.loads(checked(collection["normalization_ref"], consumed).read_text())
        for request in collection["requests"]:
            anchor = anchors[request["anchor_name"]]
            if anchor["name"] not in labels:
                with np.load(checked(anchor["trajectory_ref"], consumed), allow_pickle=False) as z:
                    start = anchor["anchor_frame"]
                    action = z["action128"][start : start + 40]
                labels[anchor["name"]] = normalize_active10(action, **normal)["normalized_active10"].astype(np.float64)
            ledger = json.loads(checked(request["compact_ledger_ref"], consumed).read_text())
            for step, pref in enumerate(ledger["points"]):
                point = json.loads(checked(pref, consumed).read_text())
                with np.load(checked(point["arrays_ref"], consumed), allow_pickle=False) as z:
                    c0 = z["c0_active10"]
                    features = np.column_stack([z["hidden"].astype(np.float64), c0])
                target = labels[anchor["name"]] - c0
                for digest, value in zip(digests, [features, target], strict=True):
                    digest.update(np.asarray(value, dtype="<f8", order="C").tobytes())
                for label, (coeff, center, scale) in heads.items():
                    predicted = ((features - center) / scale) @ coeff[:-1] + coeff[-1]
                    error = target - predicted
                    for key in [
                        "all",
                        "task/" + str(anchor["source_identity"]["task_index"]),
                        "round/" + collection["round_name"],
                        "step/" + str(step),
                    ]:
                        add(stats[label + "/" + key], target, error)
            requests += 1
            if requests % 50 == 0:
                print(json.dumps({"requests": requests}), flush=True)
    for key, digest in zip(["features", "target_clean_residual"], digests, strict=True):
        if digest.hexdigest() != fit["fit_metadata"]["replay_sha256"][key]:
            raise ValueError("Comparison row stream changed")
    if stats["probe/all"]["rows"] != 840000:
        raise ValueError("Comparison denominator changed")
    for path, sha in consumed.items():
        if reference(Path(path))["sha256"] != sha:
            raise ValueError("Comparison inputs changed")
    result = {
        "complete": True,
        "passed": True,
        "probe_ref": reference(args.probe),
        "scope": "Teacher comparison on same off-policy D0/D1 rows; not independent validation or deployed sampling",
        "script_ref": reference(Path(__file__)),
        "source_refs": [
            reference(Path(__file__).with_name(p))
            for p in [
                "analyze_wam_clean_residual_error.py",
                "analyze_wam_clean_capacity.py",
                "analyze_wam_release_horizon.py",
            ]
        ],
        "normalization_source_ref": reference(Path("src/s2a/clean_training_data.py")),
        "training": {k: finish(v) for k, v in stats.items()},
        "input_sha256": consumed,
        "inputs_unchanged_after": True,
        "new_fits": 0,
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result["training"][k] for k in ["old/all", "probe/all"]}), flush=True)


if __name__ == "__main__":
    main()
