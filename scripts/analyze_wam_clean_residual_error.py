"""Evaluate the existing W2 head on frozen training captures; never fit parameters."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_wam_clean_capacity import checked
from analyze_wam_release_horizon import reference

from s2a.clean_training_data import normalize_active10


def bucket():
    return {"rows": 0, "base_sse": np.zeros(10), "head_sse": np.zeros(10), "target_sum": np.zeros(10)}


def add(state, target, error):
    state["rows"] += len(target)
    state["base_sse"] += np.sum(target**2, axis=0)
    state["head_sse"] += np.sum(error**2, axis=0)
    state["target_sum"] += target.sum(axis=0)


def finish(state):
    n = state["rows"]
    return {
        "rows": n,
        "base_mse_active10": (state["base_sse"] / n).tolist(),
        "head_mse_active10": (state["head_sse"] / n).tolist(),
        "target_mean_active10": (state["target_sum"] / n).tolist(),
    }


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("Requires explicit CPU environment")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fit = json.loads(args.fit.read_text())
    consumed = {str(args.fit): reference(args.fit)["sha256"], str(args.state): reference(args.state)["sha256"]}
    params = json.loads(checked(fit["params_ref"], consumed).read_text())
    prepared = json.loads(checked(fit["prepared_ref"], consumed).read_text())
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    coefficients, center = np.asarray(params["coefficients"]), np.asarray(params["center"])
    labels = {}
    groups = defaultdict(bucket)
    feature_digest, target_digest = hashlib.sha256(), hashlib.sha256()
    requests = 0
    for cref in fit["collection_refs"]:
        collection = json.loads(checked(cref, consumed).read_text())
        normal = json.loads(checked(collection["normalization_ref"], consumed).read_text())
        for request in collection["requests"]:
            anchor = anchors[request["anchor_name"]]
            name = anchor["name"]
            if name not in labels:
                path = checked(anchor["trajectory_ref"], consumed)
                with np.load(path, allow_pickle=False) as archive:
                    start = anchor["anchor_frame"]
                    physical = archive["action128"][start : start + 40].copy()
                labels[name] = normalize_active10(physical, **normal)["normalized_active10"].astype(np.float64)
            x0 = labels[name]
            ledger = json.loads(checked(request["compact_ledger_ref"], consumed).read_text())
            if len(ledger["points"]) != 35:
                raise ValueError("Incomplete capture")
            for step, pref in enumerate(ledger["points"]):
                point = json.loads(checked(pref, consumed).read_text())
                with np.load(checked(point["arrays_ref"], consumed), allow_pickle=False) as z:
                    c0 = z["c0_active10"]
                    features = np.concatenate([z["hidden"].astype(np.float64), c0], axis=1)
                    target = x0 - c0
                feature_digest.update(np.asarray(features, dtype="<f8", order="C").tobytes())
                target_digest.update(np.asarray(target, dtype="<f8", order="C").tobytes())
                prediction = ((features - center) / params["scale"]) @ coefficients[:-1] + coefficients[-1]
                error = target - prediction
                for key in [
                    "all",
                    "round/" + collection["round_name"],
                    "task/" + str(anchor["source_identity"]["task_index"]),
                    "step/" + str(step),
                ]:
                    add(groups[key], target, error)
            requests += 1
            if requests % 50 == 0:
                print(json.dumps({"requests": requests, "rows": groups["all"]["rows"]}), flush=True)
    hashes = {"features": feature_digest.hexdigest(), "target_clean_residual": target_digest.hexdigest()}
    if any(value != fit["fit_metadata"]["replay_sha256"][key] for key, value in hashes.items()):
        raise ValueError("Feature or target replay differs from original fit")
    if groups["all"]["rows"] != fit["actual_rows"]:
        raise ValueError("Training denominator changed")
    state = json.loads(args.state.read_text())
    validation = {}
    for task in range(3):
        for seed in [42, 43]:
            for horizon in [8, 16, 40]:
                variants = {}
                for variant in ["raw", "w2"]:
                    rows = [
                        r
                        for r in state["state_rows"]
                        if r["task_index"] == task and r["sampling_seed"] == seed and r["variant"] == variant
                    ]
                    if len(rows) != 8:
                        raise ValueError("Validation group denominator changed")
                    errors = {
                        k: np.concatenate([r["errors"]["wam"][k][:horizon] for r in rows])
                        for k in ["position", "rotation", "opening"]
                    }
                    variants[variant] = {
                        "valid_endpoints": len(errors["opening"]),
                        "position_rmse_m": float(np.sqrt(np.mean(errors["position"] ** 2))),
                        "rotation_rmse_rad": float(np.sqrt(np.mean(errors["rotation"] ** 2))),
                        "opening_mae": float(np.mean(np.abs(errors["opening"]))),
                    }
                validation[f"task{task}/seed{seed}/h{horizon}"] = variants
    for path, sha in consumed.items():
        if reference(Path(path))["sha256"] != sha:
            raise ValueError("Consumed input changed")
    result = {
        "complete": True,
        "passed": True,
        "scope": "Training teacher clean-residual errors plus archived validation state errors; no refit or new inference",
        "script_ref": reference(Path(__file__)),
        "fit_ref": reference(args.fit),
        "state_ref": reference(args.state),
        "helper_refs": [
            reference(Path(__file__).with_name(p))
            for p in ["analyze_wam_clean_capacity.py", "analyze_wam_release_horizon.py"]
        ],
        "label_normalization_ref": reference(Path("src/s2a/clean_training_data.py")),
        "replay_sha256": hashes,
        "training": {k: finish(v) for k, v in groups.items()},
        "validation": validation,
        "decoded_training_windows": len(labels),
        "inputs_unchanged_after": True,
        "input_sha256": consumed,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"complete": True, "all": result["training"]["all"]}), flush=True)


if __name__ == "__main__":
    main()
