"""Compare cached training and synchronous generation features without labels or fit."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(k) != "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")):
        raise ValueError("Explicit CPU environment required")
    if args.output.exists():
        raise FileExistsError(args.output)
    consumed = {}

    def checked(ref):
        path = Path(ref["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ref["sha256"]:
            raise ValueError(f"Input SHA differs: {path}")
        consumed[str(path)] = digest
        return path

    def load(path):
        path = Path(path)
        payload = path.read_bytes()
        consumed[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    probe = load(args.probe)
    params = load(checked(probe["params_ref"]))
    fit = load(checked(probe["previous_fit_ref"]))
    center, scale, coeff = np.asarray(params["center"]), params["scale"], np.asarray(params["coefficients"])
    stats = defaultdict(
        lambda: {"rows": 0, "norm": [], "hidden_norm": [], "c0_sum": np.zeros(10), "residual_sq": np.zeros(10)}
    )
    digest = hashlib.sha256()

    def add(requests, label, training=False):
        for count, request in enumerate(requests):
            ledger = load(checked(request["compact_ledger_ref"]))
            if not ledger["complete"] or len(ledger["points"]) != 35:
                raise ValueError("Expected complete 35-node ledger")
            for step, ref in enumerate(ledger["points"]):
                point = load(checked(ref))
                if point["metadata"]["step_index"] != step or point["metadata"]["feature_terms"]:
                    raise ValueError("Feature contract differs")
                with np.load(checked(point["arrays_ref"]), allow_pickle=False) as z:
                    x = np.column_stack((z["hidden"].astype(np.float64), z["c0_active10"]))
                if x.shape != (40, 1290) or not np.isfinite(x).all():
                    raise ValueError("Feature shape/finiteness differs")
                if training:
                    digest.update(np.asarray(x, dtype="<f8", order="C").tobytes())
                normalized = (x - center) / scale
                prediction = normalized @ coeff[:-1] + coeff[-1]
                labels = [label, "training"] if training else [request["variant"]]
                for group in labels:
                    stat = stats[f"{group}/step/{step}"]
                    stat["rows"] += 40
                    stat["norm"].extend(np.linalg.norm(normalized, axis=1).tolist())
                    stat["hidden_norm"].extend(np.linalg.norm(normalized[:, :1280], axis=1).tolist())
                    stat["c0_sum"] += x[:, 1280:].sum(axis=0)
                    stat["residual_sq"] += (prediction**2).sum(axis=0)
            if (count + 1) % 50 == 0:
                print(json.dumps({"group": label, "requests": count + 1}), flush=True)

    for ref in fit["collection_refs"]:
        collection = load(checked(ref))
        add(collection["requests"], collection["round_name"], training=True)
    if digest.hexdigest() != fit["fit_metadata"]["replay_sha256"]["features"]:
        raise ValueError("Training stream differs")
    generations = []
    for path in args.sampling:
        sampling = load(path)
        generations.extend(sampling["generations"])
    generations = [g for g in generations if g["variant"] in ("raw", "probe052") and g["complete"]]
    if len(generations) != 96 or len({g["name"] for g in generations}) != 96:
        raise ValueError("Expected 48 distinct raw/probe052 pairs")
    add(generations, "generation")
    result = {}
    for key, stat in stats.items():
        step = key.split("/step/")[-1]
        threshold = float(np.quantile(stats[f"training/step/{step}"]["norm"], 0.99))
        result[key] = {
            "rows": stat["rows"],
            "normalized_feature_norm_quantiles_50_95_99": np.quantile(stat["norm"], [0.5, 0.95, 0.99]).tolist(),
            "normalized_hidden_norm_quantiles_50_95_99": np.quantile(stat["hidden_norm"], [0.5, 0.95, 0.99]).tolist(),
            "fraction_above_training_step_norm_q99": float(np.mean(np.asarray(stat["norm"]) > threshold)),
            "raw_c0_mean_active10": (stat["c0_sum"] / stat["rows"]).tolist(),
            "probe_predicted_residual_rms_active10": np.sqrt(stat["residual_sq"] / stat["rows"]).tolist(),
        }
    for path, sha in consumed.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Input changed: {path}")
    output = {
        "complete": True,
        "passed": True,
        "training_rows": sum(stats[f"training/step/{i}"]["rows"] for i in range(35)),
        "generation_requests": len(generations),
        "training_feature_sha256": digest.hexdigest(),
        "groups": result,
        "input_sha256": consumed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary": "Feature norm and predicted correction distribution; not a density test, target error or causal attribution. Raw predictions are offline counterfactuals, not deployed corrections.",
        "future_labels_read": 0,
        "new_fits": 0,
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
    }
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
