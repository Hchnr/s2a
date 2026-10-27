"""Read all registered W2 features to measure ridge spectrum without labels or fit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from analyze_wam_release_horizon import reference


def checked(ref, consumed):
    path = Path(ref["path"])
    if reference(path) != ref:
        raise ValueError(f"Changed input: {path}")
    consumed[path.as_posix()] = ref["sha256"]
    return path


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("Requires explicitly empty CUDA and LD")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fit = json.loads(args.fit.read_text())
    consumed = {str(args.fit): reference(args.fit)["sha256"]}
    params = json.loads(checked(fit["params_ref"], consumed).read_text())
    center, scale = np.asarray(params["center"]), params["scale"]
    dimensions = len(center)
    rows = fit["actual_rows"]
    gram = np.zeros((dimensions, dimensions))
    mean = np.zeros(dimensions)
    digest = hashlib.sha256()
    count = requests = 0
    for collection_ref in fit["collection_refs"]:
        collection = json.loads(checked(collection_ref, consumed).read_text())
        for request in collection["requests"]:
            ledger = json.loads(checked(request["compact_ledger_ref"], consumed).read_text())
            blocks = []
            for point_ref in ledger["points"]:
                point = json.loads(checked(point_ref, consumed).read_text())
                with np.load(checked(point["arrays_ref"], consumed), allow_pickle=False) as z:
                    features = np.concatenate([z["hidden"].astype(np.float64), z["c0_active10"]], axis=1)
                    if features.shape != (40, dimensions) or not np.isfinite(features).all():
                        raise ValueError("Feature shape or finiteness changed")
                    blocks.append(features)
            features = np.concatenate(blocks)
            digest.update(np.asarray(features, dtype="<f8", order="C").tobytes())
            normalized = (features - center) / scale
            gram += (normalized.T @ normalized) / rows
            mean += normalized.sum(axis=0) / rows
            count += len(features)
            requests += 1
            if requests % 25 == 0:
                print(json.dumps({"requests": requests, "rows": count}), flush=True)
    expected = fit["fit_metadata"]["replay_sha256"]["features"]
    if digest.hexdigest() != expected or count != rows:
        raise ValueError("Full feature replay differs from original W2")
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues.min() < -1e-10 or abs(np.trace(gram) - 1) > 1e-8:
        raise ValueError("Normalized covariance invariant failed")
    eigenvalues = np.maximum(eigenvalues, 0)
    ridge = params["ridge"]
    capacity = float(np.sum(eigenvalues / (eigenvalues + ridge)))
    result = {
        "complete": True,
        "passed": True,
        "scope": "Exact registered feature replay and ridge spectrum; no target labels, refit, model, physics or fresh",
        "script_ref": reference(Path(__file__)),
        "helper_ref": reference(Path(__file__).with_name("analyze_wam_release_horizon.py")),
        "fit_ref": reference(args.fit),
        "feature_replay_sha256": digest.hexdigest(),
        "rows": count,
        "requests": requests,
        "dimensions": dimensions,
        "ridge": ridge,
        "covariance_trace": float(np.trace(gram)),
        "normalized_mean_norm": float(np.linalg.norm(mean)),
        "eigenvalues_ascending": eigenvalues.tolist(),
        "feature_effective_degrees_of_freedom": capacity,
        "top10_eigenvalues_descending": eigenvalues[-10:][::-1].tolist(),
        "source_inputs_sha256": consumed,
        "limitations": [
            "Capacity alone does not prove underfit or task causation",
            "No alternative ridge selected or fitted",
        ],
    }
    # This diagnostic retains no fitted parameters and never edits a source artifact.
    for name, sha in consumed.items():
        if reference(Path(name))["sha256"] != sha:
            raise ValueError(f"Input changed during diagnosis: {name}")
    result["inputs_unchanged_after"] = True
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"complete": True, "feature_effective_degrees_of_freedom": capacity}), flush=True)


if __name__ == "__main__":
    main()
