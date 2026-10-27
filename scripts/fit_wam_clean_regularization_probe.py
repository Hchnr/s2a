"""One separately registered CPU fit on exactly the former W2 feature/target stream."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from analyze_wam_clean_capacity import checked
from analyze_wam_release_horizon import reference

from s2a.clean_training_data import normalize_active10
from s2a.wam_clean_streaming import CleanFitBlock, CleanFitLayout, fit_clean_residual_streaming


def execute(spec, output):
    consumed = {}
    fit = json.loads(checked(spec["previous_fit_ref"], consumed).read_text())
    previous_params = json.loads(checked(fit["params_ref"], consumed).read_text())
    verified = json.loads(checked(spec["previous_verification_ref"], consumed).read_text())
    if (
        not verified["passed"]
        or not verified["complete"]
        or verified["fit_ref"] != spec["previous_fit_ref"]
        or verified["params_ref"] != fit["params_ref"]
    ):
        raise ValueError("Original independent W2 qualification differs")
    if not fit["passed"] or not fit["complete"] or fit["actual_rows"] != 840000:
        raise ValueError("Require completed original W2")
    prepared = json.loads(checked(fit["prepared_ref"], consumed).read_text())
    if spec["ridge"] != 0.1 / 1290 or previous_params["ridge"] != 0.1:
        raise ValueError("Only the fixed052 single ridge contrast is allowed")
    for ref in spec["source_refs"]:
        checked(ref, consumed)
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    if len(anchors) != 150:
        raise ValueError("Training anchor roster differs")
    layout = CleanFitLayout(tuple(fit["fit_metadata"]["feature_columns"]), "active10")
    labels, passes = {}, []

    def replay():
        digests = [hashlib.sha256() for _ in range(3)]
        rows = requests = 0
        for cref in fit["collection_refs"]:
            collection = json.loads(checked(cref, consumed).read_text())
            normal = json.loads(checked(collection["normalization_ref"], consumed).read_text())
            for request in collection["requests"]:
                anchor = anchors[request["anchor_name"]]
                name = anchor["name"]
                if name not in labels:
                    with np.load(checked(anchor["trajectory_ref"], consumed), allow_pickle=False) as z:
                        start = anchor["anchor_frame"]
                        physical = z["action128"][start : start + 40].copy()
                    labels[name] = normalize_active10(physical, **normal)["normalized_active10"].astype(np.float64)
                ledger = json.loads(checked(request["compact_ledger_ref"], consumed).read_text())
                if len(ledger["points"]) != 35:
                    raise ValueError("Missing denoise nodes")
                xx, yy = [], []
                for pref in ledger["points"]:
                    point = json.loads(checked(pref, consumed).read_text())
                    with np.load(checked(point["arrays_ref"], consumed), allow_pickle=False) as z:
                        c0 = z["c0_active10"]
                        xx.append(np.concatenate([z["hidden"].astype(np.float64), c0], axis=1))
                        yy.append(labels[name] - c0)
                features, target = np.concatenate(xx), np.concatenate(yy)
                weights = np.full(len(features), 1 / 840000)
                for digest, values in zip(digests, [features, target, weights], strict=True):
                    digest.update(np.asarray(values, dtype="<f8", order="C").tobytes())
                rows += len(features)
                requests += 1
                yield CleanFitBlock(features, target, weights, layout)
                if requests % 50 == 0:
                    print(json.dumps({"pass": len(passes) + 1, "requests": requests, "rows": rows}), flush=True)
        actual = dict(
            zip(["features", "target_clean_residual", "weights"], [v.hexdigest() for v in digests], strict=True)
        )
        if rows != 840000 or requests != 600 or actual != fit["fit_metadata"]["replay_sha256"]:
            raise ValueError("Full fit stream differs from original W2")
        passes.append(actual)

    fitted = fit_clean_residual_streaming(replay, layout=layout, ridge=spec["ridge"])
    if len(passes) != 2 or not np.allclose(fitted.readout.center, previous_params["center"], rtol=1e-12, atol=1e-12):
        raise ValueError("Normalization or replay changed")
    if not np.isclose(fitted.readout.scale, previous_params["scale"], rtol=1e-12, atol=1e-12):
        raise ValueError("Feature scale changed")
    for path, sha in consumed.items():
        if reference(Path(path))["sha256"] != sha:
            raise ValueError("Consumed input drifted")
    params_path = output / "parameters.json"
    with params_path.open("x") as stream:
        json.dump(fitted.readout.to_parameters(), stream, indent=2)
        stream.write("\n")
    result = {
        "complete": True,
        "passed": True,
        "schema_version": "s2a.clean-regularization-probe.v1",
        "producer_fits": 1,
        "independent_numerical_qualification": False,
        "registration_ref": reference(output / "registration.json"),
        "params_ref": reference(params_path),
        "fit_metadata": fitted.fit_metadata(),
        "previous_fit_ref": spec["previous_fit_ref"],
        "input_sha256": consumed,
        "inputs_unchanged_after": True,
        "future_training_windows_decoded": len(labels),
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
        "deployment_authorized": False,
        "fresh_authorized": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"complete": True, "producer_fits": 1, "qualified": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(k) != "" for k in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]):
        raise ValueError("Explicit empty CUDA/LD required")
    if any(os.environ.get(k) != "8" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]):
        raise ValueError("Explicit CPU thread limits8 required")
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root or not args.registration.resolve().is_relative_to(root / "outputs"):
        raise ValueError("Requires repository cwd and outputs registration")
    spec = json.loads(args.registration.read_text())
    if spec["entrypoint_ref"] != reference(Path(__file__)) or spec["producer_fits"] != 1:
        raise ValueError("Registration source or fit budget changed")
    output = args.registration.resolve().parent
    with (output / "attempt.json").open("x") as stream:
        json.dump(
            {
                "started": True,
                "producer_fit_attempt": 1,
                "pid": os.getpid(),
                "registration_ref": reference(args.registration),
            },
            stream,
        )
    try:
        execute(spec, output)
    except BaseException as error:
        with (output / "failure.json").open("x") as stream:
            json.dump({"passed": False, "producer_fit_attempt": 1, "error": repr(error)}, stream)
        raise


if __name__ == "__main__":
    main()
