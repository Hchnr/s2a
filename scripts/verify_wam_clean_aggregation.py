"""Independent label/raw-C0 reconstruction and full aggregate QR for056."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from analyze_wam_clean_capacity import checked
from analyze_wam_release_horizon import reference

from s2a.clean_training_aggregation import ROWS, admit_aggregation
from s2a.streaming_ridge_qr import independent_ridge


def verify(spec, output):
    if (
        spec["schema_version"] != "s2a.clean-aggregation-audit-registration.v1"
        or spec["independent_reconstructions"] != 1
        or spec["automatic_retry"] is not False
        or spec["rtol"] != 1e-9
        or spec["atol"] != 1e-11
    ):
        raise ValueError("056 fixed independent QR policy differs")
    consumed = {}
    for ref in spec["source_refs"]:
        checked(ref, consumed)
    produced = json.loads(checked(spec["producer_ref"], consumed).read_text())
    if (
        produced["schema_version"] != "s2a.clean-aggregation-fit.v1"
        or not produced["passed"]
        or not produced["complete"]
        or produced["producer_fits"] != 1
        or produced["fit_metadata"]["rows"] != ROWS
    ):
        raise ValueError("Complete056 aggregate producer required")
    old_fit = json.loads(checked(produced["previous_fit_ref"], consumed).read_text())
    params = json.loads(checked(produced["params_ref"], consumed).read_text())
    registration = json.loads(checked(produced["registration_ref"], consumed).read_text())
    if params["ridge"] != 0.1 / 1290 or registration["ridge"] != params["ridge"] or registration["rows"] != ROWS:
        raise ValueError("Fixed056 ridge changed")
    _, _, expected_refs, _ = admit_aggregation(
        registration, read_json=lambda ref: json.loads(checked(ref, consumed).read_text())
    )
    if produced["collection_refs"] != expected_refs:
        raise ValueError("056 producer round references differ from qualified aggregation")
    prepared = json.loads(checked(produced["training_prepared_ref"], consumed).read_text())
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    labels, pass_count = {}, []

    def blocks():
        requests = 0
        for cref in produced["collection_refs"]:
            collection = json.loads(checked(cref, consumed).read_text())
            normal = json.loads(checked(collection["normalization_ref"], consumed).read_text())
            if normal["normalization_id"] != 31 or normal["projection_id"] != 0:
                raise ValueError("Normalization identity changed")
            transform = normal["transform"]
            mask = np.asarray(transform["normalization_mask"], dtype=bool)
            scale = np.asarray(transform["scale"], dtype=np.float32)
            bias = np.asarray(transform["bias"], dtype=np.float32)
            for request in collection["requests"]:
                anchor = anchors[request["anchor_name"]]
                if anchor["name"] not in labels:
                    with np.load(checked(anchor["trajectory_ref"], consumed), allow_pickle=False) as z:
                        start = anchor["anchor_frame"]
                        value = z["action128"][start : start + 40].astype(np.float32)
                    if value.shape != (40, 128) or np.any(value[:, 10:]):
                        raise ValueError("Wrong label layout")
                    multiplied = np.multiply(value[:, mask], scale[mask], dtype=np.float32)
                    shifted = np.add(multiplied, bias[mask], dtype=np.float32)
                    value[:, mask] = np.clip(
                        shifted, np.float32(transform["clip_min"]), np.float32(transform["clip_max"])
                    )
                    labels[anchor["name"]] = value[:, :10].astype(np.float64)
                ledger = json.loads(checked(request["compact_ledger_ref"], consumed).read_text())
                if len(ledger["points"]) != 35:
                    raise ValueError("Missing denoise nodes")
                features, targets = [], []
                for step, pref in enumerate(ledger["points"]):
                    point = json.loads(checked(pref, consumed).read_text())
                    sigma = point["metadata"]["sigma_actual"]
                    if point["metadata"]["step_index"] != step or sigma <= 0:
                        raise ValueError("Denoise time identity changed")
                    with np.load(checked(point["arrays_ref"], consumed), allow_pickle=False) as z:
                        raw_c0 = z["sample_active10_fp32"].astype(np.float64) - sigma * z[
                            "raw_velocity_active10"
                        ].astype(np.float64)
                        if not np.array_equal(raw_c0, z["c0_active10"]):
                            raise ValueError("C0 differs from actual raw sampler capture")
                        feature = np.column_stack([z["hidden"].astype(np.float64), raw_c0])
                    features.append(feature)
                    targets.append(labels[anchor["name"]] - raw_c0)
                yield np.vstack(features), np.vstack(targets), np.ones(1400) / ROWS
                requests += 1
                if requests % 50 == 0:
                    print(json.dumps({"qr_pass": len(pass_count) + 1, "requests": requests}), flush=True)
        if requests != 900:
            raise ValueError("Independent request denominator changed")
        pass_count.append(requests)

    qr = independent_ridge(
        blocks,
        expected_rows=ROWS,
        feature_columns=tuple(old_fit["fit_metadata"]["feature_columns"]),
        target_columns=10,
        ridge=params["ridge"],
        max_block_rows=1400,
    )
    old_hashes = produced["fit_metadata"]["replay_sha256"]
    if tuple(old_hashes[k] for k in ["features", "target_clean_residual", "weights"]) != qr.replay_sha256:
        raise ValueError("Independent aggregate source stream differs from producer")
    comparisons = {}
    for name, expected in [("coefficients", qr.coefficients), ("center", qr.center), ("scale", np.asarray(qr.scale))]:
        actual = np.asarray(params[name])
        comparisons[name] = {
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "passed": bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-11)),
        }
    if not all(v["passed"] for v in comparisons.values()):
        raise ValueError(f"Independent QR mismatch: {comparisons}")
    for path, sha in consumed.items():
        if reference(Path(path))["sha256"] != sha:
            raise ValueError("Independent consumed input changed")
    result = {
        "complete": True,
        "passed": True,
        "schema_version": "s2a.clean-aggregation-qr.v1",
        "auditor_registration_ref": reference(output / "audit_registration.json"),
        "producer_ref": spec["producer_ref"],
        "params_ref": produced["params_ref"],
        "collection_refs": produced["collection_refs"],
        "d2_owner_ref": produced["d2_owner_ref"],
        "comparisons": comparisons,
        "rtol": 1e-9,
        "atol": 1e-11,
        "independent_reconstructions": 1,
        "qr_metadata": qr.metadata(),
        "training_windows_decoded": len(labels),
        "input_sha256": consumed,
        "inputs_unchanged_after": True,
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
        "deployment_authorized": False,
    }
    with (output / "independent_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"complete": True, "independent_qr_passed": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(k) != "" for k in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]):
        raise ValueError("Explicit empty CUDA/LD required")
    if any(os.environ.get(k) != "8" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]):
        raise ValueError("Explicit CPU thread limits8 required")
    spec = json.loads(args.registration.read_text())
    if spec["entrypoint_ref"] != reference(Path(__file__)):
        raise ValueError("Independent source changed")
    output = args.registration.resolve().parent
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root or not output.is_relative_to(root / "outputs"):
        raise ValueError("Requires repository cwd and outputs root")
    with (output / "audit_attempt.json").open("x") as stream:
        json.dump(
            {"independent_attempt": 1, "pid": os.getpid(), "registration_ref": reference(args.registration)}, stream
        )
    try:
        verify(spec, output)
    except BaseException as error:
        with (output / "audit_failure.json").open("x") as stream:
            json.dump({"passed": False, "error": repr(error)}, stream)
        raise


if __name__ == "__main__":
    main()
