"""Offline terminal opening losses on qualified D0/D1/D2 training caches."""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_aggregation_candidate import candidate_head
from s2a.clean_training_aggregation import admit_aggregation
from s2a.clean_training_data import normalize_active10
from s2a.clean_training_runtime import publish, reference


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/clean_on_policy_aggregation_20261024/aggregate_fit")
    output = root / "terminal_opening_diagnosis.json"
    require(not output.exists(), "Immutable output already exists")
    e = Evidence()
    spec = e.json(reference(root / "registration.json"))
    _, prepared, _, collections = admit_aggregation(spec, read_json=e.json)
    head, params_ref = candidate_head(reference(root / "orchestration/result.json"), read_json=e.json)
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    labels, groups = {}, defaultdict(list)
    requests = 0
    for round_index, collection in enumerate(collections):
        normal = e.json(collection["normalization_ref"])
        for request in collection["requests"]:
            anchor = anchors[request["anchor_name"]]
            if anchor["name"] not in labels:
                with np.load(e.path(anchor["trajectory_ref"]), allow_pickle=False) as z:
                    start = anchor["anchor_frame"]
                    physical = z["action128"][start : start + 40].copy()
                labels[anchor["name"]] = physical
            physical = labels[anchor["name"]]
            truth = normalize_active10(physical, **normal)["normalized_active10"].astype(np.float64)
            ledger = e.json(request["compact_ledger_ref"])
            require(ledger["complete"] and len(ledger["points"]) == 35, "Full ledger required")
            point = e.json(ledger["points"][-1])
            require(
                point["metadata"]["step_index"] == 34 and not point["metadata"]["feature_terms"], "Terminal contract"
            )
            with np.load(e.path(point["arrays_ref"]), allow_pickle=False) as z:
                raw = z["sample_active10_fp32"].astype(np.float64) - point["metadata"]["sigma_actual"] * z[
                    "raw_velocity_active10"
                ].astype(np.float64)
                require(np.array_equal(raw, z["c0_active10"]), "Raw clean reconstruction differs")
                features = np.column_stack((z["hidden"].astype(np.float64), raw))
            corrected = raw + head.predict_clean_residual(features)
            require(np.isfinite(corrected).all(), "Nonfinite prediction")
            transform = normal["transform"]
            for space in ("normalized", "ideal_physical_clipped"):
                a, b, target = raw[:, 9], corrected[:, 9], truth[:, 9]
                if space == "ideal_physical_clipped":
                    if transform["normalization_mask"][9]:
                        scale, bias = transform["scale"][9], transform["bias"][9]
                        require(scale != 0, "Noninvertible opening normalization")
                        a, b = (a - bias) / scale, (b - bias) / scale
                    a, b, target = np.clip(a, 0, 1), np.clip(b, 0, 1), physical[:, 9]
                for phase, mask in (
                    ("all", np.ones(40, dtype=bool)),
                    ("closed", physical[:, 9] <= 0.1),
                    ("open", physical[:, 9] >= 0.9),
                    ("intermediate", (physical[:, 9] > 0.1) & (physical[:, 9] < 0.9)),
                ):
                    for group in ("all", f"D{round_index}"):
                        if mask.any():
                            groups[f"{group}/{space}/{phase}"].append(np.column_stack((a - target, b - target))[mask])
            requests += 1
        print(f"D{round_index}: {requests} requests", flush=True)
    require(requests == 900 and len(labels) == 150, "Full terminal denominator differs")
    results = {}
    for name, blocks in groups.items():
        errors = np.concatenate(blocks)
        results[name] = dict(
            rows=len(errors),
            raw=dict(
                mse=float(np.mean(errors[:, 0] ** 2)),
                mae=float(np.mean(abs(errors[:, 0]))),
                bias=float(np.mean(errors[:, 0])),
            ),
            corrected=dict(
                mse=float(np.mean(errors[:, 1] ** 2)),
                mae=float(np.mean(abs(errors[:, 1]))),
                bias=float(np.mean(errors[:, 1])),
            ),
        )
    e.finish()
    publish(
        output,
        dict(
            passed=True,
            complete=True,
            requests=requests,
            terminal_rows=36000,
            training_windows=len(labels),
            params_ref=params_ref,
            groups=results,
            input_sha256=e.inputs,
            source_ref=reference(Path(__file__)),
            boundary="Offline counterfactual head on saved histories; ideal FP64 inverse normalization is not deployed scheduler output. Truth bins are diagnostic only. No validation/fresh supervision.",
            new_fits=0,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
