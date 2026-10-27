"""Prepare only qualified D2 terminal training rows for plan057; no fitting."""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_aggregation import admit_aggregation
from s2a.clean_training_runtime import publish, reference
from s2a.geometry import rot6d_to_matrix
from s2a.wam_clean_readout import CleanResidualReadout


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    output = Path("outputs/terminal_trajectory_objective_20261025")
    require(not output.exists(), "New namespace required")
    e = Evidence()
    registration = reference(Path("outputs/clean_on_policy_aggregation_20261024/aggregate_fit/registration.json"))
    _, prepared, refs, collections = admit_aggregation(e.json(registration), read_json=e.json)
    collection = collections[2]
    require(len(collection["requests"]) == 300, "Full D2 required")
    params = e.json(collection["readout_ref"])
    require(
        collection["readout_ref"]["sha256"] == "ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0",
        "Fixed052 required",
    )
    head = CleanResidualReadout.from_parameters(params)
    normal = e.json(collection["normalization_ref"])["transform"]
    mask = np.asarray(normal["normalization_mask"][:10], dtype=bool)
    scale = np.where(mask, normal["scale"][:10], 1.0)
    bias = np.where(mask, normal["bias"][:10], 0.0)
    require((scale != 0).all(), "Invertible active transform required")
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    windows, items, names = {}, [], []
    for request in collection["requests"]:
        anchor = anchors[request["anchor_name"]]
        if anchor["name"] not in windows:
            with np.load(e.path(anchor["trajectory_ref"]), allow_pickle=False) as z:
                start = anchor["anchor_frame"]
                require(z["action_valid"][start : start + 40].all(), "Valid complete training window required")
                windows[anchor["name"]] = z["state128"][start : start + 41].copy()
        states = windows[anchor["name"]]
        require(states.shape == (41, 128), "Complete physical endpoints required")
        ledger = e.json(request["compact_ledger_ref"])
        require(ledger["complete"] and len(ledger["points"]) == 35, "Complete D2 ledger required")
        point = e.json(ledger["points"][-1])
        require(
            point["metadata"]["step_index"] == 34 and not point["metadata"]["feature_terms"],
            "Terminal contract differs",
        )
        with np.load(e.path(point["arrays_ref"]), allow_pickle=False) as z:
            raw = z["sample_active10_fp32"].astype(float) - point["metadata"]["sigma_actual"] * z[
                "raw_velocity_active10"
            ].astype(float)
            require(np.array_equal(raw, z["c0_active10"]), "Raw reconstruction differs")
            x = np.column_stack((z["hidden"].astype(float), raw))
            saved = z["clean_residual_active10"].copy()
        residual = head.predict_clean_residual(x)
        require(np.allclose(residual, saved, atol=1e-11, rtol=1e-9), "052 deployed residual differs")
        physical = (raw + residual - bias) / scale
        attitude = rot6d_to_matrix(states[0, 3:9])
        before = []
        for increment in rot6d_to_matrix(physical[:, 3:9]):
            before.append(attitude.copy())
            attitude = attitude @ increment
        items.append(
            dict(
                phi=np.column_stack(((x - head.center) / head.scale, np.ones(40))),
                raw_clean=raw,
                before_rotation=np.asarray(before),
                initial_position=states[0, :3],
                desired_positions=states[1:, :3],
                desired_opening=states[1:, 9],
            )
        )
        names.append(anchor["name"])
    require(len(items) == 300 and len(windows) == 150, "Exact terminal training denominator required")
    e.finish()
    output.mkdir()
    arrays = {k: np.stack([item[k] for item in items]) for k in items[0]}
    arrays.update(
        action_scale=scale,
        action_bias=bias,
        coefficients052=head.coefficients,
        endpoint_weights=sum((np.arange(40) < h) / (3 * h) for h in (8, 16, 40)),
    )
    with (output / "training_cache.npz").open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    publish(
        output / "prepared.json",
        dict(
            passed=True,
            complete=True,
            seed=42,
            plan_ref=reference(Path("plans/v0.0.1/057_terminal_trajectory_objective_20261025.md")),
            source_refs=[
                reference(Path(__file__)),
                reference(Path("src/s2a/trajectory_objective.py")),
                reference(Path("src/s2a/trajectory_readout_objective.py")),
            ],
            collection_ref=refs[2],
            params_ref=collection["readout_ref"],
            normalization_ref=collection["normalization_ref"],
            cache_ref=reference(output / "training_cache.npz"),
            anchor_names=names,
            requests=300,
            rows=12000,
            training_windows=150,
            input_sha256=e.inputs,
            deployment_history_qualified=False,
            producer_fits=0,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
