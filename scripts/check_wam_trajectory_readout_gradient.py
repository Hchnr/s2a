"""One deterministic actual training-cache witness for the trajectory gradient."""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_aggregation_candidate import candidate_head
from s2a.clean_training_aggregation import admit_aggregation
from s2a.clean_training_runtime import publish, reference
from s2a.geometry import rot6d_to_matrix
from s2a.trajectory_readout_objective import readout_objectives


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/clean_on_policy_aggregation_20261024/aggregate_fit")
    output = root / "trajectory_gradient_witness.json"
    require(not output.exists(), "Output exists")
    e = Evidence()
    _, prepared, _, collections = admit_aggregation(e.json(reference(root / "registration.json")), read_json=e.json)
    head, params_ref = candidate_head(reference(root / "orchestration/result.json"), read_json=e.json)
    request = collections[2]["requests"][0]
    anchor = next(a for a in prepared["training_anchors"] if a["name"] == request["anchor_name"])
    normal = e.json(collections[2]["normalization_ref"])["transform"]
    mask = np.asarray(normal["normalization_mask"][:10], dtype=bool)
    scale = np.where(mask, normal["scale"][:10], 1.0)
    bias = np.where(mask, normal["bias"][:10], 0.0)
    ledger = e.json(request["compact_ledger_ref"])
    require(ledger["complete"] and len(ledger["points"]) == 35, "Full cache required")
    point = e.json(ledger["points"][-1])
    require(point["metadata"]["step_index"] == 34, "Terminal node required")
    with np.load(e.path(point["arrays_ref"]), allow_pickle=False) as z:
        raw = z["sample_active10_fp32"].astype(float) - point["metadata"]["sigma_actual"] * z[
            "raw_velocity_active10"
        ].astype(float)
        require(np.array_equal(raw, z["c0_active10"]), "Raw clean differs")
        x = np.column_stack((z["hidden"].astype(float), raw))
    with np.load(e.path(anchor["trajectory_ref"]), allow_pickle=False) as z:
        start = anchor["anchor_frame"]
        states = z["state128"][start : start + 41].copy()
        require(z["action_valid"][start : start + 40].all(), "Invalid training window")
    physical = (raw + head.predict_clean_residual(x) - bias) / scale
    attitude = rot6d_to_matrix(states[0, 3:9])
    before = []
    for increment in rot6d_to_matrix(physical[:, 3:9]):
        before.append(attitude.copy())
        attitude = attitude @ increment
    weights = sum((np.arange(40) < h) / (3 * h) for h in (8, 16, 40))
    args = (
        head.center,
        head.scale,
        raw,
        scale,
        bias,
        np.asarray(before),
        states[0, :3],
        states[1:, :3],
        states[1:, 9],
        weights,
    )
    losses, grads = readout_objectives(x, head.coefficients, *args)
    rng = np.random.default_rng(42)
    checks = {}
    for name, columns in (("position", [0, 1, 2]), ("opening", [9])):
        direction = np.zeros_like(head.coefficients)
        direction[:, columns] = rng.normal(size=(len(direction), len(columns)))
        direction /= np.linalg.norm(direction)
        analytic = float(np.sum(grads[name] * direction))
        step = 1e-6
        plus = readout_objectives(x, head.coefficients + step * direction, *args)[0][name]
        minus = readout_objectives(x, head.coefficients - step * direction, *args)[0][name]
        numeric = (plus - minus) / (2 * step)
        require(np.isclose(analytic, numeric, atol=1e-8, rtol=1e-5), "Actual-cache directional gradient failed")
        checks[name] = dict(analytic=analytic, numeric=numeric, absolute_error=abs(analytic - numeric))
    e.finish()
    publish(
        output,
        dict(
            passed=True,
            complete=True,
            seed=42,
            request_name=request["name"] if "name" in request else anchor["name"],
            terminal_step=34,
            params_ref=params_ref,
            losses=losses,
            checks=checks,
            input_sha256=e.inputs,
            source_ref=reference(Path(__file__)),
            new_fits=0,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
            boundary="One deterministic training-cache mathematical witness, not fit qualification or performance. Fixed predicted rotations and ideal FP64 inverse normalization.",
        ),
    )


if __name__ == "__main__":
    main()
