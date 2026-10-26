"""164 compare audited control conditions against original24fit natural states."""

import json

import numpy as np
from arm_joint_wrench_independent import conditions
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.joint_wrench_condition_forward import raw_condition, with_history


def main():
    root = REPO / "outputs/native_condition_support_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    e.read(__file__)
    config = e.json("configs/left_table_forward_20261025.json")
    parameter_path = "outputs/left_table_fit_20261025/parameters.json"
    parameters = e.json(parameter_path)
    source = e.json(config["data_result_ref"]["path"])
    chunks, identities = [], []
    for entry in source["trajectories"]:
        if entry["split"] != "fit":
            continue
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = with_history({k: cache[k].copy() for k in cache.files}, parameters)
        data = attach_current_wrench(data, entry, config, e)
        raw = raw_condition(data)
        np.testing.assert_allclose(raw, conditions(data), atol=1e-12, rtol=0)
        chunks.append(raw)
        identities.extend(dict(native_index=entry["index"], cycle=i) for i in range(len(raw)))
    train = np.vstack(chunks)
    assert train.shape == (4865, 89)
    mean, scale = (np.asarray(parameters[k]) for k in ("state_mean", "state_scale"))
    ztrain = (train - mean) / scale
    lower, upper = train.min(axis=0), train.max(axis=0)
    path = next((REPO / "outputs/qualified_inverse_enumerated_20261025/case00").rglob("steps.jsonl"))
    verified = path.parents[3] / "independent_verified.json"
    # Locate the collector case independently from simulation nesting.
    verified = REPO / "outputs/qualified_inverse_enumerated_20261025/case00/independent_verified.json"
    assert e.json(verified)["mechanical_labels_images_passed"]
    rows = []
    for line in e.read(path).decode().splitlines():
        row = json.loads(line)
        data = {k: np.asarray(v) for k, v in row["qualified_forward_input"].items()}
        raw = raw_condition(data)[0]
        np.testing.assert_allclose(raw, conditions(data)[0], atol=1e-12, rtol=0)
        z = (raw - mean) / scale
        distances = np.linalg.norm(ztrain - z, axis=1)
        nearest = int(np.argmin(distances))
        outside = np.maximum(lower - raw, raw - upper) / scale
        rows.append(
            dict(
                step=row["step"],
                nearest_native=identities[nearest],
                nearest_standardized_distance=float(distances[nearest]),
                outside_indices=np.flatnonzero(outside > 1e-10).tolist(),
                max_standardized_range_excess=float(max(0, outside.max())),
            )
        )
    e.recheck()
    publish(
        root / "result.json",
        dict(
            complete=True,
            training_natural_rows=4865,
            rows=rows,
            lower=lower.tolist(),
            upper=upper.tolist(),
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Original24fit natural-state support only; does not include additional207 contrast anchor conditions; marginal ranges and nearest distances do not prove physical derivative or joint support",
        ),
    )
    print(json.dumps([r for r in rows if r["step"] in (0, 70, 74, 78, 80, 84, 90)]))


if __name__ == "__main__":
    main()
