"""110 fixed109 prediction error attribution; no fit or simulator."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/110_joint_wrench_failure_attribution_20261025.md")
    root = Path("outputs/joint_wrench_condition_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    link = closed["validation_ref"]
    e.check(link["path"], link["sha256"])
    validation = e.json(link["path"])
    config = e.json("configs/joint_wrench_condition_forward_20261025.json")
    link = config["data_result_ref"]
    e.check(link["path"], link["sha256"])
    entries = e.json(link["path"])["trajectories"]
    held = []
    for row in validation["holdout"]:
        entry = next(r for r in entries if r["index"] == row["index"])
        assert entry["split"] == "holdout"
        for ref in (row["predictions_ref"], entry["cache_ref"]):
            e.check(ref["path"], ref["sha256"])
        with np.load(row["predictions_ref"]["path"], allow_pickle=False) as predictions:
            values = {k: predictions[k].copy() for k in ("fitted", "nominal")}
        with np.load(entry["cache_ref"]["path"], allow_pickle=False) as data:
            target = data["y_twist"][:, :3].copy()
            settled = data["protocol_native_goal_override"].astype(bool)
        errors = {k: v[:, 6:9] / 0.05 - target for k, v in values.items()}
        phases = {}
        for phase, mask in (("all", np.ones(len(target), dtype=bool)), ("native", ~settled), ("settled", settled)):
            assert np.any(mask)
            phases[phase] = dict(rows=int(mask.sum()))
            for name, error in errors.items():
                phases[phase][name] = dict(
                    sse=float(np.sum(error[mask] ** 2)),
                    rmse=float(np.sqrt(np.mean(np.sum(error[mask] ** 2, axis=1)))),
                    axis_sse=np.sum(error[mask] ** 2, axis=0).tolist(),
                )
        for name in errors:
            np.testing.assert_allclose(
                phases["native"][name]["sse"] + phases["settled"][name]["sse"], phases["all"][name]["sse"]
            )
            np.testing.assert_allclose(phases["all"][name]["rmse"], row["metrics"][name]["linear_velocity"])
        held.append(dict(index=row["index"], phases=phases))
    local = []
    for row in validation["local_response"]:
        metrics = {}
        for channel in ("so3_increment", "angular_velocity"):
            saved = row["metrics"][channel]
            target = np.asarray(saved["measured"])
            item = {}
            for name in ("predicted", "nominal"):
                error = np.asarray(saved[name]) - target
                columns = np.linalg.norm(error, axis=0)
                key = "fitted_frobenius_error" if name == "predicted" else "nominal_frobenius_error"
                np.testing.assert_allclose(np.linalg.norm(columns), saved[key], rtol=0, atol=1e-12)
                item[name] = columns.tolist()
            metrics[channel] = item
        local.append(dict(anchor_group=row["anchor_group"], metrics=metrics))
    e.recheck()
    output = Path("outputs/joint_wrench_failure_attribution_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            holdout=held,
            local_response=local,
            diagnostic_only=True,
            new_fits=0,
            physics_steps=0,
            source_sha256=e.hashes,
        ),
    )
    print("holdout28", next(r for r in held if r["index"] == 28))
    print("group4", next(r for r in local if r["anchor_group"] == 4))


if __name__ == "__main__":
    main()
