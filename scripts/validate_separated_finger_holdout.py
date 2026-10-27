"""182 six held trajectories, fixed model versus gap hold and measured velocity."""

import json

import numpy as np
from arm_joint_wrench_independent import feature_matrix
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.joint_wrench_condition_forward import design, with_history


def main():
    out = REPO / "outputs/separated_finger_holdout_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    fit_proof = e.json("outputs/arm_finger_separated_fit_20261025/independent_verified.json")
    assert fit_proof["complete"] and fit_proof["numerically_verified"]
    ref = fit_proof["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    coefficients = np.asarray(e.json(ref["path"])["coefficients"])
    parameters = e.json("outputs/left_table_fit_20261025/parameters.json")
    config = e.json("configs/left_table_forward_20261025.json")
    source = e.json(config["data_result_ref"]["path"])
    produced = e.json("outputs/arm_finger_response_labels_20261025/result.json")
    ref = produced["labels_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as labels:
        target = labels["holdout_target"].copy()
    offset = 0
    results = []
    for entry in source["trajectories"]:
        if entry["split"] != "holdout":
            continue
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = with_history({k: cache[k].copy() for k in cache.files}, parameters)
        data = attach_current_wrench(data, entry, config, e)
        n = len(data["x_site_pose"])
        actual = target[offset : offset + n]
        predicted = design(data, parameters) @ coefficients
        independent = feature_matrix(data, parameters) @ coefficients
        np.testing.assert_allclose(predicted, independent, atol=1e-10, rtol=0)
        velocity = 0.05 * (data["x_finger_qvel"][:, 0] - data["x_finger_qvel"][:, 1])
        model_rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        hold_rmse = float(np.sqrt(np.mean(actual**2)))
        velocity_rmse = float(np.sqrt(np.mean((velocity - actual) ** 2)))
        results.append(
            dict(
                native_index=entry["index"],
                rows=n,
                model_rmse_m=model_rmse,
                gap_hold_rmse_m=hold_rmse,
                current_velocity_rmse_m=velocity_rmse,
                no_worse_than_hold=model_rmse <= hold_rmse,
            )
        )
        offset += n
    assert offset == 1315 and len(results) == 6
    e.recheck()
    publish(
        out / "result.json",
        dict(
            complete=True,
            results=results,
            passed_hold_baseline=sum(r["no_worse_than_hold"] for r in results),
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            local_intervention_validated=False,
        ),
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
