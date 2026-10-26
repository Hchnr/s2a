"""166 fit-only support radius, independently checked with dense distances."""

import json

import numpy as np
from arm_joint_wrench_independent import conditions
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence
from scipy.spatial.distance import cdist

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha
from s2a.joint_wrench_condition_forward import with_history
from s2a.training_support import calibrate_support


def main():
    root = REPO / "outputs/training_support_calibration_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    for p in (__file__, "src/s2a/training_support.py", "plans/v0.0.1/166_support_aware_inverse_control_20261025.md"):
        e.read(p)
    config = e.json("configs/left_table_forward_20261025.json")
    parameter_ref = "outputs/left_table_fit_20261025/parameters.json"
    parameters = e.json(parameter_ref)
    source = e.json(config["data_result_ref"]["path"])
    mean, scale = (np.asarray(parameters[k]) for k in ("state_mean", "state_scale"))
    chunks, labels = [], []
    for entry in source["trajectories"]:
        if entry["split"] != "fit":
            continue
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = with_history({k: cache[k].copy() for k in cache.files}, parameters)
        data = attach_current_wrench(data, entry, config, e)
        z = (conditions(data) - mean) / scale
        chunks.append(z)
        labels.extend([entry["index"]] * len(z))
    points, groups = np.vstack(chunks), np.asarray(labels)
    assert points.shape == (4865, 89) and len(np.unique(groups)) == 24
    equations = e.json("outputs/left_table_forward_equations_20261025/result.json")
    ref = equations["training_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as cache:
        phi = cache["features"][:4865, :282]
        np.testing.assert_allclose(points, np.c_[phi[:, 1:84], phi[:, 212:218]], atol=1e-10, rtol=0)
    calibrated = calibrate_support(points, groups)
    independent = np.empty(len(points))
    for group in np.unique(groups):
        selected = groups == group
        independent[selected] = cdist(points[selected], points[~selected]).min(axis=1)
    np.testing.assert_allclose(independent, calibrated["cross_trajectory_distances"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.percentile(independent, 95), calibrated["radius"], atol=1e-12, rtol=0)
    e.recheck()
    root.mkdir()
    arrays = root / "support.npz"
    with arrays.open("xb") as stream:
        np.savez_compressed(stream, points=points, groups=groups, cross_trajectory_distances=independent)
    publish(
        root / "verified.json",
        dict(
            complete=True,
            fit_trajectories=24,
            fit_rows=4865,
            quantile=0.95,
            radius=calibrated["radius"],
            independent_dense_verified=True,
            control_qualified=False,
            parameter_ref=dict(path=str((REPO / parameter_ref).resolve()), sha256=sha(parameter_ref)),
            arrays_ref=dict(path=str(arrays), sha256=sha(arrays)),
            source_sha256=e.hashes,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
            evaluation_rows_used_for_calibration=0,
        ),
    )
    print(json.dumps(dict(radius=calibrated["radius"], rows=len(points), independent_dense_verified=True)))


if __name__ == "__main__":
    main()
