"""104 bind complete103 qualification and fit-only current wrench statistics."""

from pathlib import Path

import numpy as np
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def ref(path):
    return dict(path=str(path.resolve()), sha256=sha256(path))


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/104_wrench_conditioned_arm_forward_20261025.md")
    old = Path("configs/nonlinear_conditioned_arm_forward_20261025.json")
    config = e.json(old)
    closure = Path("outputs/native_wrench_data_20261025/closure_verified.json")
    qualified = e.json(closure)
    assert qualified["complete"] and qualified["data_qualified"] and qualified["owner_normal_wait_verified"]
    assert qualified["verified_trajectories"] == 30 and qualified["wrench_endpoints"] == 6210
    for path, signature in qualified["source_sha256"].items():
        e.check(path, signature)
    config["wrench_closure_ref"] = ref(closure)
    ref_data = config["data_result_ref"]
    e.check(ref_data["path"], ref_data["sha256"])
    fit = [p for p in e.json(ref_data["path"])["trajectories"] if p["split"] == "fit"]
    assert len(fit) == 24
    values, weights = [], []
    for entry in fit:
        source = entry["cache_ref"]
        e.check(source["path"], source["sha256"])
        with np.load(source["path"], allow_pickle=False) as cache:
            data = {k: cache[k].copy() for k in cache.files}
        current = attach_current_wrench(data, entry, config, e)["x_current_wrench_world"]
        values.append(current)
        weights.extend([1 / (24 * len(current))] * len(current))
    values = np.vstack(values)
    weights = np.array(weights)
    assert values.shape == (4865, 6) and abs(weights.sum() - 1) < 1e-12
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1
    prior = Path("outputs/nonlinear_conditioned_arm_forward_20261025/closure_verified.json")
    c = e.json(prior)
    assert c["complete"] and c["numerically_verified"] and c["owner_normal_wait_verified"]
    config.update(
        output_dir="outputs/wrench_conditioned_arm_forward_20261025",
        state_dimension=89,
        design_dimension=872,
        wrench_mean=mean.tolist(),
        wrench_scale=scale.tolist(),
        wrench_rows_used=4865,
        wrench_statistics_holdout_rows=0,
        source_protocol_ref=ref(old),
        prior_fit_closure_ref=ref(prior),
    )
    e.recheck()
    config["binding_source_sha256"] = e.hashes
    write_once("configs/wrench_conditioned_arm_forward_20261025.json", config)
    print("104 bound30 qualified sources and24fit wrench stats; parameter_fits0")


if __name__ == "__main__":
    main()
