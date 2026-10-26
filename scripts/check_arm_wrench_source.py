"""104 actual native history/features and selected contrast parity without fitting."""

from pathlib import Path

import numpy as np
from arm_wrench_data import attach_current_wrench
from arm_wrench_data_independent import attach_current_wrench as independent_attach
from arm_wrench_independent import augment, conditions, feature_matrix
from arm_wrench_independent_equations import equations
from arm_wrench_training import contrast_rows
from audit_wam_position import Evidence

from s2a.wrench_conditioned_arm_forward import design, raw_condition, with_history


def main():
    e = Evidence()
    config = e.json("configs/wrench_conditioned_arm_forward_20261025.json")
    for key in (
        "data_closure_ref",
        "data_result_ref",
        "paired_closure_ref",
        "prior_fit_closure_ref",
        "additional_closure_ref",
        "statistics_ref",
        "wrench_closure_ref",
    ):
        ref = config[key]
        e.check(ref["path"], ref["sha256"])
    closed = e.json(config["data_closure_ref"]["path"])
    assert closed["complete"] and closed["data_qualified"]
    source = e.json(config["data_result_ref"]["path"])
    entries = [row for row in source["trajectories"] if row["split"] == "fit"]
    assert len(entries) == 24
    primary, independent, raw, actions, weights = [], [], [], [], []
    p = None
    for entry in entries:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        meta = Path(entry["source"]["path"]).parent / "result.json"
        e.check(meta, closed["source_sha256"][str(meta.resolve())])
        contract = e.json(meta)["controller_contract"]
        current = dict(dt=0.05, kp=contract["kp"], kd=contract["kd"])
        if p is None:
            p = current
        else:
            assert p == current
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = {k: cache[k].copy() for k in cache.files}
        a = attach_current_wrench(with_history(data, p), entry, config, e)
        b = independent_attach(augment(data, p), entry, config, e)
        np.testing.assert_allclose(a["x_current_wrench_world"], b["x_current_wrench_world"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(a["x_response_residual"], b["x_response_residual"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(raw_condition(a), conditions(b), rtol=0, atol=1e-10)
        primary.append(a)
        independent.append(b)
        raw.append(raw_condition(a))
        actions.append(a["action_goal_delta"][:, :3])
        weights.extend([1 / (24 * len(a["step"]))] * len(a["step"]))
    w = np.asarray(weights)
    assert len(w) == 4865 and abs(w.sum() - 1) < 1e-12
    for values, prefix in ((np.vstack(raw), "state"), (np.vstack(actions), "translation")):
        mu = np.sum(values * w[:, None], axis=0)
        scale = np.sqrt(np.sum((values - mu) ** 2 * w[:, None], axis=0))
        scale[scale < 1e-8] = 1
        p[prefix + "_mean"], p[prefix + "_scale"] = mu.tolist(), scale.tolist()
    stats = e.json(config["statistics_ref"]["path"])
    for key in ("state_mean", "state_scale", "translation_mean", "translation_scale"):
        if key.startswith("state_"):
            np.testing.assert_allclose(p[key][:83], stats[key], rtol=0, atol=1e-12)
            new = config["wrench_mean" if key == "state_mean" else "wrench_scale"]
            np.testing.assert_allclose(p[key][83:], new, rtol=0, atol=1e-10)
            p[key] = stats[key] + new
        else:
            np.testing.assert_allclose(p[key], stats[key], rtol=0, atol=1e-12)
            p[key] = stats[key]
    p["frequencies"] = config["frequencies"]
    x = np.vstack([design(a, p) for a in primary])
    xx = np.vstack([feature_matrix(b, p) for b in independent])
    assert x.shape == xx.shape == (4865, 872)
    np.testing.assert_allclose(x, xx, rtol=1e-8, atol=1e-8)
    cx, y, base, records = contrast_rows(config, e, p)
    cxx, yy, bb, ids = equations(config, e, p)
    np.testing.assert_allclose(cx, cxx, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(y, yy, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(base, bb, rtol=1e-10, atol=1e-10)
    assert cx.shape == (72, 872) and y.shape == base.shape == (72, 12)
    assert ids == [(r["dataset"], r["anchor_group"], r["axis"], r["minus_case"], r["plus_case"]) for r in records]
    e.recheck()
    print("104 all4865native histories/features and72contrasts independently matched; new_fits0")


if __name__ == "__main__":
    main()
