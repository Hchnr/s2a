"""198 independent history donor indexing and scalar block reconstruction."""

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    output = REPO / "outputs/finger_history_equations_20261025/independent_verified.json"
    if output.exists():
        raise FileExistsError(output)
    e = Evidence()
    e.read(__file__)
    prepared = e.json("outputs/finger_history_equations_20261025/result.json")
    for p, h in prepared["source_sha256"].items():
        e.check(p, h)
    hp = e.json("outputs/measured_finger_history_20261025/verified.json")
    assert hp["independently_reconstructed"]
    ref = hp["history_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        history = a["history"].copy()
        mean = a["fit_mean"].copy()
        scale = a["fit_scale"].copy()
    offsets = {}
    begin = 0
    donors = []
    for r in hp["identities"]:
        offsets[r["native_index"]] = begin
        if r["split"] == "fit":
            donors.extend(range(begin, begin + r["rows"]))
        begin += r["rows"]
    old = e.json("outputs/left_table_forward_equations_20261025/result.json")
    merged = e.json("outputs/finger_loading_equations_20261025/result.json")
    source = e.json("outputs/native_arm_response_data_20261025/reconstruction_result.json")
    by_hash = {r["source"]["sha256"]: r["index"] for r in source["trajectories"]}
    names = {
        "086": "paired_arm_response",
        "097": "training_response_coverage",
        "106": "wrench_coverage_interventions",
        "115": "condition_span_interventions",
        "144": "left_table_interventions",
        "191": "finger_loading_interventions",
    }
    for record in old["contrast_sources"] + merged["new_contrasts"]:
        registry = e.json(f"configs/{names[record['dataset']]}_20261025.json")
        trial = registry["trials"][record["anchor_group"] * 7]
        profile = registry["profiles"][trial["profile_index"]]
        donors.append(offsets[by_hash[profile["trajectory_ref"]["sha256"]]] + trial["anchor_cycle"])
    ref = merged["training_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        base, labels, weights = (a[k].copy() for k in ("features", "target", "weights"))
    ref = prepared["training_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        actual, y, w, h = (a[k].copy() for k in ("features", "target", "weights", "raw_history"))
    np.testing.assert_array_equal(h, history[donors])
    np.testing.assert_array_equal(y, labels)
    np.testing.assert_array_equal(w, weights)
    params = e.json(prepared["parameter_ref"]["path"])
    np.testing.assert_array_equal(params["finger_history_mean"], mean)
    np.testing.assert_array_equal(params["finger_history_scale"], scale)
    for i in range(5081):
        for block in range(4):
            start = 321 * block
            np.testing.assert_array_equal(actual[i, start : start + 282], base[i, 282 * block : 282 * (block + 1)])
            expected = [
                float(base[i, 282 * block]) * (float(history[donors[i], j]) - float(mean[j])) / float(scale[j])
                for j in range(39)
            ]
            np.testing.assert_allclose(actual[i, start + 282 : start + 321], expected, atol=1e-10, rtol=0)
    e.recheck()
    publish(
        output,
        dict(
            complete=True,
            independent_equations_verified=True,
            independent_history_donors_verified=True,
            rows=5081,
            dimension=1284,
            source_sha256=e.hashes,
            fits=0,
            physics_steps=0,
            control_qualified=False,
        ),
    )
    print("198 all5081 history donors and1284 blocks independently verified")


if __name__ == "__main__":
    main()
