"""198 align audited history with unchanged natural and paired equations."""

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha


def main():
    out = REPO / "outputs/finger_history_equations_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/finger_history_forward.py")
    e.read("plans/v0.0.1/198_finger_history_response_candidate_20261025.md")
    history = e.json("outputs/measured_finger_history_20261025/verified.json")
    assert history["independently_reconstructed"] and history["prefix_invariance_verified"]
    for p, h in history["source_sha256"].items():
        e.check(p, h)
    ref = history["history_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        h = a["history"].copy()
        mean = a["fit_mean"].copy()
        scale = a["fit_scale"].copy()
    by_index = {}
    offset = 0
    natural = []
    for row in history["identities"]:
        values = h[offset : offset + row["rows"]]
        by_index[row["native_index"]] = values
        if row["split"] == "fit":
            natural.append(values)
        offset += row["rows"]
    source = e.json("outputs/native_arm_response_data_20261025/reconstruction_result.json")
    by_sha = {r["source"]["sha256"]: r["index"] for r in source["trajectories"]}
    previous = e.json("outputs/left_table_forward_equations_20261025/result.json")
    merged = e.json("outputs/finger_loading_equations_20261025/result.json")
    proof = e.json("outputs/finger_loading_equations_20261025/independent_verified.json")
    assert proof["independent_equations_verified"]
    for p, s in proof["source_sha256"].items():
        e.check(p, s)
    names = {
        "086": "paired_arm_response",
        "097": "training_response_coverage",
        "106": "wrench_coverage_interventions",
        "115": "condition_span_interventions",
        "144": "left_table_interventions",
        "191": "finger_loading_interventions",
    }
    registries = {k: e.json(f"configs/{v}_20261025.json") for k, v in names.items()}
    contrasts = []
    for row in previous["contrast_sources"] + merged["new_contrasts"]:
        dataset = row["dataset"]
        group = row["anchor_group"]
        registry = registries[dataset]
        profile = registry["profiles"][group // 2 if dataset == "086" else group]
        native = by_sha[profile["trajectory_ref"]["sha256"]]
        anchor = registry["trials"][7 * group]["anchor_cycle"]
        contrasts.append(by_index[native][anchor])
    raw_history = np.vstack([np.vstack(natural), contrasts])
    assert raw_history.shape == (5081, 39)
    ref = merged["training_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        x, y, w = (a[k].copy() for k in ("features", "target", "weights"))
    z = (raw_history - mean) / scale
    blocks = [np.c_[x[:, 282 * i : 282 * (i + 1)], z * x[:, 282 * i : 282 * i + 1]] for i in range(4)]
    extended = np.hstack(blocks)
    assert extended.shape == (5081, 1284)
    recovered = np.hstack([extended[:, 321 * i : 321 * i + 282] for i in range(4)])
    np.testing.assert_array_equal(recovered, x)
    parameters = e.json("outputs/left_table_fit_20261025/parameters.json")
    parameters.pop("coefficients")
    parameters.update(finger_history_mean=mean.tolist(), finger_history_scale=scale.tolist())
    e.recheck()
    out.mkdir()
    path = out / "training.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, features=extended, target=y, weights=w, raw_history=raw_history)
    publish(out / "parameters.json", parameters)
    publish(
        out / "result.json",
        dict(
            complete=True,
            rows=5081,
            design_dimension=1284,
            base_dimension=321,
            contrast_equations=216,
            history_dimension=39,
            old_design_bitwise_preserved=True,
            training_ref=dict(path=str(path), sha256=sha(path)),
            parameter_ref=dict(path=str(out / "parameters.json"), sha256=sha(out / "parameters.json")),
            source_sha256=e.hashes,
            fits=0,
            physics_steps=0,
            control_qualified=False,
        ),
    )
    print("198 causal39 history aligned;old1128 recovered bitwise;5081x1284 prepared")


if __name__ == "__main__":
    main()
