"""115 independent scalar features and pivoted QR selection audit."""

import math
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from scipy.linalg import qr


def main():
    e = Evidence()
    e.read(__file__)
    selected = e.json("configs/condition_span_interventions_20261025.json")
    for path, digest in selected["source_sha256"].items():
        e.check(path, digest)
    old = e.json("configs/training_response_coverage_20261025.json")
    recent = e.json("configs/wrench_coverage_interventions_20261025.json")
    original = e.json("configs/paired_arm_response_20261025.json")
    assert selected["profiles"] == old["profiles"] == recent["profiles"]
    assert selected["all_new_data_split"] == "fit" and len(selected["trials"]) == 147
    ref = selected["normalization_parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    params = e.json(ref["path"])
    ref = selected["selection_fit_cache_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as cache:
        raw = cache["raw"].copy()
    fit = e.json("outputs/joint_wrench_condition_forward_20261025/fit_result.json")
    assert fit["cache_ref"] == ref
    offsets, offset = {}, 0
    for row in fit["fit_sources"]:
        offsets[row["index"]] = offset
        offset += row["rows"]
    assert raw.shape == (offset, 89) == (4865, 89)
    ref = selected["native_data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = {r["index"]: r for r in e.json(ref["path"])["trajectories"] if r["split"] == "fit"}
    mean, scale = np.asarray(params["state_mean"]), np.asarray(params["state_scale"])
    frequencies, joint = np.asarray(params["frequencies"]), np.asarray(params["joint_frequencies"])
    phi = []
    for value in raw:
        z = (value - mean) / scale
        old_phase = [sum(float(w[k]) * float(z[k]) for k in range(83)) for w in frequencies]
        new_phase = [sum(float(w[k]) * float(z[k]) for k in range(89)) for w in joint]
        phi.append(
            [
                1.0,
                *z[:83],
                *[math.cos(v) / 8 for v in old_phase],
                *[math.sin(v) / 8 for v in old_phase],
                *z[83:],
                *[math.cos(v) / math.sqrt(32) for v in new_phase],
                *[math.sin(v) / math.sqrt(32) for v in new_phase],
            ]
        )
    phi = np.asarray(phi)
    assert phi.shape == (4865, 282)
    support = []
    for dataset, registry in (("086", original), ("097", old), ("106", recent)):
        for group in (0, 3, 5) if dataset == "086" else range(21):
            profile = registry["profiles"][group // 2 if dataset == "086" else group]
            index = next(i for i, r in entries.items() if r["source"] == profile["trajectory_ref"])
            anchor = registry["trials"][7 * group]["anchor_cycle"]
            support.append(phi[offsets[index] + anchor])
    assert len(support) == 45 and len(selected["selection_records"]) == 21
    total = 0
    for group, profile in enumerate(old["profiles"]):
        index = profile["native_data_index"]
        assert entries[index]["source"] == profile["trajectory_ref"]
        assert profile["trajectory_ref"]["sha256"] not in old["excluded_086_trajectory_sha256"]
        excluded = {registry["trials"][7 * group]["anchor_cycle"] for registry in (old, recent)}
        ref = entries[index]["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            mask = cache["protocol_native_goal_override"].astype(bool)
        candidates = [t for t in range(1, profile["record"]["native_steps"]) if t not in excluded]
        assert not mask[candidates].any()
        q, r, _ = qr(np.asarray(support).T, mode="economic", pivoting=True)
        # All existing directions must be independent, as registered by the SVD producer.
        assert np.min(np.abs(np.diag(r))) > 1e-10
        values = phi[offsets[index] + np.asarray(candidates)]
        residual = values - (values @ q) @ q.T
        scores = np.sum(residual**2, axis=1) / np.sum(values**2, axis=1)
        chosen = int(np.argmax(scores))
        anchor = candidates[chosen]
        record = selected["selection_records"][group]
        assert record["profile_index"] == group and record["native_data_index"] == index
        assert record["anchor_cycle"] == anchor and record["previous_support_rank"] == len(support)
        assert record["excluded_anchors"] == sorted(excluded) and record["candidate_count"] == len(candidates)
        np.testing.assert_allclose(record["unspanned_feature_squared_fraction"], scores[chosen], rtol=0, atol=1e-10)
        support.append(values[chosen])
        for branch in range(7):
            expected = dict(recent["trials"][7 * group + branch])
            expected.update(anchor_cycle=anchor, physics_steps=anchor + 1, phase="fit_condition_span")
            assert selected["trials"][7 * group + branch] == expected
            total += anchor + 1
    assert total == selected["physics_steps"] == 10626
    assert [sum(p["task"] == task for p in selected["profiles"]) for task in range(3)] == [7, 7, 7]
    e.recheck()
    output = Path("outputs/condition_span_selection_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "independent_verified.json",
        dict(
            complete=True,
            selection_verified=True,
            planned_control_periods=total,
            anchors=21,
            trials=147,
            physics_steps=0,
            source_sha256=e.hashes,
        ),
    )
    print("115 independent scalar/QR21anchors147branches verified;planned10626periods;physics0")


if __name__ == "__main__":
    main()
