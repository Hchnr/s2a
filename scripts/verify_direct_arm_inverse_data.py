"""209 independently rebuild current/desired inputs, teacher targets and physical baseline."""

import numpy as np
from arm_joint_wrench_independent import augment
from arm_wrench_data_independent import attach_current_wrench
from audit_wam_position import Evidence
from direct_arm_inverse_independent import features, inputs, nominal

from s2a.artifacts import REPO
from s2a.direct_arm_inverse import feature_matrix
from s2a.durable_stage_supervisor import publish


def main():
    root = REPO / "outputs/direct_arm_inverse_data_checked_20261026"
    if (root / "independent_verified.json").exists():
        raise FileExistsError(root / "independent_verified.json")
    e = Evidence()
    for p in (
        __file__,
        "scripts/direct_arm_inverse_independent.py",
        "src/s2a/direct_arm_inverse.py",
        "src/s2a/direct_arm_inverse_inputs.py",
    ):
        e.read(p)
    record = e.json(root / "result.json")
    for p, h in record["source_sha256"].items():
        e.check(p, h)
    ref = record["training_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        provided = {k: a[k].copy() for k in a.files}
    params = e.json(record["parameter_ref"]["path"])
    config = e.json("configs/left_table_forward_20261025.json")
    physical = e.json("outputs/left_table_fit_20261025/parameters.json")
    source = e.json(config["data_result_ref"]["path"])
    registration = e.json("configs/native_arm_response_data_20261025.json")
    train_values = []
    train_target = []
    train_nominal = []
    held_values = []
    held_target = []
    held_nominal = []
    states = {}
    for entry, profile in zip(source["trajectories"], registration["trajectories"], strict=True):
        ref = profile["result_ref"]
        e.check(ref["path"], ref["sha256"])
        assert e.json(ref["path"])["controller_contract"] == params["controller_contract"]
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = augment({k: a[k].copy() for k in a.files}, physical)
        data = attach_current_wrench(data, entry, config, e)
        ref = entry["source"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            desired = a["controller_site_pose"][1:].copy()
            commands = a["sent_commands"].copy()
        values = inputs(data, desired, commands[:, -1])
        base = nominal(data, desired, params["controller_contract"])
        if entry["split"] == "fit":
            states[entry["index"]] = data
            train_values.append(values)
            train_target.append(commands[:, :3])
            train_nominal.append(base)
        else:
            held_values.append(values)
            held_target.append(commands[:, :3])
            held_nominal.append(base)
    natural = np.vstack(train_values)
    np.testing.assert_allclose(params["normalization_mean"], natural.mean(axis=0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(params["normalization_scale"], np.maximum(natural.std(axis=0), 1e-8), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(params["frequencies"], np.random.default_rng(42).normal(size=(64, 89)) / np.sqrt(89))
    original = e.json("outputs/left_table_forward_equations_20261025/result.json")
    expected = [(r["dataset"], r["anchor_group"]) for r in original["contrast_sources"][::3]] + [
        ("191", i) for i in range(3)
    ]
    assert [(r["dataset"], r["anchor_group"]) for r in record["branch_records"]] == expected
    names = {
        "086": "paired_arm_response",
        "097": "training_response_coverage",
        "106": "wrench_coverage_interventions",
        "115": "condition_span_interventions",
        "144": "left_table_interventions",
        "191": "finger_loading_interventions",
    }
    by_hash = {r["source"]["sha256"]: r["index"] for r in source["trajectories"]}
    for group in record["branch_records"]:
        registry = e.json(f"configs/{names[group['dataset']]}_20261025.json")
        trial = registry["trials"][7 * group["anchor_group"]]
        profile = registry["profiles"][trial["profile_index"]]
        assert (
            group["native_index"] == by_hash[profile["trajectory_ref"]["sha256"]]
            and group["anchor_cycle"] == trial["anchor_cycle"]
        )
        assert len(group["refs"]) == 7
        if group["dataset"] == "191":
            added_summary = e.json("outputs/finger_loading_interventions_20261025/data_summary.json")
            expected_refs = [
                r["arrays_ref"]
                for r in added_summary["cases"][7 * group["anchor_group"] : 7 * group["anchor_group"] + 7]
            ]
        else:
            matched = [
                r
                for r in original["contrast_sources"]
                if r["dataset"] == group["dataset"] and r["anchor_group"] == group["anchor_group"]
            ]
            assert len(matched) == 3
            matched.sort(key=lambda r: r["axis"])
            expected_refs = [matched[0]["baseline_ref"]]
            for r in matched:
                expected_refs.extend([r["minus_ref"], r["plus_ref"]])
        assert group["refs"] == expected_refs

        full = states[group["native_index"]]
        anchor = trial["anchor_cycle"]
        data = {
            k: np.asarray(v)[anchor : anchor + 1]
            for k, v in full.items()
            if np.asarray(v).ndim > 0 and len(v) == len(full["x_site_pose"])
        }
        for ref in group["refs"]:
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as a:
                desired = a["controller_site_pose"][-1:].copy()
                command = a["sent_commands"][-1:].copy()
            train_values.append(inputs(data, desired, command[:, -1]))
            train_target.append(command[:, :3])
            train_nominal.append(nominal(data, desired, params["controller_contract"]))
    independent = dict(
        values=np.vstack(train_values),
        target=np.vstack(train_target),
        nominal=np.vstack(train_nominal),
        holdout_values=np.vstack(held_values),
        holdout_target=np.vstack(held_target),
        holdout_nominal=np.vstack(held_nominal),
    )
    for key, value in independent.items():
        np.testing.assert_allclose(provided[key], value, atol=1e-10, rtol=0)
    np.testing.assert_allclose(provided["weights"][:4865], 0.5 / 4865, atol=1e-15, rtol=0)
    np.testing.assert_allclose(provided["weights"][4865:], 0.5 / 504, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        feature_matrix(independent["values"], params), features(independent["values"], params), atol=1e-10, rtol=0
    )
    e.recheck()
    publish(
        root / "independent_verified.json",
        dict(
            complete=True,
            independent_input_and_nominal_verified=True,
            training_rows=5369,
            holdout_rows=1315,
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
        ),
    )
    print("209 all5369train/1315holdout input/target/nominal and218features verified;no control claim")


if __name__ == "__main__":
    main()
