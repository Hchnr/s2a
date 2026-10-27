"""208 measured-current/generated-target supervision, no teacher arm inputs."""

import numpy as np
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.direct_arm_inverse import nominal_translation, physical_parameters
from s2a.direct_arm_inverse_inputs import inverse_condition
from s2a.durable_stage_supervisor import publish, sha
from s2a.joint_wrench_condition_forward import with_history


def main():
    out = REPO / "outputs/direct_arm_inverse_data_checked_20261026"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    for p in (
        __file__,
        "src/s2a/direct_arm_inverse.py",
        "src/s2a/direct_arm_inverse_inputs.py",
        "plans/v0.0.1/208_direct_arm_inverse_identification_20261026.md",
    ):
        e.read(p)
    config = e.json("configs/left_table_forward_20261025.json")
    parameters = e.json("outputs/left_table_fit_20261025/parameters.json")
    source = e.json(config["data_result_ref"]["path"])
    registration = e.json("configs/native_arm_response_data_20261025.json")
    closed = e.json("outputs/native_arm_response_data_20261025/closure_verified.json")
    assert closed["complete"] and closed["data_qualified"]
    inputs = []
    targets = []
    nominals = []
    held_inputs = []
    held_targets = []
    held_nominals = []
    held_records = []
    current_by_hash = {}
    native_by_hash = {}
    contract = None
    physical = None
    for entry, profile in zip(source["trajectories"], registration["trajectories"], strict=True):
        assert entry["source"] == profile["ref"]
        ref = profile["result_ref"]
        e.check(ref["path"], ref["sha256"])
        meta = e.json(ref["path"])
        if contract is None:
            contract = meta["controller_contract"]
            physical = physical_parameters(contract)
        else:
            assert meta["controller_contract"] == contract
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = with_history({k: a[k].copy() for k in a.files}, parameters)
        data = attach_current_wrench(data, entry, config, e)
        ref = entry["source"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            desired = a["controller_site_pose"][1:].copy()
            commands = a["sent_commands"].copy()
            np.testing.assert_allclose(a["controller_site_pose"][:-1], data["x_site_pose"], atol=1e-9, rtol=0)
            np.testing.assert_array_equal(commands, data["action_command"])
        values = inverse_condition(data, desired, gripper_command=commands[:, -1])
        nominal = nominal_translation(data, desired, physical)
        if entry["split"] == "fit":
            inputs.append(values)
            targets.append(commands[:, :3])
            nominals.append(nominal)
            current_by_hash[entry["source"]["sha256"]] = data
            native_by_hash[entry["source"]["sha256"]] = entry["index"]
        else:
            held_inputs.append(values)
            held_targets.append(commands[:, :3])
            held_nominals.append(nominal)
            held_records.append(dict(native_index=entry["index"], rows=len(values)))
    natural = np.vstack(inputs)
    assert natural.shape == (4865, 89)
    mean = natural.mean(axis=0)
    scale = np.maximum(natural.std(axis=0), 1e-8)
    old = e.json("outputs/left_table_forward_equations_20261025/result.json")
    added = e.json("outputs/finger_loading_equations_20261025/result.json")
    assert added["complete"] and added["rows"] == 5081 and added["contrast_equations"] == 216
    proof = e.json("outputs/finger_loading_equations_20261025/independent_verified.json")
    assert proof["independent_equations_verified"]
    names = {
        "086": "paired_arm_response",
        "097": "training_response_coverage",
        "106": "wrench_coverage_interventions",
        "115": "condition_span_interventions",
        "144": "left_table_interventions",
        "191": "finger_loading_interventions",
    }
    registries = {k: e.json(f"configs/{v}_20261025.json") for k, v in names.items()}
    summary = e.json("outputs/finger_loading_interventions_20261025/data_summary.json")
    assert summary["data_qualified"] and summary["owner_normal_wait_verified"] and summary["verified_cases"] == 21
    groups = []
    old_rows = old["contrast_sources"]
    for i in range(0, 207, 3):
        row = old_rows[i]
        refs = [row["baseline_ref"]]
        for axis in range(3):
            selected = old_rows[i + axis]
            assert selected["axis"] == axis
            refs.extend([selected["minus_ref"], selected["plus_ref"]])
        groups.append(dict(dataset=row["dataset"], group=row["anchor_group"], refs=refs))
    for group in range(3):
        groups.append(
            dict(
                dataset="191", group=group, refs=[r["arrays_ref"] for r in summary["cases"][7 * group : 7 * group + 7]]
            )
        )
    assert len(groups) == 72 and len({(r["dataset"], r["group"]) for r in groups}) == 72
    records = []
    for group in groups:
        registry = registries[group["dataset"]]
        trial = registry["trials"][7 * group["group"]]
        profile = registry["profiles"][trial["profile_index"]]
        source_hash = profile["trajectory_ref"]["sha256"]
        assert source_hash in current_by_hash
        full = current_by_hash[source_hash]
        anchor = trial["anchor_cycle"]
        condition = {
            k: np.asarray(v)[anchor : anchor + 1].copy()
            for k, v in full.items()
            if np.asarray(v).ndim > 0 and len(v) == len(full["x_site_pose"])
        }
        for branch, ref in enumerate(group["refs"]):
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as a:
                np.testing.assert_allclose(
                    a["controller_site_pose"][-2], condition["x_site_pose"][0], atol=1e-9, rtol=0
                )
                desired = a["controller_site_pose"][-1:]
                commands = a["sent_commands"][-1:]
                values = inverse_condition(condition, desired, gripper_command=commands[:, -1])
                nominal = nominal_translation(condition, desired, physical)
            inputs.append(values)
            targets.append(commands[:, :3])
            nominals.append(nominal)
        records.append(
            dict(
                dataset=group["dataset"],
                anchor_group=group["group"],
                native_index=native_by_hash[source_hash],
                anchor_cycle=anchor,
                refs=group["refs"],
            )
        )
    values = np.vstack(inputs)
    target = np.vstack(targets)
    nominal = np.vstack(nominals)
    weights = np.r_[np.full(4865, 0.5 / 4865), np.full(504, 0.5 / 504)]
    held_values = np.vstack(held_inputs)
    held_target = np.vstack(held_targets)
    held_nominal = np.vstack(held_nominals)
    assert values.shape == (5369, 89) and target.shape == nominal.shape == (5369, 3) and held_values.shape == (1315, 89)
    for a in (values, target, nominal, held_values, held_target, held_nominal):
        assert np.isfinite(a).all()
    physical.update(
        normalization_mean=mean.tolist(),
        normalization_scale=scale.tolist(),
        frequencies=(np.random.default_rng(42).normal(size=(64, 89)) / np.sqrt(89)).tolist(),
        ridge=0.001,
        seed=42,
        controller_contract=contract,
        schema="207 current83 plus explicit desired state6;no teacher arm input",
    )
    e.recheck()
    out.mkdir()
    path = out / "training.npz"
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            values=values,
            target=target,
            nominal=nominal,
            weights=weights,
            holdout_values=held_values,
            holdout_target=held_target,
            holdout_nominal=held_nominal,
        )
    publish(out / "parameters.json", physical)
    publish(
        out / "result.json",
        dict(
            complete=True,
            training_rows=5369,
            natural_rows=4865,
            branch_rows=504,
            holdout_rows=1315,
            branch_records=records,
            holdout_records=held_records,
            training_ref=dict(path=str(path), sha256=sha(path)),
            parameter_ref=dict(path=str(out / "parameters.json"), sha256=sha(out / "parameters.json")),
            source_sha256=e.hashes,
            fits=0,
            physics_steps=0,
            independently_verified=False,
            control_qualified=False,
        ),
    )
    print("208 prepared5369training/1315holdout;no fit;independent verification pending")


if __name__ == "__main__":
    main()
