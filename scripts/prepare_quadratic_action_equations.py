"""138 branchwise quadratic action equations."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from scipy.spatial.transform import Rotation

from s2a.artifacts import sha256
from s2a.diagonal_quadratic_action_forward import expand
from s2a.joint_wrench_condition_forward import nominal


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/138_quadratic_action_equations_20261025.md")
    root = Path("outputs/condition_span_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit_path = root / "fit_result.json"
    e.check(fit_path, closed["source_sha256"][str(fit_path.resolve())])
    fit = e.json(fit_path)
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    parameters = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        raw, y, base = (cache[k].copy() for k in ("raw", "target", "nominal"))
        old_x = cache["features"].copy()
        weights = cache["weights"].copy()
    assert raw.shape == (4865, 89) and y.shape == base.shape == (5063, 12)
    offsets, offset = {}, 0
    for row in fit["fit_sources"]:
        offsets[row["index"]] = offset
        offset += row["rows"]
    assert offset == 4865
    configs = {
        "086": e.json("configs/paired_arm_response_20261025.json"),
        "097": e.json("configs/training_response_coverage_20261025.json"),
        "106": e.json("configs/wrench_coverage_interventions_20261025.json"),
        "115": e.json("configs/condition_span_interventions_20261025.json"),
    }
    config = e.json("configs/condition_span_forward_20261025.json")
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = e.json(ref["path"])["trajectories"]
    states, targets, nominals, native_ids, identities = [], [], [], [], []
    anchor_rows = []
    records = fit["contrast_sources"]
    for dataset in ("086", "097", "106", "115"):
        groups = sorted({r["anchor_group"] for r in records if r["dataset"] == dataset})
        for group in groups:
            registry = configs[dataset]
            trial = registry["trials"][7 * group]
            profile = registry["profiles"][group // 2 if dataset == "086" else group]
            native = next(r for r in entries if r["source"] == profile["trajectory_ref"])
            assert native["split"] == "fit"
            indices = [i for i, r in enumerate(records) if r["dataset"] == dataset and r["anchor_group"] == group]
            assert [records[i]["axis"] for i in indices] == [0, 1, 2]
            anchor_rows.append(offsets[native["index"]] + trial["anchor_cycle"])
            states.append(raw[offsets[native["index"]] + trial["anchor_cycle"]])
            targets.append(y[np.asarray(indices) + 4865])
            nominals.append(base[np.asarray(indices) + 4865])
            native_ids.append(native["index"])
            identities.append(dict(dataset=dataset, anchor_group=group, native_index=native["index"]))
    assert len(states) == 66 and len(set(native_ids)) == 24
    e.read("src/s2a/diagonal_quadratic_action_forward.py")
    natural = expand(old_x[:4865])
    odd_x = []
    even_x = []
    odd_y = []
    even_y = []
    odd_b = []
    even_b = []
    for j, r in enumerate(records):
        index = next(
            i
            for i, v in enumerate(identities)
            if v["dataset"] == r["dataset"] and v["anchor_group"] == r["anchor_group"]
        )
        phi = old_x[anchor_rows[index], :282]
        xx = []
        yy = []
        bb = []
        commands = []
        current = None
        for key in ("baseline_ref", "minus_ref", "plus_ref"):
            ref = r[key]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as a:
                poses = a["controller_site_pose"][-2:]
                twist = a["source_twist"][-2].copy()
                if current is None:
                    current = poses[0].copy()
                    current_twist = twist.copy()
                np.testing.assert_allclose(current, poses[0], rtol=0, atol=1e-9)
                np.testing.assert_allclose(current_twist, twist, rtol=0, atol=1e-9)
                goal = np.r_[
                    a["controller_goal_pos"][-1] - current[:3, 3],
                    Rotation.from_matrix(a["controller_goal_ori"][-1] @ current[:3, :3].T).as_rotvec(),
                ]
                u = (goal[:3] - parameters["translation_mean"]) / parameters["translation_scale"]
                linear = np.r_[phi, *(phi * v for v in u)]
                value = expand(linear[None, :])[0]
                independent = np.concatenate([phi * m for m in [1, *u, *(v * v for v in u)]])
                np.testing.assert_allclose(value, independent, rtol=1e-12, atol=1e-12)
                xx.append(value)
                yy.append(
                    np.r_[
                        poses[1, :3, 3] - current[:3, 3],
                        Rotation.from_matrix(poses[1, :3, :3] @ current[:3, :3].T).as_rotvec(),
                        0.05 * a["source_twist"][-1],
                    ]
                )
                bb.append(nominal(dict(action_goal_delta=goal[None, :], x_site_twist=twist[None, :]), parameters)[0])
                commands.append(a["sent_commands"][-1].copy())
        delta = r["actual_normalized_delta"]
        axis = r["axis"]
        fraction = (commands[0][axis] - commands[1][axis]) / delta
        assert 0 < fraction < 1
        for values, odd, even in ((xx, odd_x, even_x), (yy, odd_y, even_y), (bb, odd_b, even_b)):
            odd.append((values[2] - values[1]) / delta)
            even.append(values[0] - (1 - fraction) * values[1] - fraction * values[2])
        np.testing.assert_allclose(odd_x[-1][:1128], old_x[4865 + j], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(odd_y[-1], y[4865 + j], rtol=0, atol=1e-9)
        np.testing.assert_allclose(odd_b[-1], base[4865 + j], rtol=0, atol=1e-9)
        np.testing.assert_allclose(even_x[-1][:1128], 0, rtol=0, atol=1e-10)
    x = np.vstack([natural, odd_x, even_x])
    target = np.vstack([y[:4865], odd_y, even_y])
    baseline = np.vstack([base[:4865], odd_b, even_b])
    assert x.shape == (5261, 1974) and target.shape == baseline.shape == (5261, 12)
    w = np.r_[weights[:4865], np.full(396, 0.5 / 396)]
    parameters.pop("coefficients")
    e.recheck()
    out = Path("outputs/quadratic_action_equations_20261025")
    out.mkdir(exist_ok=False)
    path = out / "training.npz"
    with path.open("xb") as stream:
        np.savez(stream, features=x, target=target, nominal=baseline, weights=w)
    write_once(out / "parameters.json", parameters)
    write_once(
        out / "result.json",
        dict(
            complete=True,
            rows=5261,
            design_dimension=1974,
            contrast_equations=198,
            curvature_equations=198,
            training_ref=dict(path=str(path.resolve()), sha256=sha256(path)),
            parameter_ref=dict(path=str((out / "parameters.json").resolve()), sha256=sha256(out / "parameters.json")),
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print("138 all5261x1974 equations branchwise verified;fit0")


if __name__ == "__main__":
    main()
