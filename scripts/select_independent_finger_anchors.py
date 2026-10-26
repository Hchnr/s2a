"""189 select training-only novel loaded-gap anchors, no evaluation target reads."""

import numpy as np
from arm_joint_wrench_independent import conditions
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence
from scipy.spatial.distance import cdist

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.joint_wrench_condition_forward import raw_condition, with_history


def main():
    out = REPO / "outputs/independent_finger_anchor_selection_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/189_independent_finger_anchor_selection_20261025.md")
    config = e.json("configs/left_table_forward_20261025.json")
    params = e.json("outputs/left_table_fit_20261025/parameters.json")
    equations = e.json("outputs/left_table_forward_equations_20261025/result.json")
    ref = equations["training_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        x = a["features"][4865:].copy()
    points = []
    for i in range(0, 207, 3):
        phi = x[i, 282:564] / x[i, 282]
        points.append(np.r_[phi[1:84], phi[212:218]])
    points = np.asarray(points)
    used = {}
    profiles = {}
    for name in (
        "training_response_coverage",
        "wrench_coverage_interventions",
        "condition_span_interventions",
        "left_table_interventions",
    ):
        registration = e.json(f"configs/{name}_20261025.json")
        for index, profile in enumerate(registration["profiles"]):
            native = profile["native_data_index"]
            used.setdefault(native, set()).add(registration["trials"][7 * index]["anchor_cycle"])
            profiles.setdefault(native, profile)
    source = e.json(config["data_result_ref"]["path"])
    candidates = []
    for entry in source["trajectories"]:
        native = entry["index"]
        if entry["split"] != "fit" or native in (0, 10, 20):
            continue
        assert native in profiles
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = with_history({k: a[k].copy() for k in a.files}, params)
        data = attach_current_wrench(data, entry, config, e)
        raw = raw_condition(data)
        np.testing.assert_allclose(raw, conditions(data), atol=1e-12, rtol=0)
        gap = raw[:, 43] - raw[:, 44]
        target = 0.04 + 0.02 * (raw[:, 47] - raw[:, 48])
        z = (raw - params["state_mean"]) / params["state_scale"]
        novelty = cdist(z, points).min(axis=1)
        eligible = (target <= 0.005) & (gap >= 0.005)
        for cycle in np.flatnonzero(eligible):
            cycle = int(cycle)
            if cycle in used.get(native, set()):
                continue
            candidates.append(
                dict(
                    native_index=native,
                    anchor_cycle=cycle,
                    distance=float(novelty[cycle]),
                    gap_m=float(gap[cycle]),
                    motor_target_gap_m=float(target[cycle]),
                    standardized_condition=z[cycle].tolist(),
                )
            )
    candidates.sort(key=lambda r: (-r["distance"], r["native_index"], r["anchor_cycle"]))
    selected = []
    seen = set()
    for row in candidates:
        if row["native_index"] in seen:
            continue
        selected.append(dict(row, profile=profiles[row["native_index"]], physics_steps=7 * (row["anchor_cycle"] + 1)))
        seen.add(row["native_index"])
        if len(selected) == 3:
            break
    e.recheck()
    publish(
        out / "result.json",
        dict(
            complete=True,
            training_only=True,
            selected=selected,
            candidates=candidates,
            excluded_profiles=[0, 10, 20],
            excluded_used_cycles={str(k): sorted(v) for k, v in used.items()},
            training_points=points.tolist(),
            state_mean=params["state_mean"],
            state_scale=params["state_scale"],
            planned_trials=7 * len(selected),
            planned_physics_steps=sum(r["physics_steps"] for r in selected),
            source_sha256=e.hashes,
            fits=0,
            physics_steps=0,
            independently_verified=False,
        ),
    )
    print(
        [(r["native_index"], r["anchor_cycle"], r["distance"]) for r in selected],
        sum(r["physics_steps"] for r in selected),
    )


if __name__ == "__main__":
    main()
