"""Compare complete, paired scene trajectories without changing the running policy."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from verify_wam_acceptance import verify_case

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.geometry import pose_error
from s2a.task_acceptance import task_outcome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")


def load_case(path, config, fixtures, variants):
    integrity = verify_case(path, config, fixtures, variants)
    manifest = json.loads(path.with_name("manifest.json").read_text())
    tool = Path(manifest["config"]["tool_contract"])
    h = np.asarray(json.loads(tool.read_text())["source_from_tcp"])
    steps = [json.loads(line) for line in path.with_name("steps.jsonl").read_text().splitlines()]
    measured = np.asarray([r["measured_source"] for r in steps]) @ h
    actual = np.asarray([r["after_source"] for r in steps]) @ h
    desired = np.asarray([r["desired_source"] for r in steps]) @ h
    error = pose_error(desired, actual)
    position = np.linalg.norm(error[:, :3], axis=1)
    rotation = np.linalg.norm(error[:, 3:], axis=1)
    fixture = fixtures[manifest["record"]["name"]]
    target = TARGETS[fixture["task_index"]]
    contact = np.asarray([r["objects"][target]["grasp_contact"] for r in steps], dtype=bool)
    success = np.asarray([r["success"] for r in steps], dtype=bool)
    time = (np.arange(len(steps)) + 1) / config["control_fps"]
    summary = dict(
        path=str(path),
        outcome=task_outcome(
            success,
            expected_steps=config["online_max_steps"],
            stable_steps=config["acceptance"]["stable_success_last_steps"],
        ),
        first_success_time_s=float(time[success][0]) if success.any() else None,
        first_grasp_contact_time_s=float(time[contact][0]) if contact.any() else None,
        grasp_contact_steps=int(contact.sum()),
        tracking={},
    )
    for name, mask in {"first_5s": time <= 5, "full_budget": np.ones(len(time), dtype=bool)}.items():
        summary["tracking"][name] = dict(
            steps=int(mask.sum()),
            tcp_position_rmse_m=float(np.sqrt(np.mean(position[mask] ** 2))),
            rotation_rmse_rad=float(np.sqrt(np.mean(rotation[mask] ** 2))),
            saturated_steps=sum(r["saturated"] for r, keep in zip(steps, mask, strict=True) if keep),
            rate_limited_steps=sum(r["rate_limited"] for r, keep in zip(steps, mask, strict=True) if keep),
        )
    arrays = dict(
        time=time,
        position_mm=position * 1000,
        rotation=rotation,
        opening_target=np.asarray([r["opening_target"] for r in steps]),
        opening_after=np.asarray([r["opening_after"] for r in steps]),
        object_height_mm=np.asarray([r["objects"][target]["height_gain_m"] for r in steps]) * 1000,
        contact=contact,
        success=success,
        measured=measured,
        desired=desired,
        actual=actual,
        object_position=np.asarray([r["objects"][target]["position"] for r in steps]),
    )
    return summary, arrays, integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_independent_20261020.json")
    parser.add_argument("--record", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("study seed differs")
    fixture_path = root / "simulation/fixtures.json"
    fixture_data = json.loads(fixture_path.read_text())
    if fixture_data["config"] != c:
        raise ValueError("frozen fixture configuration changed")
    fixtures = {r["name"]: r for r in fixture_data["records"]}
    if args.record not in fixtures:
        raise ValueError("record outside frozen scene set")
    variants = {r["name"]: r for r in c["checkpoints"]}
    # A scene figure requires all registered checkpoints and seeds for that scene.
    # It is never a complete-matrix acceptance summary.
    cases = {}
    for name in variants:
        cases[name] = []
        for seed in c["sampling_seeds"]:
            path = root / f"simulation/{name}/{args.record}_s{seed}/result.json"
            cases[name].append((seed, *load_case(path, c, fixtures, variants)))
    dest = root / "scene_diagnostics" / args.record
    dest.mkdir(parents=True, exist_ok=False)
    rows, integrity, figures = [], [], {}
    for name, paired in cases.items():
        fig, axes = plt.subplots(len(paired), 5, figsize=(21, 3.4 * len(paired)), squeeze=False)
        for axis, (seed, summary, data, check) in zip(axes, paired, strict=True):
            rows.append(dict(checkpoint=name, sampling_seed=seed, **summary))
            integrity.append(check)
            t = data["time"]
            short = t <= 5
            axis[0].plot(data["desired"][short, 0, 3], data["desired"][short, 1, 3], label="desired TCP")
            axis[0].plot(data["actual"][short, 0, 3], data["actual"][short, 1, 3], label="actual TCP", alpha=0.75)
            axis[0].plot(data["object_position"][short, 0], data["object_position"][short, 1], label="target object")
            axis[0].scatter(*data["measured"][0, :2, 3], color="black", marker="x", label="initial TCP")
            axis[0].set(xlabel="world x (m)", ylabel="world y (m)", title="First 5 s: XY projection")
            axis[0].set_aspect("equal", adjustable="datalim")
            axis[1].plot(t, data["position_mm"], label="TCP error")
            axis[1].axhline(c["qualification_position_tolerance_m"] * 1000, color="gray", linestyle=":")
            axis[1].set(xlabel="physical time (s)", ylabel="position error (mm)", title="Desired vs actual TCP")
            axis[2].plot(t, data["rotation"], label="SO(3) error")
            axis[2].axhline(c["qualification_rotation_tolerance_rad"], color="gray", linestyle=":")
            axis[2].set(
                xlabel="physical time (s)", ylabel="rotation error (rad)", title="Desired vs actual orientation"
            )
            axis[3].plot(t, data["opening_target"], label="desired opening")
            axis[3].plot(t, data["opening_after"], label="actual opening", alpha=0.75)
            axis[3].set(xlabel="physical time (s)", ylabel="opening [0,1]", title="Executed opening reference")
            axis[4].plot(t, data["object_height_mm"], label="target height gain")
            axis[4].scatter(
                t[data["contact"]], data["object_height_mm"][data["contact"]], s=9, label="native grasp contact"
            )
            axis[4].set(xlabel="physical time (s)", ylabel="height gain (mm)", title="Object motion; not task success")
            for a in axis[1:]:
                a.fill_between(
                    t, 0, 1, where=data["success"], transform=a.get_xaxis_transform(), color="green", alpha=0.1
                )
            for a in axis:
                a.grid(alpha=0.2)
                a.legend(fontsize=7)
            axis[0].text(0, 1.13, f"seed {seed}; stable={summary['outcome']['stable']}", transform=axis[0].transAxes)
        fig.suptitle(f"{name} / {args.record}; green shading = native task success; complete paired scene only")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for suffix in ("png", "pdf"):
            path = dest / f"{name}.{suffix}"
            fig.savefig(path, dpi=140)
            figures[str(path)] = sha256(path)
        plt.close(fig)
    write_json(
        dest / "diagnostics.json",
        dict(
            code=code_fingerprint(),
            config_sha256=sha256(args.config),
            fixtures_sha256=sha256(fixture_path),
            record=args.record,
            complete_matrix=False,
            interpretation="Paired scene diagnostics, not population success or unique causal attribution. Object-center XY is not a grasp target.",
            cases=rows,
            integrity=integrity,
            figures_sha256=figures,
        ),
    )
    print(dest)


if __name__ == "__main__":
    main()
