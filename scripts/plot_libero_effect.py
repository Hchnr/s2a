"""Export physical-time effect diagnostics; no inference or parameter fitting."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def trace(root, case, trial=0):
    path = root / "sync" / case / f"trial{trial}" / "steps.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(fig, root, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_effect_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    root = Path(c["output_dir"])
    summary = json.loads((root / "summary.json").read_text())
    require = summary["complete"] and summary["integrity_passed"]
    if not require:
        raise ValueError("verify complete experiments before plotting")
    dest = root / "figures"
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for column, case in enumerate(("oracle_response", "wam_affine", "wam_response")):
        rows = trace(root, case)
        actual = np.asarray([r["after_source"] for r in rows])[:, :3, 3]
        desired = np.asarray([r["desired_source"] for r in rows])[:, :3, 3]
        t = np.arange(len(rows)) / c["control_fps"]
        bowl, plate = np.asarray(rows[0]["bowl_position"]), np.asarray(rows[0]["plate_position"])
        ax = axes[0, column]
        ax.plot(desired[:, 0], desired[:, 1], "--", lw=1, label="desired source")
        ax.plot(actual[:, 0], actual[:, 1], lw=1, label="physical source")
        ax.scatter(*bowl[:2], marker="o", label="bowl start")
        ax.scatter(*plate[:2], marker="s", label="plate")
        ax.set(xlabel="world x [m]", ylabel="world y [m]", title=case.replace("_", " "))
        ax.axis("equal")
        ax.legend(fontsize=7)
        axes[1, column].plot(t, desired[:, 2], "--", lw=1, label="desired source z")
        axes[1, column].plot(t, actual[:, 2], lw=1, label="physical source z")
        axes[1, column].axhline(bowl[2], color="grey", ls=":", label="bowl initial center z")
        axes[1, column].set(xlabel="physical simulation time [s]", ylabel="world z [m]")
        axes[1, column].legend(fontsize=7)
    fig.suptitle("Fixed trial 0: shared-path oracle and live WAM; HTTP pauses physics")
    save(fig, dest, "effect_paths")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for case, color in (("wam_affine", "C0"), ("wam_response", "C1")):
        rows = trace(root, case)
        t = np.arange(len(rows)) / c["control_fps"]
        rotation = []
        opening = []
        for row in rows:
            desired, measured = np.asarray(row["desired_source"]), np.asarray(row["measured_source"])
            rotation.append(Rotation.from_matrix(desired[:3, :3] @ measured[:3, :3].T).magnitude())
            opening.append(row["opening_target"])
        axes[0].plot(t, rotation, lw=0.6, color=color, label=case)
        axes[1].plot(t, opening, lw=0.6, color=color, label=case)
    axes[0].set(xlabel="physical simulation time [s]", ylabel="requested orientation error [rad]")
    axes[1].set(xlabel="physical simulation time [s]", ylabel="clipped predicted opening [0, 1]")
    for ax in axes:
        ax.legend()
    fig.suptitle("Fixed trial 0: actual reference requests; opening clip preserves raw values in artifacts")
    save(fig, dest, "reference_requests")
    groups = summary["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = [r["case"].replace("_", " ") for r in groups]
    colors = ["C0" if r["case"].startswith("wam_") and "oracle" not in r["case"] else "C2" for r in groups]
    y = np.arange(len(groups))
    axes[0].barh(y, [r["terminal_successes"] for r in groups], color=colors)
    axes[0].set(xlabel="terminal successes / 5 known scenes", yticks=y, yticklabels=labels, xlim=(0, 5))
    axes[1].barh(y, [r["position_rmse_mm"] for r in groups], color=colors)
    axes[1].set(xlabel="source position tracking RMSE [mm]", yticks=y, yticklabels=[])
    fig.suptitle("Blue: pure WAM; green: privileged oracle controls / component interventions")
    save(fig, dest, "effect_groups")
    print(dest)


if __name__ == "__main__":
    main()
