"""Scientific plots for the frozen LIBERO control audit."""

import argparse
import json
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path("outputs/libero_audit_20261019/matplotlib").resolve())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
if args.seed != 42:
    raise ValueError("seed differs from frozen experiment")
out = Path("outputs/libero_audit_20261019/figures")
out.mkdir(exist_ok=True)
names = [
    "native_measured",
    "affine_position",
    "response_position",
    "affine_intent",
    "response_intent",
    "affine_latch",
    "response_latch",
]
labels = [
    "Native expert",
    "Affine + position",
    "PD inverse + position",
    "Affine + intent",
    "PD inverse + intent",
    "Affine + latch",
    "PD inverse + latch",
]
roots = [Path("outputs/libero_audit_20261019"), Path("outputs/libero_holdout_20261019")]
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
rng = np.random.default_rng(args.seed)
for column, (root, title) in enumerate(
    zip(roots, ["5 previously inspected episodes", "10 preselected additional episodes"])
):
    rows = json.loads((root / "matrix.json").read_text())["results"]
    for i, name in enumerate(names):
        r = [a for a in rows if a["case"]["name"] == name]
        n = len(r)
        success = sum(a["success"] for a in r)
        axes[0, column].bar(i, success / n, color="tab:blue" if name.startswith("affine") else "tab:orange", alpha=0.7)
        axes[0, column].text(i, success / n + 0.025, f"{success}/{n}", ha="center", fontsize=9)
        y = [a["position_rmse_m"] * 1000 for a in r]
        axes[1, column].scatter(i + rng.uniform(-0.15, 0.15, n), y, s=17, alpha=0.65)
        axes[1, column].plot(i, np.mean(y), "k_", ms=15, mew=2)
    axes[0, column].set(title=title, ylabel="Task success fraction", ylim=(0, 1.18))
    axes[1, column].set(ylabel="Position RMSE (mm), logarithmic", yscale="log")
    for ax in axes[:, column]:
        ax.set_xticks(range(len(names)), labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
fig.suptitle("Oracle reference: tracking error and task success are separate outcomes")
for ext in ["png", "pdf"]:
    fig.savefig(out / f"oracle_control_comparison.{ext}", dpi=180)
plt.close(fig)
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
for name, label in [
    ("response_position", "PD + position"),
    ("response_intent", "PD + intent"),
    ("response_latch", "PD + latch"),
    ("native_measured", "Native expert"),
]:
    rows = [json.loads(a) for a in (roots[0] / name / "trial4/steps.jsonl").read_text().splitlines()]
    t = np.arange(len(rows)) * 0.05
    axes[0].plot(t, [np.linalg.norm(a["error"][:3]) * 1000 for a in rows], label=label)
    axes[1].plot(t, [a["opening_after"] for a in rows])
    axes[2].step(t, [a["actuator_control"][0] * 1000 for a in rows], where="post")
    z0 = rows[0]["bowl_position"][2]
    axes[3].plot(t, [(a["bowl_position"][2] - z0) * 1000 for a in rows])
    if name == "response_position":
        axes[1].plot(t, [a["opening_target"] for a in rows], "k--", lw=1, label="Expert opening reference")
axes[0].set(ylabel="OSC control-point error (mm)", title="Episode 132: aperture matching and closing commands differ")
axes[0].legend(ncol=2)
axes[1].set(ylabel="Normalized opening")
axes[1].legend()
axes[2].set(ylabel="Finger actuator target (mm)")
axes[3].set(ylabel="Bowl lift (mm)", xlabel="Elapsed simulated time (s)")
for ax in axes:
    ax.grid(alpha=0.2)
for ext in ["png", "pdf"]:
    fig.savefig(out / f"gripper_contact_diagnostic.{ext}", dpi=180)
print(out)
