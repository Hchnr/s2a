"""Export state-error curves and task counts from a complete acceptance matrix."""

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s2a.artifacts import load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    source = root / "summary.json"
    summary = json.loads(source.read_text())
    native_source = root / "native_horizon40/summary.json"
    native = json.loads(native_source.read_text())
    if not summary["complete"] or not native["complete"]:
        raise ValueError("figures require complete matched matrices")
    out = root / "figures"
    out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = ["warm6200_masked", "8400_masked", "8600_masked", "8800_masked"]
    labels = ["6200 warm-start", "8400", "8600", "8800"]
    colors = ["#8172b2", "#64b5cd", "#dd8452", "#4c72b0"]
    metrics = [
        ("position_rmse_m", 1000, "Position RMSE (mm)"),
        ("rotation_rmse_rad", 180 / np.pi, "SO(3) RMSE (degrees)"),
        ("opening_mae", 1, "Opening MAE (0–1)"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, split in enumerate(("diagnostic", "study_heldout")):
        for name, label, color in zip(names, labels, colors, strict=True):
            result = next(
                x
                for x in summary["state"]
                if x["checkpoint"] == name
                and x["split"] == split
                and x["prompt_mode"] == "task_only"
                and x["horizon"] == 40
            )
            series = [(result["model"]["endpoints_by_time"], label, color, "-")]
            if name == names[-1]:
                series += [
                    (result[k]["endpoints_by_time"], title, color, "--")
                    for k, title, color in (
                        ("hold", "Hold current", "#333333"),
                        ("constant_velocity", "History velocity", "#999999"),
                    )
                ]
            for column, (metric, scale, ylabel) in enumerate(metrics):
                ax = axes[row, column]
                for points, title, line_color, style in series:
                    ax.plot(
                        [x["time_s"] for x in points],
                        [x[metric] * scale for x in points],
                        label=title,
                        color=line_color,
                        linestyle=style,
                        marker="o",
                        markersize=3,
                    )
                ax.set(xlabel="Prediction time (s)", ylabel=ylabel, title=split.replace("_", " "))
                ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Frozen WAM state prediction after active-slot masking; task-only input\nStudy-heldout is not certified unseen during pretraining"
    )
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"state_errors.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for task_index, ax in enumerate(axes):
        success, terminal, counts = [], [], []
        comparisons = [summary["physics"][name]["cases"] for name in ["oracle", *names]]
        comparisons.append(native["physics"]["8800_masked"]["cases"])
        for cases in comparisons:
            rows = [r for r in cases if r["record"].startswith(f"task{task_index}_")]
            counts.append(len(rows))
            success.append(sum(r["success"] for r in rows))
            terminal.append(sum(r["terminal_success"] for r in rows))
        x = np.arange(len(success))
        ax.bar(x - 0.17, success, width=0.34, label="Ever successful", color="#4c72b0")
        ax.bar(x + 0.17, terminal, width=0.34, label="Terminal success", color="#55a868")
        ax.set_xticks(x, ["Oracle", "6200 H16", "8400 H16", "8600 H16", "8800 H16", "8800 H40"], rotation=35)
        ax.set(ylim=(0, max(counts) + 0.8), ylabel="Successful / 4 trials", title=c["tasks"][task_index])
        ax.set_yticks(range(max(counts) + 1))
        for i, (a, b) in enumerate(zip(success, terminal, strict=True)):
            ax.text(i - 0.17, a + 0.08, str(a), ha="center", fontsize=9)
            ax.text(i + 0.17, b + 0.08, str(b), ha="center", fontsize=9)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Fixed 600-step task evaluation; physics paused for inference\nTwo initial states × two sampling seeds per task; frozen response S2A"
    )
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"task_success.{suffix}", dpi=180)
    plt.close(figure)
    write_json(
        out / "manifest.json",
        dict(
            summary_sha256=sha256(source),
            native_horizon_summary_sha256=sha256(native_source),
            artifacts={p.name: sha256(p) for p in out.iterdir() if p.suffix in (".png", ".pdf")},
        ),
    )


if __name__ == "__main__":
    main()
