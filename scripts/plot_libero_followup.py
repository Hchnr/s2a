"""Export source-matched prediction and continuous-physics timing figures."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import load_config
from s2a.geometry import rot6d_to_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    summary = json.loads((out / "summary.json").read_text())
    if not summary["complete"] or not summary["integrity_passed"]:
        raise ValueError("figures require complete independently verified results")
    images = out / "training_image_queries"
    action = np.asarray(
        json.loads((images / "trial0_h16/response_0000.json").read_text())["outputs"][0]["content"]["action"]
    )
    matched = json.loads((out / "alignment/matched_targets.json").read_text())
    record = next(r for r in matched["cases"] if r["trial"] == 0)
    reference = np.asarray([r["action"] for r in record["rows"][:16]])
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})

    def save(figure, name):
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(figures / f"{name}.{suffix}", dpi=160)
        plt.close(figure)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    x = np.arange(1, 17)
    for values, label in ((action, "WAM prediction"), (reference, "Matched R2 label")):
        axes[0].semilogy(x, Rotation.from_matrix(rot6d_to_matrix(values[:, 3:9])).magnitude(), ".-", label=label)
        axes[1].plot(x, values[:, 9], ".-", label=label)
    axes[0].set(
        xlabel="Predicted step", ylabel="Local rotation (rad)", title="Rotation magnitude; source-matched image"
    )
    axes[1].axhspan(0, 1, color="green", alpha=0.06, label="Opening range [0, 1]")
    axes[1].set(xlabel="Predicted step", ylabel="Raw opening", title="Opening before clipping")
    for axis in axes:
        axis.legend()
    fig.suptitle("Offline WAM target diagnostics, trial 0; no task-success claim", y=1.04)
    save(fig, "published_image_targets")

    dest = out / "async/live_affine/trial0"
    rows = [json.loads(line) for line in (dest / "steps.jsonl").read_text().splitlines()]
    events = json.loads((dest / "events.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    wall = np.asarray([r["clock_s"] for r in rows])
    physical = np.asarray([r["simulation_elapsed_s"] for r in rows])
    axes[0].plot(wall, physical, label="Actual MuJoCo time")
    axes[0].plot([0, wall[-1]], [0, wall[-1]], "--", color="gray", label="Equal wall/physics time")
    for i, event in enumerate(events):
        axes[0].axvline(event["observed_at"], color="blue", alpha=0.18, label="Image capture" if i == 0 else None)
        if "admitted_at" in event:
            axes[0].axvline(event["admitted_at"], color="red", alpha=0.4, label="Expired response" if i == 0 else None)
    axes[0].set(xlabel="Wall time (s)", ylabel="Physical time (s)", title="Physics advances during HTTP inference")
    axes[0].legend()
    admitted = [e for e in events if "age_s" in e]
    axes[1].bar(np.arange(len(admitted)), [e["age_s"] for e in admitted], label="Response age at arrival")
    axes[1].axhline(c["async_max_age_s"], color="red", linestyle="--", label="Reference validity: 0.4 s")
    axes[1].set(xlabel="Completed request", ylabel="Observation age (s)", title="Expired targets are never executed")
    axes[1].legend()
    save(fig, "continuous_physics_timing")


if __name__ == "__main__":
    main()
