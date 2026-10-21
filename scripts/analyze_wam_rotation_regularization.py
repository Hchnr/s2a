"""Diagnose task-neutral SO(3) attenuation on the frozen development screens."""

import argparse
import os
from pathlib import Path

import numpy as np
from summarize_wam_stability_screen import LABELS, Evidence, aggregate, equal_metrics, verify_cell

from s2a.artifacts import write_json
from s2a.checkpoint_audit import state_metrics
from s2a.geometry import integrate_local, pose9_to_transform
from s2a.wam_regularization import attenuate_wam_rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/wam_stability_20261021"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU-only audit requires empty CUDA and LD paths")
    output = args.output or args.root / "rotation_regularization_diagnostic.json"
    if output.exists():
        raise FileExistsError("use a new output path; completed diagnostics are not overwritten")
    evidence = Evidence()
    for source in (
        "scripts/analyze_wam_rotation_regularization.py",
        "scripts/summarize_wam_stability_screen.py",
        "src/s2a/wam_regularization.py",
    ):
        evidence.read(source)
    results = {}
    for label in LABELS:
        config, _, verified, progress = verify_cell(label, args.root, evidence)
        if not progress["complete"]:
            raise ValueError("all four diagnostic screens must finish before this paired analysis")
        checkpoint = config["checkpoints"][0]["name"]
        groups = {str(alpha): [] for alpha in (0.0, 0.5, 1.0)}
        for name, case in verified.items():
            sample = case["sample"]
            path = Path(config["output_dir"]) / "state_queries" / checkpoint / name / "response.json"
            actions = np.asarray(evidence.json(path)["outputs"][0]["content"]["action"])
            with np.load(sample["arrays_path"]) as data:
                states = data["states"]
            poses = pose9_to_transform(states[:, :9])
            frame = sample["anchor_frame"]
            count = case["metrics"]["wam"]["endpoints"]
            truth = poses[frame + 1 : frame + 1 + count]
            opening = states[frame + 1 : frame + 1 + count, 9]
            for alpha in (0.0, 0.5, 1.0):
                modified = attenuate_wam_rotation(actions, alpha=alpha)
                predicted = integrate_local(poses[frame], modified[:count, :9])[1:]
                metrics = state_metrics(truth, predicted, opening, np.clip(modified[:count, 9], 0, 1))
                if alpha == 1:
                    equal_metrics(metrics, case["metrics"]["wam"])
                groups[str(alpha)].append(dict(case, metrics=dict(case["metrics"], wam=metrics)))
        results[label] = {alpha: aggregate(cases) for alpha, cases in groups.items()}
    evidence.recheck()
    write_json(
        output,
        dict(
            complete=True,
            scope="Old diagnostic development samples only; no heldout or physical task-success claim",
            intervention="Model output rotation attenuation, not S2A response compensation or raw WAM performance",
            alpha_1_exact_raw_regression=True,
            alpha_0_diagnostic_only=True,
            alpha_05_future_candidate="Moderate task-neutral hypothesis from prior diagnostics; requires live WAM validation",
            local_translation_and_absolute_opening_unchanged=True,
            world_translation_can_change_through_local_rotation_integration=True,
            new_model_calls=0,
            source_sha256=evidence.hashes,
            results=results,
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
