"""Read-only equal-weight two-seed consensus of already generated references.

World TCP positions/openings use their arithmetic midpoint and orientation uses
the shortest SO(3) geodesic midpoint. This changes the inference policy, is not
S2A compensation, and never selects a seed using future truth. Ground truth is
read only for offline scoring on the original diagnostic development cases.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from summarize_wam_stability_screen import LABELS, Evidence, aggregate, verify_cell

from s2a.artifacts import write_json
from s2a.checkpoint_audit import state_metrics
from s2a.geometry import integrate_local, pose9_to_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/wam_stability_20261021"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    if args.output.exists():
        raise FileExistsError("diagnostic snapshots are immutable; choose a new output")
    evidence = Evidence()
    for path in (
        __file__,
        "scripts/summarize_wam_stability_screen.py",
        "src/s2a/geometry.py",
        "src/s2a/checkpoint_audit.py",
    ):
        evidence.read(path)
    results = {}
    for label in LABELS:
        config, selected, verified, progress = verify_cell(label, args.root, evidence)
        if not progress["complete"]:
            raise ValueError("all four original screens must complete before paired consensus scoring")
        checkpoint = config["checkpoints"][0]["name"]
        grouped = {sample["name"]: {} for sample in selected}
        for name, case in verified.items():
            grouped[case["sample"]["name"]][case["sampling_seed"]] = (name, case)
        consensus, disagreement, maximum_relative_angle = [], [], 0.0
        for pair in grouped.values():
            if set(pair) != {42, 43}:
                raise ValueError("requires paired seeds42/43 on the identical observed anchor")
            name, case = pair[42]
            sample = case["sample"]
            if sample != pair[43][1]["sample"]:
                raise ValueError("seed observations differ")
            with np.load(sample["arrays_path"]) as data:
                states = data["states"]
            poses, frame = pose9_to_transform(states[:, :9]), sample["anchor_frame"]
            count = case["metrics"]["wam"]["endpoints"]
            predicted, openings = [], []
            for seed in (42, 43):
                path = Path(config["output_dir"]) / "state_queries" / checkpoint / pair[seed][0] / "response.json"
                actions = np.asarray(evidence.json(path)["outputs"][0]["content"]["action"])
                predicted.append(integrate_local(poses[frame], actions[:count, :9])[1:])
                openings.append(np.clip(actions[:count, 9], 0, 1))
            first, second = predicted
            relative = first[:, :3, :3].transpose(0, 2, 1) @ second[:, :3, :3]
            angular = Rotation.from_matrix(relative).as_rotvec()
            maximum_relative_angle = max(maximum_relative_angle, float(np.max(np.linalg.norm(angular, axis=1))))
            if np.any(np.linalg.norm(angular, axis=1) >= np.pi - 1e-6):
                raise ValueError("two-seed rotation midpoint is ambiguous at pi; no silent branch selection")
            middle = first.copy()
            middle[:, :3, 3] = (first[:, :3, 3] + second[:, :3, 3]) / 2
            middle[:, :3, :3] = first[:, :3, :3] @ Rotation.from_rotvec(angular / 2).as_matrix()
            reverse = second[:, :3, :3] @ Rotation.from_rotvec(-angular / 2).as_matrix()
            np.testing.assert_allclose(middle[:, :3, :3], reverse, atol=1e-12, rtol=0)
            midpoint_opening = (openings[0] + openings[1]) / 2
            truth = poses[frame + 1 : frame + 1 + count]
            true_opening = states[frame + 1 : frame + 1 + count, 9]
            metrics = state_metrics(truth, middle, true_opening, midpoint_opening)
            consensus.append(dict(case, metrics=dict(case["metrics"], wam=metrics)))
            disagreement.append(
                dict(sample=sample["name"], metrics=state_metrics(first, second, openings[0], openings[1]))
            )
        results[label] = dict(
            seed42=aggregate([c for c in verified.values() if c["sampling_seed"] == 42]),
            seed43=aggregate([c for c in verified.values() if c["sampling_seed"] == 43]),
            consensus=aggregate(consensus),
            between_seed_disagreement=disagreement,
            maximum_between_seed_rotation_rad=maximum_relative_angle,
            anchors=len(consensus),
        )
    evidence.recheck()
    write_json(
        args.output,
        dict(
            complete=True,
            scope="Original diagnostic development anchors only; no fresh heldout or physical task-success claim",
            method="Equal midpoint of two generated world-TCP paths: linear position/opening, shortest SO3 midpoint",
            task_or_future_truth_used_by_consensus=False,
            inference_policy_changed=True,
            pure_s2a_compensation=False,
            new_model_calls=0,
            new_physics_steps=0,
            source_sha256=evidence.hashes,
            results=results,
        ),
    )
    print(args.output)
    for label, result in results.items():
        print(label, {m: result[m]["all"]["40"]["wam"] for m in ("seed42", "seed43", "consensus")})


if __name__ == "__main__":
    main()
