"""Independently verify cached two-seed consensus with SciPy Slerp.

Only source JSON/NPZ is read. No consensus implementation, model, simulator or
S2A decoder is imported. Truth is used for offline scoring, never seed choice.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]


def rotation(columns):
    first = columns[:3] / np.linalg.norm(columns[:3])
    second = columns[3:] - first * np.dot(first, columns[3:])
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, np.cross(first, second)))


def poses(states):
    result = np.repeat(np.eye(4)[None], len(states), axis=0)
    result[:, :3, 3] = states[:, :3]
    result[:, :3, :3] = np.asarray([rotation(row[3:9]) for row in states])
    return result


def integrate(anchor, actions):
    current = anchor.copy()
    result = []
    for row in actions:
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = rotation(row[3:9]), row[:3]
        current = current @ delta
        result.append(current.copy())
    return np.asarray(result)


def score(truth, prediction, opening, predicted_opening):
    return dict(
        position=np.linalg.norm(truth[:, :3, 3] - prediction[:, :3, 3], axis=1),
        rotation=Rotation.from_matrix(truth[:, :3, :3] @ prediction[:, :3, :3].transpose(0, 2, 1)).magnitude(),
        opening=np.asarray(predicted_opening) - opening,
    )


def aggregate(cases):
    result = {}
    keys = (
        ["all"]
        + sorted({"task:" + c["sample"]["task"] for c in cases})
        + sorted({"phase:" + c["sample"]["phase"] for c in cases})
    )
    for key in keys:
        selected = (
            cases if key == "all" else [c for c in cases if c["sample"][key.split(":")[0]] == key.split(":", 1)[1]]
        )
        result[key] = {}
        for prefix in (8, 16, 40):
            result[key][str(prefix)] = {}
            for method in ("wam", "hold", "constant_velocity"):
                p, r, o = (
                    np.concatenate([c["metrics"][method][v][:prefix] for c in selected])
                    for v in ("position", "rotation", "opening")
                )
                result[key][str(prefix)][method] = dict(
                    queries=len(selected),
                    episodes=len({c["sample"]["episode"] for c in selected}),
                    distinct_anchors=len({(c["sample"]["episode"], c["sample"]["anchor_frame"]) for c in selected}),
                    valid_endpoints=len(p),
                    full_prefix_queries=sum(len(c["metrics"][method]["position"]) >= prefix for c in selected),
                    position_rmse_m=float(np.sqrt(np.mean(p**2))),
                    position_rmse_mm=float(np.sqrt(np.mean(p**2)) * 1000),
                    rotation_rmse_rad=float(np.sqrt(np.mean(r**2))),
                    opening_mae=float(np.mean(np.abs(o))),
                )
    return result


def same(saved, actual):
    assert saved.keys() == actual.keys()
    maximum = 0.0
    for key in saved:
        if isinstance(saved[key], dict):
            maximum = max(maximum, same(saved[key], actual[key]))
        else:
            a, b = np.asarray(saved[key]), np.asarray(actual[key])
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            maximum = max(maximum, float(np.max(np.abs(a - b))))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/wam_stability_20261021/seed_consensus_diagnostic.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU audit requires empty CUDA and LD paths")
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        raise ValueError("requires a new repository output path")
    signatures = {}

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert str(path) not in signatures or signatures[str(path)] == digest
        signatures[str(path)] = digest
        return raw

    read(__file__)
    original = json.loads(read(args.source))
    assert original["complete"] and original["new_model_calls"] == original["new_physics_steps"] == 0
    assert original["inference_policy_changed"] and original["pure_s2a_compensation"] is False
    for path, digest in original["source_sha256"].items():
        assert hashlib.sha256(read(path)).hexdigest() == digest, path
    summaries = {}
    for label, saved in original["results"].items():
        config = json.loads(read(ROOT / f"configs/wam_stability_screen_{label}_20261021.json"))
        sample_manifest = json.loads(read(Path(config["output_dir"]) / "samples/manifest.json"))
        selected = sample_manifest["cases"]
        assert len(selected) == 24 and len({s["episode"] for s in selected}) == 6
        assert len({(s["episode"], s["anchor_frame"]) for s in selected}) == 22
        ck = config["checkpoints"][0]
        cases = {kind: [] for kind in ("seed42", "seed43", "consensus")}
        maximum_angle, midpoint_symmetry = 0.0, 0.0
        for sample in selected:
            read(sample["arrays_path"])
            with np.load(sample["arrays_path"]) as data:
                states = data["states"].copy()
            absolute, frame = poses(states), sample["anchor_frame"]
            count = min(40, sample["valid_future_steps"])
            truth, true_opening = absolute[frame + 1 : frame + 1 + count], states[frame + 1 : frame + 1 + count, 9]
            hold = np.repeat(absolute[frame : frame + 1], count, axis=0)
            constant = hold.copy()
            history = min(frame, 4)
            if history:
                k = np.arange(1, count + 1) / history
                constant[:, :3, 3] += k[:, None] * (absolute[frame, :3, 3] - absolute[frame - history, :3, 3])
                velocity = Rotation.from_matrix(
                    absolute[frame, :3, :3] @ absolute[frame - history, :3, :3].T
                ).as_rotvec()
                constant[:, :3, :3] = Rotation.from_rotvec(k[:, None] * velocity).as_matrix() @ absolute[frame, :3, :3]
            baselines = dict(
                hold=score(truth, hold, true_opening, np.repeat(states[frame, 9], count)),
                constant_velocity=score(truth, constant, true_opening, np.repeat(states[frame, 9], count)),
            )
            predictions, openings = [], []
            for seed in (42, 43):
                name = f"{sample['name']}_h40_task_only_s{seed}"
                path = Path(config["output_dir"]) / "state_queries" / ck["name"] / name / "response.json"
                response = json.loads(read(path))
                content = response["outputs"][0]["content"]
                assert response["status"] == "success" and content["action128"]["seed"] == seed
                expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
                assert Path(content["action128"]["checkpoint_path"]).resolve() == expected.resolve()
                actions = np.asarray(content["action"], dtype=float)
                assert actions.shape == (40, 128) and np.isfinite(actions).all() and not actions[:, 10:].any()
                prediction, opening = integrate(absolute[frame], actions[:count]), np.clip(actions[:count, 9], 0, 1)
                predictions.append(prediction)
                openings.append(opening)
                cases[f"seed{seed}"].append(
                    dict(sample=sample, metrics=dict(baselines, wam=score(truth, prediction, true_opening, opening)))
                )
            first, second = predictions
            middle = first.copy()
            middle[:, :3, 3] = np.mean(np.stack([first[:, :3, 3], second[:, :3, 3]]), axis=0)
            for i in range(count):
                rotations = Rotation.from_matrix(np.stack([first[i, :3, :3], second[i, :3, :3]]))
                angle = (rotations[0].inv() * rotations[1]).magnitude()
                assert angle < np.pi - 1e-6
                midpoint = Slerp([0.0, 1.0], rotations)([0.5])[0]
                middle[i, :3, :3] = midpoint.as_matrix()
                maximum_angle = max(maximum_angle, float(angle))
                midpoint_symmetry = max(
                    midpoint_symmetry,
                    abs(float((rotations[0].inv() * midpoint).magnitude()) - angle / 2),
                    abs(float((rotations[1].inv() * midpoint).magnitude()) - angle / 2),
                )
            cases["consensus"].append(
                dict(
                    sample=sample,
                    metrics=dict(
                        baselines, wam=score(truth, middle, true_opening, np.mean(np.stack(openings), axis=0))
                    ),
                )
            )
        discrepancies = {kind: same(saved[kind], aggregate(items)) for kind, items in cases.items()}
        np.testing.assert_allclose(maximum_angle, saved["maximum_between_seed_rotation_rad"], atol=1e-12, rtol=0)
        summaries[label] = dict(
            aggregates_match=True,
            max_abs_aggregate_discrepancy=discrepancies,
            max_midpoint_half_distance_residual_rad=midpoint_symmetry,
            maximum_between_seed_rotation_rad=maximum_angle,
            phase_entries=24,
            distinct_anchors=22,
            episodes=6,
            h40_consensus=aggregate(cases["consensus"])["all"]["40"]["wam"],
        )
    for path, digest in list(signatures.items()):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    value = dict(
        schema_version="s2a.seed-consensus-independent-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        verified=True,
        method="Independent scalar GS/integration, SciPy Slerp midpoint, independent GT/baselines/strata aggregation",
        groups=summaries,
        original_bound_inputs_verified=len(original["source_sha256"]),
        input_sha256=signatures,
        new_model_calls=0,
        new_physics_steps=0,
        boundaries=[
            "24 phase entries are22 distinct episode/anchor pairs from6 historical diagnostic episodes, not24 independent scenes.",
            "Midpoint changes inference policy; it is not raw single-seed checkpoint or pure S2A compensation.",
            "No task-success, fresh-heldout or pretraining-unseen claim follows from offline state errors.",
            "Both fixed seeds are used equally; truth is used only for scoring, never seed selection.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(payload)
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), groups=summaries)))


if __name__ == "__main__":
    main()
