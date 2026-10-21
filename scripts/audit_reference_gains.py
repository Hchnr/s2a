"""Independently rebuild008 folds, constrained fits and metrics from cached arrays.

Uses direct matrix integration and augmented least squares. It imports neither
production ReferenceGains nor any production pooling/geometry/metric helper.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("raw", "calibrated", "no_wam", "shuffled", "position_only", "hold", "constant_velocity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(columns):
    first, second = np.asarray(columns[:3], dtype=float), np.asarray(columns[3:], dtype=float)
    assert np.linalg.norm(first) >= 1e-8
    first = first / np.linalg.norm(first)
    second = second - first * np.dot(first, second)
    assert np.linalg.norm(second) >= 1e-8
    second = second / np.linalg.norm(second)
    return np.column_stack((first, second, np.cross(first, second)))


def features(anchor, poses):
    positions = np.asarray([anchor[:3, :3].T @ (pose[:3, 3] - anchor[:3, 3]) for pose in poses])
    rotations = np.asarray([Rotation.from_matrix(anchor[:3, :3].T @ pose[:3, :3]).as_rotvec() for pose in poses])
    assert np.all(np.linalg.norm(rotations, axis=1) < np.pi - 1e-6)
    return positions, rotations


def load_cases(label):
    config = json.loads((ROOT / f"configs/wam_stability_screen_{label}_20261021.json").read_text())
    root = ROOT / config["output_dir"]
    samples = json.loads((root / "samples/manifest.json").read_text())["cases"]
    queries = sorted(
        (f"{sample['name']}_h40_task_only_s{seed}", sample, seed) for sample in samples for seed in (42, 43)
    )
    unique, aliases = {}, []
    for name, sample, seed in queries:
        assert sha(sample["arrays_path"]) == sample["arrays_sha256"]
        with np.load(sample["arrays_path"]) as arrays:
            states = np.asarray(arrays["states"], dtype=float)
        matrices = np.repeat(np.eye(4)[None], len(states), axis=0)
        matrices[:, :3, 3] = states[:, :3]
        matrices[:, :3, :3] = [rotation(state[3:9]) for state in states]
        frame = sample["anchor_frame"]
        count = min(40, sample["valid_future_steps"], len(states) - frame - 1)
        anchor = matrices[frame]
        path = root / "state_queries" / config["checkpoints"][0]["name"] / name / "response.json"
        actions = np.asarray(json.loads(path.read_text())["outputs"][0]["content"]["action"], dtype=float)
        current, predictions = anchor.copy(), []
        for action in actions[:count]:
            relative = np.eye(4)
            relative[:3, 3], relative[:3, :3] = action[:3], rotation(action[3:9])
            current = current @ relative
            predictions.append(current.copy())
        predictions = np.asarray(predictions)
        history = np.repeat(anchor[None], count, axis=0)
        history_steps = min(frame, 4)
        if history_steps:
            prior = matrices[frame - history_steps]
            angle = Rotation.from_matrix(anchor[:3, :3] @ prior[:3, :3].T).as_rotvec()
            for step in range(1, count + 1):
                history[step - 1, :3, 3] += step / history_steps * (anchor[:3, 3] - prior[:3, 3])
                history[step - 1, :3, :3] = (
                    Rotation.from_rotvec(step / history_steps * angle).as_matrix() @ anchor[:3, :3]
                )
        key = (sample["episode"], frame, seed)
        item = dict(
            name=name,
            key=key,
            episode=sample["episode"],
            anchor=anchor,
            anchor_opening=float(states[frame, 9]),
            predicted=predictions,
            opening=actions[:count, 9],
            truth=matrices[frame + 1 : frame + 1 + count],
            true_opening=states[frame + 1 : frame + 1 + count, 9],
            count=count,
            history=history,
        )
        item["predicted_features"], item["truth_features"] = (
            features(anchor, predictions),
            features(anchor, item["truth"]),
        )
        if key in unique:
            for field in ("anchor", "predicted", "opening", "truth", "true_opening"):
                np.testing.assert_array_equal(unique[key][field], item[field])
            aliases.append(dict(duplicate=name, retained=unique[key]["name"], key=list(key)))
        else:
            unique[key] = item
    assert len(unique) == 44 and len(aliases) == 4
    return [unique[key] for key in sorted(unique)], aliases


def pairs(fit, test):
    rng = np.random.default_rng(42)
    episode_order = sorted({c["episode"] for c in fit})
    rng.shuffle(episode_order)
    ordered = []
    for episode in episode_order:
        block = [c for c in fit if c["episode"] == episode]
        rng.shuffle(block)
        ordered.extend(block)
    offset = max(Counter(c["episode"] for c in ordered).values())
    donors = ordered[offset:] + ordered[:offset]
    lookup = {left["name"]: right for left, right in zip(ordered, donors, strict=True)}
    fit_pairs = [(c, lookup[c["name"]]) for c in fit]
    selection = rng.permutation(len(fit))
    test_pairs = [(c, fit[selection[i % len(selection)]]) for i, c in enumerate(test)]
    assert all(c["episode"] != d["episode"] for c, d in fit_pairs + test_pairs)
    return fit_pairs, test_pairs


def view(case, donor, count, method):
    if method == "no_wam":
        return np.repeat(case["anchor"][None], count, axis=0), np.full(count, case["anchor_opening"])
    if method != "shuffled":
        return case["predicted"][:count], case["opening"][:count]
    positions, rotations = donor["predicted_features"]
    output = np.repeat(case["anchor"][None], count, axis=0)
    for i in range(count):
        output[i, :3, 3] += case["anchor"][:3, :3] @ positions[i]
        output[i, :3, :3] = case["anchor"][:3, :3] @ Rotation.from_rotvec(rotations[i]).as_matrix()
    return output, donor["opening"][:count]


def fit_augmented(paired, method, config):
    counts = Counter(c["episode"] for c, _ in paired)
    x, y, opening_x, opening_y, weights, episode_weights = [], [], [], [], [], Counter()
    for case, donor in paired:
        count = min(case["count"], donor["count"])
        pose, opening = view(case, donor, count, method)
        x.append(features(case["anchor"], pose))
        y.append(tuple(v[:count] for v in case["truth_features"]))
        opening_x.extend(opening)
        opening_y.extend(case["true_opening"][:count])
        weight = 1 / (len(counts) * counts[case["episode"]] * count)
        weights.extend([weight] * count)
        episode_weights[case["episode"]] += count * weight
    weights = np.asarray(weights)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(list(episode_weights.values()), 1 / 5, atol=1e-12, rtol=0)
    ridge = config["ridge"]
    gains = []
    for channel, scale, upper in ((0, config["position_scale_m"], 2), (1, config["rotation_scale_rad"], 1)):
        inputs = np.concatenate([v[channel] for v in x]) / scale
        labels = np.concatenate([v[channel] for v in y]) / scale
        design = np.r_[(np.sqrt(weights)[:, None] * inputs).ravel(), np.sqrt(ridge)][:, None]
        targets = np.r_[(np.sqrt(weights)[:, None] * labels).ravel(), np.sqrt(ridge)]
        optimum = np.linalg.lstsq(design, targets, rcond=None)[0][0]
        gains.append(float(np.clip(optimum, 0, upper)))
    scale = config["opening_scale"]
    design = np.r_[np.sqrt(weights) / scale, np.sqrt(ridge)][:, None]
    targets = np.r_[np.sqrt(weights) * (np.asarray(opening_y) - opening_x) / scale, 0]
    bias = np.linalg.lstsq(design, targets, rcond=None)[0][0]
    return dict(position_gain=gains[0], rotation_gain=gains[1], opening_bias=float(np.clip(bias, -0.25, 0.25)))


def apply(case, poses, opening, gains):
    position, angles = features(case["anchor"], poses)
    result = poses.copy()
    for i in range(len(result)):
        if gains["position_gain"] != 1:
            result[i, :3, 3] = case["anchor"][:3, 3] + case["anchor"][:3, :3] @ (gains["position_gain"] * position[i])
        if gains["rotation_gain"] != 1:
            result[i, :3, :3] = (
                case["anchor"][:3, :3] @ Rotation.from_rotvec(gains["rotation_gain"] * angles[i]).as_matrix()
            )
    return result, np.clip(opening + gains["opening_bias"], 0, 1)


def metrics(case, poses, opening, count):
    truth = case["truth"][:count]
    position = np.linalg.norm(truth[:, :3, 3] - poses[:, :3, 3], axis=1)
    angles = [Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude() for a, b in zip(truth, poses, strict=True)]
    grip = opening - case["true_opening"][:count]
    return dict(
        endpoints=count,
        position_rmse_m=float(np.sqrt(np.mean(position**2))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.square(angles)))),
        opening_mae=float(np.mean(np.abs(grip))),
        position_errors_m=position.tolist(),
        rotation_errors_rad=angles,
        opening_errors=grip.tolist(),
    )


def aggregate(cases):
    result = {}
    for prefix in (8, 16, 40):
        result[str(prefix)] = {}
        for method in METHODS:
            p = np.concatenate([c[method]["position_errors_m"][:prefix] for c in cases])
            r = np.concatenate([c[method]["rotation_errors_rad"][:prefix] for c in cases])
            g = np.concatenate([c[method]["opening_errors"][:prefix] for c in cases])
            result[str(prefix)][method] = dict(
                endpoints=len(p),
                position_rmse_m=float(np.sqrt(np.dot(p, p) / len(p))),
                rotation_rmse_rad=float(np.sqrt(np.dot(r, r) / len(r))),
                opening_mae=float(np.sum(np.abs(g)) / len(g)),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("independent008 audit requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("output must be a new repository artifact")
    artifact_sha = sha(args.artifact)
    assert artifact_sha == "903baa946ad26a89bdb500328e3c108d1b661183f6c01963d30b1d5260b5f269"
    artifact = json.loads(args.artifact.read_text())
    assert artifact["complete"] and not artifact["physical_promotion_authorized"]
    for path, digest in artifact["input_sha256"].items():
        assert sha(path) == digest
    config = json.loads((ROOT / "configs/reference_gain_diagnostic_20261021.json").read_text())
    maximum, verified = dict(parameter=0.0, state_metric=0.0, aggregate=0.0), {}

    def compare(saved, actual, kind):
        assert set(saved) == set(actual)
        for key in saved:
            a, b = np.asarray(saved[key]), np.asarray(actual[key])
            np.testing.assert_allclose(a, b, atol=2e-12, rtol=0, err_msg=f"{kind}:{key}")
            maximum[kind] = max(maximum[kind], float(np.max(np.abs(a - b))))

    for label in config["checkpoints"]:
        cases, aliases = load_cases(label)
        saved = artifact["results"][label]
        assert aliases == saved["duplicate_queries_removed"] and len({c["episode"] for c in cases}) == 6
        assert len(saved["folds"]) == 6
        assert {fold["held_episode"] for fold in saved["folds"]} == {c["episode"] for c in cases}
        all_methods = []
        tested_keys = []
        for fold in saved["folds"]:
            held = fold["held_episode"]
            fit, test = [c for c in cases if c["episode"] != held], [c for c in cases if c["episode"] == held]
            assert sorted({c["episode"] for c in fit}) == fold["fit_episodes"] and len(fold["fit_episodes"]) == 5
            fit_pairs, test_pairs = pairs(fit, test)
            assert fold["fit_pairs"] == [
                dict(query=c["name"], donor=d["name"], common_endpoints=min(c["count"], d["count"]))
                for c, d in fit_pairs
            ]
            models = {
                method: fit_augmented(fit_pairs, method, config) for method in ("calibrated", "no_wam", "shuffled")
            }
            for method, model in models.items():
                compare(fold["models"][method], model, "parameter")
            outputs = []
            for (case, donor), existing in zip(test_pairs, fold["test_cases"], strict=True):
                tested_keys.append(case["key"])
                assert existing["query"] == case["name"] and existing["key"] == list(case["key"])
                assert (
                    existing["shuffled_donor"] == donor["name"]
                    and existing["shuffled_donor_episode"] == donor["episode"]
                )
                assert donor["episode"] in fold["fit_episodes"] and case["episode"] == held
                count = min(case["count"], donor["count"])
                assert existing["valid_endpoints"] == count and existing["original_valid_endpoints"] == case["count"]
                own, opening = view(case, donor, count, "calibrated")
                identity, identity_opening = apply(
                    case, own, opening, dict(position_gain=1, rotation_gain=1, opening_bias=0)
                )
                np.testing.assert_array_equal(identity, own)
                np.testing.assert_array_equal(identity_opening, np.clip(opening, 0, 1))
                values = dict(raw=metrics(case, own, np.clip(opening, 0, 1), count))
                for method, model in dict(
                    models,
                    position_only=dict(
                        position_gain=models["calibrated"]["position_gain"], rotation_gain=1, opening_bias=0
                    ),
                ).items():
                    source, gripper = view(case, donor, count, method)
                    transformed, gripper = apply(case, source, gripper, model)
                    values[method] = metrics(case, transformed, gripper, count)
                hold = np.repeat(case["anchor"][None], count, axis=0)
                values["hold"] = metrics(case, hold, np.full(count, case["anchor_opening"]), count)
                values["constant_velocity"] = metrics(
                    case, case["history"][:count], np.full(count, case["anchor_opening"]), count
                )
                for method in METHODS:
                    compare(existing["methods"][method], values[method], "state_metric")
                outputs.append(values)
            aggregate_fold = aggregate(outputs)
            for prefix in ("8", "16", "40"):
                for method in METHODS:
                    compare(fold["metrics"][prefix][method], aggregate_fold[prefix][method], "aggregate")
            all_methods.extend(outputs)
        assert len(all_methods) == len(tested_keys) == len(set(tested_keys)) == 44
        assert set(tested_keys) == {c["key"] for c in cases}
        pooled = aggregate(all_methods)
        for prefix in ("8", "16", "40"):
            for method in METHODS:
                compare(saved["pooled"][prefix][method], pooled[prefix][method], "aggregate")
        verified[label] = dict(
            unique_queries=len(cases),
            duplicate_queries_removed=len(aliases),
            whole_episode_folds=6,
            fit_episodes_per_fold=5,
            no_held_episode_fit_or_shuffle_donor_leak=True,
            equal_episode_query_endpoint_weights_verified=True,
            pooled=pooled,
        )
    assert sha(args.artifact) == artifact_sha
    for path, digest in artifact["input_sha256"].items():
        assert sha(path) == digest
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        artifact_path=str(args.artifact),
        artifact_sha256=artifact_sha,
        independently_rechecked_input_files=len(artifact["input_sha256"]),
        folds_rebuilt=18,
        three_parameter_models_refit=54,
        held_queries_rebuilt=132,
        per_query_method_results_rebuilt=924,
        maximum_absolute_recompute_difference=maximum,
        results=verified,
        production_reference_gains_imported=False,
        production_pooling_imported=False,
        fit_solver="Independent augmented scalar least squares with identity/zero ridge rows, followed by fixed box clipping",
        objective_limits="Rotation fits anchor-relative log-vector quadratic surrogate, not geodesic loss; opening fits preclip squared error and is clipped only for evaluation.",
        new_model_queries=0,
        new_physics_steps=0,
        physical_promotion_authorized=False,
        scope="Six previously viewed development episodes only. Whole-episode cross-diagnosis is not fresh heldout, pretraining-unseen evidence or closed-loop task success.",
        audit_script_sha256=sha(__file__),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), complete=True, maximum_difference=maximum)))


if __name__ == "__main__":
    main()
