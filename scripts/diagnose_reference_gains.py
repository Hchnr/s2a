"""Fit three reference gains with whole-episode folds on cached WAM outputs.

Development diagnosis only: no model, simulator, task planner or fresh fixture
is loaded. Parameters and controls were registered in008 before fitting.
"""

import argparse
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from summarize_wam_earlier_screen import verify_earlier
from summarize_wam_stability_screen import Evidence, equal_metrics, verify_cell

from s2a.artifacts import REPO, write_json
from s2a.checkpoint_audit import extrapolate_history, state_metrics
from s2a.geometry import integrate_local, pose9_to_transform
from s2a.reference_gains import ReferenceGains, relative_features


def load_cases(label, root, evidence):
    config, _, verified, progress = (
        verify_earlier(label, root, evidence) if label == "8400_g10" else verify_cell(label, root, evidence)
    )
    assert progress["complete"] and len(verified) == 48
    checkpoint = config["checkpoints"][0]["name"]
    unique, aliases = {}, []
    for name, case in sorted(verified.items()):
        sample = case["sample"]
        raw = np.asarray(
            evidence.json(Path(config["output_dir"]) / "state_queries" / checkpoint / name / "response.json")[
                "outputs"
            ][0]["content"]["action"]
        )
        with np.load(sample["arrays_path"]) as data:
            states = data["states"]
        poses = pose9_to_transform(states[:, :9])
        frame, count = sample["anchor_frame"], case["metrics"]["wam"]["endpoints"]
        anchor = poses[frame]
        predicted = integrate_local(anchor, raw[:count, :9])[1:]
        truth = poses[frame + 1 : frame + 1 + count]
        true_opening = states[frame + 1 : frame + 1 + count, 9]
        equal_metrics(
            state_metrics(truth, predicted, true_opening, np.clip(raw[:count, 9], 0, 1)), case["metrics"]["wam"]
        )
        key = (sample["episode"], frame, case["sampling_seed"])
        item = dict(
            name=name,
            key=key,
            episode=sample["episode"],
            anchor=anchor,
            anchor_opening=float(states[frame, 9]),
            predicted=predicted,
            opening=raw[:count, 9],
            truth=truth,
            true_opening=true_opening,
            predicted_features=relative_features(anchor, predicted),
            truth_features=relative_features(anchor, truth),
            history=extrapolate_history(poses[max(0, frame - 4) : frame + 1], count),
            count=count,
        )
        if key in unique:
            prior = unique[key]
            for field in ("anchor", "predicted", "opening", "truth", "true_opening"):
                np.testing.assert_array_equal(prior[field], item[field])
            aliases.append(dict(duplicate=name, retained=prior["name"], key=key))
        else:
            unique[key] = item
    cases = [unique[key] for key in sorted(unique)]
    assert len(cases) == 44 and len(aliases) == 4 and len({c["episode"] for c in cases}) == 6
    return cases, aliases


def shuffled_pairs(fit, test, seed):
    """Whole-query derangement within fit; test donors come only from fit."""
    rng = np.random.default_rng(seed)
    groups = sorted({c["episode"] for c in fit})
    rng.shuffle(groups)
    ordered = []
    for group in groups:
        block = [c for c in fit if c["episode"] == group]
        rng.shuffle(block)
        ordered.extend(block)
    shift = max(Counter(c["episode"] for c in ordered).values())
    donors = ordered[shift:] + ordered[:shift]
    assert all(left["episode"] != right["episode"] for left, right in zip(ordered, donors, strict=True))
    lookup = {left["name"]: right for left, right in zip(ordered, donors, strict=True)}
    fit_pairs = [(c, lookup[c["name"]]) for c in fit]
    test_order = rng.permutation(len(fit))
    test_pairs = [(c, fit[test_order[i % len(test_order)]]) for i, c in enumerate(test)]
    assert all(left["episode"] != right["episode"] for left, right in test_pairs)
    return fit_pairs, test_pairs


def view(case, donor, count, method):
    if method == "no_wam":
        return np.repeat(case["anchor"][None], count, axis=0), np.full(count, case["anchor_opening"])
    if method != "shuffled":
        return case["predicted"][:count], case["opening"][:count]
    position, rotation = (v[:count] for v in donor["predicted_features"])
    anchor = case["anchor"]
    poses = np.repeat(anchor[None], count, axis=0)
    poses[:, :3, 3] += position @ anchor[:3, :3].T
    poses[:, :3, :3] = anchor[:3, :3] @ Rotation.from_rotvec(rotation).as_matrix()
    return poses, donor["opening"][:count]


def fit_model(pairs, method, config):
    groups = Counter(case["episode"] for case, _ in pairs)
    predicted, truth, openings, true_openings, weights = [], [], [], [], []
    for case, donor in pairs:
        count = min(case["count"], donor["count"])
        poses, opening = view(case, donor, count, method)
        predicted.append(relative_features(case["anchor"], poses))
        truth.append(tuple(v[:count] for v in case["truth_features"]))
        openings.append(opening)
        true_openings.append(case["true_opening"][:count])
        weights.append(np.full(count, 1 / (len(groups) * groups[case["episode"]] * count)))
    return ReferenceGains.fit(
        tuple(np.concatenate([f[channel] for f in predicted]) for channel in (0, 1)),
        tuple(np.concatenate([f[channel] for f in truth]) for channel in (0, 1)),
        np.concatenate(openings),
        np.concatenate(true_openings),
        np.concatenate(weights),
        ridge=config["ridge"],
        scales=[config["position_scale_m"], config["rotation_scale_rad"], config["opening_scale"]],
    )


def metrics(case, poses, opening, count):
    return state_metrics(case["truth"][:count], poses, case["true_opening"][:count], opening)


def pooled(cases, prefixes):
    methods = list(cases[0]["methods"])
    result = {}
    for prefix in prefixes:
        result[str(prefix)] = {}
        for method in methods:
            position, rotation, opening = (
                np.concatenate([c["methods"][method][field][:prefix] for c in cases])
                for field in ("position_errors_m", "rotation_errors_rad", "opening_errors")
            )
            result[str(prefix)][method] = dict(
                endpoints=len(position),
                position_rmse_m=float(np.sqrt(np.mean(position**2))),
                rotation_rmse_rad=float(np.sqrt(np.mean(rotation**2))),
                opening_mae=float(np.mean(np.abs(opening))),
            )
    return result


def run_fold(cases, held_episode, config):
    fit = [c for c in cases if c["episode"] != held_episode]
    test = [c for c in cases if c["episode"] == held_episode]
    assert len({c["episode"] for c in fit}) == 5 and not ({c["episode"] for c in fit} & {c["episode"] for c in test})
    fit_pairs, test_pairs = shuffled_pairs(fit, test, config["seed"])
    models = {method: fit_model(fit_pairs, method, config) for method in ("calibrated", "no_wam", "shuffled")}
    outputs = []
    for case, donor in test_pairs:
        count = min(case["count"], donor["count"])
        own, opening = view(case, donor, count, "calibrated")
        identity_poses, identity_opening = ReferenceGains().apply(case["anchor"], own, opening)
        np.testing.assert_array_equal(identity_poses, own)
        np.testing.assert_array_equal(identity_opening, np.clip(opening, 0, 1))
        values = {"raw": metrics(case, own, np.clip(opening, 0, 1), count)}
        for method, gains in dict(
            models, position_only=ReferenceGains(models["calibrated"].position_gain, 1, 0)
        ).items():
            source, source_opening = view(case, donor, count, method)
            transformed, transformed_opening = gains.apply(case["anchor"], source, source_opening)
            values[method] = metrics(case, transformed, transformed_opening, count)
        values["hold"] = metrics(
            case, np.repeat(case["anchor"][None], count, axis=0), np.full(count, case["anchor_opening"]), count
        )
        values["constant_velocity"] = metrics(
            case, case["history"][:count], np.full(count, case["anchor_opening"]), count
        )
        outputs.append(
            dict(
                query=case["name"],
                key=case["key"],
                shuffled_donor=donor["name"],
                shuffled_donor_episode=donor["episode"],
                valid_endpoints=count,
                original_valid_endpoints=case["count"],
                methods=values,
            )
        )
    return dict(
        held_episode=held_episode,
        fit_episodes=sorted({c["episode"] for c in fit}),
        fit_pairs=[
            dict(query=c["name"], donor=d["name"], common_endpoints=min(c["count"], d["count"])) for c, d in fit_pairs
        ],
        models={method: asdict(gains) for method, gains in models.items()},
        test_cases=outputs,
        metrics=pooled(outputs, config["prefixes"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reference_gain_diagnostic_20261021.json")
    parser.add_argument("--output-root", help="new artifact directory only; does not change the registered fit")
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("run CPU-only from repository with empty CUDA/LD paths")
    evidence = Evidence()
    config = evidence.json(args.config)
    expected = dict(
        seed=42,
        checkpoints=["8400_g10", "8800_g10", "15400_g10"],
        ridge=0.1,
        position_scale_m=0.1,
        rotation_scale_rad=0.5,
        opening_scale=1.0,
        position_gain_bounds=[0.0, 2.0],
        rotation_gain_bounds=[0.0, 1.0],
        opening_bias_bounds=[-0.25, 0.25],
        prefixes=[8, 16, 40],
        split="leave_one_complete_episode_out",
        deduplicate=["episode", "anchor_frame", "sampling_seed"],
        new_model_queries=0,
        new_physics_steps=0,
        physical_promotion_authorized=False,
    )
    assert {k: config[k] for k in expected} == expected
    root = (REPO / (args.output_root or config["output_dir"])).resolve()
    if not root.is_relative_to(REPO / "outputs") or root.exists():
        raise ValueError("requires a new repository output directory")
    for name in [
        "scripts/diagnose_reference_gains.py",
        "src/s2a/reference_gains.py",
        "tests/test_reference_gains.py",
        "scripts/summarize_wam_stability_screen.py",
        "scripts/summarize_wam_earlier_screen.py",
        "src/s2a/geometry.py",
        "src/s2a/checkpoint_audit.py",
        "plans/v0.0.1/008_reference_gain_diagnostic_20261021.md",
    ]:
        evidence.read(name)
    results = {}
    for label in config["checkpoints"]:
        cases, aliases = load_cases(label, REPO / "outputs/wam_stability_20261021", evidence)
        folds = [run_fold(cases, episode, config) for episode in sorted({c["episode"] for c in cases})]
        outputs = [c for fold in folds for c in fold["test_cases"]]
        assert len(outputs) == len(cases) == 44
        results[label] = dict(
            unique_queries=44,
            duplicate_queries_removed=aliases,
            folds=folds,
            pooled=pooled(outputs, config["prefixes"]),
        )
    evidence.recheck()
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "diagnostic.json",
        dict(
            complete=True,
            seed=42,
            scope="Six already-viewed development episodes; no independent final test or pretraining-unseen claim",
            new_model_queries=0,
            new_physics_steps=0,
            identity_exact=True,
            results=results,
            input_sha256=evidence.hashes,
            physical_promotion_authorized=False,
        ),
    )
    print(root / "diagnostic.json")


if __name__ == "__main__":
    main()
