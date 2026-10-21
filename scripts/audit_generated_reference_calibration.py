"""Independently reconstruct023 preparation or its fixed cached-reference fits.

Only the frozen008 independent cache reader/shuffle implementation is reused.
The023 production feature, solver, apply, metric and decision code is not imported.
Preparation mode reconstructs labels/designs but performs zero parameter fits.
"""

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_reference_gains import load_cases, pairs, view
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = "6247a296793b35ef1c998ee0a72677657dbd11d834e083f1891f8cd17dbdbf61"
CONFIG = "configs/generated_reference_calibration_20261021.json"
PLAN = "plans/v0.0.1/023_generated_reference_calibration_20261021.md"
METHODS = ("raw", "identity", "paired", "hold", "constant_velocity", "no_wam", "shuffled")
FIT_METHODS = ("paired", "no_wam", "shuffled")
ALGORITHM_FIELDS = (
    "position_scale_m",
    "rotation_scale_rad",
    "maximum_horizon_seconds",
    "ridge",
    "maximum_translation_correction_speed_m_s",
    "maximum_rotation_vector_correction_speed_rad_s",
    "maximum_opening_correction_speed_s",
    "relative_rotation_pi_margin_rad",
    "control_fps",
)
TOLERANCE = 2e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Comparison:
    def __init__(self):
        self.maximum = {}

    def check(self, actual, expected, category):
        if isinstance(expected, dict):
            assert isinstance(actual, dict) and set(actual) == set(expected), (category, set(actual), set(expected))
            for key in expected:
                self.check(actual[key], expected[key], category)
        elif isinstance(expected, (bool, str)) or expected is None:
            assert type(actual) is type(expected) and actual == expected, (category, actual, expected)
        elif isinstance(expected, (tuple, list)) and expected and isinstance(expected[0], (dict, str)):
            assert len(actual) == len(expected), category
            for left, right in zip(actual, expected, strict=True):
                self.check(left, right, category)
        else:
            actual, expected = np.asarray(actual), np.asarray(expected)
            assert actual.shape == expected.shape, (category, actual.shape, expected.shape)
            if expected.dtype.kind in "biu":
                np.testing.assert_array_equal(actual, expected, err_msg=category)
            else:
                np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0, err_msg=category)
            difference = float(np.max(np.abs(actual - expected))) if actual.size and expected.dtype.kind != "b" else 0.0
            self.maximum[category] = max(self.maximum.get(category, 0.0), difference)


def body_features(anchor, opening0, poses, openings):
    count = len(poses)
    times = np.arange(1, count + 1, dtype=float) / 20
    assert count > 0 and np.all(times <= 2)
    u = np.asarray([anchor[:3, :3].T @ (pose[:3, 3] - anchor[:3, 3]) for pose in poses])
    v = np.asarray([Rotation.from_matrix(anchor[:3, :3].T @ pose[:3, :3]).as_rotvec() for pose in poses])
    assert np.all(np.linalg.norm(v, axis=1) < np.pi - 1e-6)
    normalized = np.column_stack((u / 0.1, v / 0.5))
    return dict(
        pose_features=np.column_stack((normalized, times / 2)),
        opening_features=np.column_stack((np.asarray(openings) - opening0, times / 2)),
        normalized_local_pose=normalized,
        local_position_m=u,
        anchor_relative_rotation_vector_rad=v,
        raw_openings=np.asarray(openings).copy(),
        endpoint_times_seconds=times,
    )


def records_and_weights(paired, origin=None):
    groups = Counter(case["episode"] for case, _ in paired)
    records, weights = [], []
    for case, donor in paired:
        count = min(case["count"], donor["count"])
        weight = 1 / (len(groups) * groups[case["episode"]] * count)
        weights.extend([weight] * count)
        records.append(
            dict(
                query=case["name"],
                query_key=list(case["key"]),
                episode=case["episode"],
                donor=donor["name"],
                donor_episode=donor["episode"],
                common_endpoints=count,
                weight_per_endpoint=weight,
                query_total_weight=float(np.full(count, weight).sum()),
                original_query_endpoints=case["count"],
                donor_endpoints=donor["count"],
                prefix_origin_held_episode=origin,
            )
        )
    weights = np.asarray(weights)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12, rtol=0)
    for episode in groups:
        np.testing.assert_allclose(
            sum(row["query_total_weight"] for row in records if row["episode"] == episode),
            1 / len(groups),
            atol=1e-12,
            rtol=0,
        )
    return records, weights


def fit_arrays(paired, method):
    records, weights = records_and_weights(paired)
    blocks = {
        key: []
        for key in (
            "pose_features",
            "pose_residuals",
            "opening_features",
            "opening_residuals",
            "endpoint_times_seconds",
        )
    }
    offset, inputs = 0, []
    for (case, donor), record in zip(paired, records, strict=True):
        count = record["common_endpoints"]
        pose, opening = view(case, donor, count, method)
        predicted = body_features(case["anchor"], case["anchor_opening"], pose, opening)
        truth = body_features(
            case["anchor"], case["anchor_opening"], case["truth"][:count], case["true_opening"][:count]
        )
        blocks["pose_features"].append(predicted["pose_features"])
        blocks["pose_residuals"].append(truth["normalized_local_pose"] - predicted["normalized_local_pose"])
        blocks["opening_features"].append(predicted["opening_features"])
        blocks["opening_residuals"].append(case["true_opening"][:count] - opening)
        blocks["endpoint_times_seconds"].append(predicted["endpoint_times_seconds"])
        inputs.append(
            dict(
                query=case["name"],
                donor=donor["name"],
                anchor=case["anchor"].copy(),
                anchor_opening=case["anchor_opening"],
                generated_world_poses=pose.copy(),
                generated_raw_openings=opening.copy(),
                fit_label_world_poses=case["truth"][:count].copy(),
                fit_label_openings=case["true_opening"][:count].copy(),
            )
        )
        record.update(endpoint_offset=offset, endpoint_stop=offset + count)
        offset += count
    arrays = {name: np.concatenate(values) for name, values in blocks.items()}
    arrays["endpoint_weights"] = weights
    arrays["input_queries"] = inputs
    return arrays, records


def augmented_fit(paired, method, config):
    arrays, records = fit_arrays(paired, method)
    x, y, xg, yg, w = (
        arrays[name]
        for name in (
            "pose_features",
            "pose_residuals",
            "opening_features",
            "opening_residuals",
            "endpoint_weights",
        )
    )
    a = np.vstack((np.sqrt(w)[:, None] * x, np.sqrt(0.1) * np.eye(7)))
    b = np.vstack((np.sqrt(w)[:, None] * y, np.zeros((7, 6))))
    ag = np.vstack((np.sqrt(w)[:, None] * xg, np.sqrt(0.1) * np.eye(2)))
    bg = np.r_[np.sqrt(w) * yg, np.zeros(2)]
    pose, opening = np.linalg.lstsq(a, b, rcond=None)[0], np.linalg.lstsq(ag, bg, rcond=None)[0]
    model = dict(pose_weights=pose, opening_weights=opening, config={key: config[key] for key in ALGORITHM_FIELDS})
    return model, dict(
        method=method,
        model=model,
        fit_episodes=sorted({case["episode"] for case, _ in paired}),
        fit_queries=records,
        training_endpoints=len(w),
        labels_are_fit_only=True,
        **arrays,
        pose_normal_matrix=x.T @ (w[:, None] * x) + 0.1 * np.eye(7),
        pose_right_hand_side=x.T @ (w[:, None] * y),
        opening_normal_matrix=xg.T @ (w[:, None] * xg) + 0.1 * np.eye(2),
        opening_right_hand_side=xg.T @ (w * yg),
    )


def corrected_reference(case, source, opening, model):
    features = body_features(case["anchor"], case["anchor_opening"], source, opening)
    raw = features["pose_features"] @ np.asarray(model["pose_weights"])
    times = features["endpoint_times_seconds"]
    dp, dv = raw[:, :3] * 0.1, raw[:, 3:] * 0.5
    pcap, rcap = np.linalg.norm(dp, axis=1) > times * 0.05, np.linalg.norm(dv, axis=1) > times * 0.25
    for values, speed in ((dp, 0.05), (dv, 0.25)):
        for index, value in enumerate(values):
            norm = float(np.linalg.norm(value))
            if norm > times[index] * speed:
                values[index] *= times[index] * speed / norm
    vectors = features["anchor_relative_rotation_vector_rad"] + dv
    assert np.all(np.linalg.norm(vectors, axis=1) < np.pi - 1e-6)
    output = source.copy()
    for index in range(len(source)):
        if np.any(dp[index] != 0):
            output[index, :3, 3] = case["anchor"][:3, 3] + case["anchor"][:3, :3] @ (
                features["local_position_m"][index] + dp[index]
            )
        if np.any(dv[index] != 0):
            output[index, :3, :3] = case["anchor"][:3, :3] @ Rotation.from_rotvec(vectors[index]).as_matrix()
    rawg = features["opening_features"] @ np.asarray(model["opening_weights"])
    dg = np.clip(rawg, -times * 0.25, times * 0.25)
    output_opening = np.clip(opening + dg, 0, 1)
    diagnostics = dict(
        **features,
        unconstrained_normalized_pose_residual=raw,
        translation_correction_m=dp,
        rotation_vector_correction_rad=dv,
        translation_cap_active=pcap,
        rotation_cap_active=rcap,
        unconstrained_opening_residual=rawg,
        opening_correction=dg,
        opening_cap_active=np.abs(rawg) > times * 0.25,
        pose_identity_weights=bool(not np.asarray(model["pose_weights"]).any()),
    )
    return output, output_opening, diagnostics


def metrics(case, pose, opening):
    truth = case["truth"][: len(pose)]
    p = np.linalg.norm(truth[:, :3, 3] - pose[:, :3, 3], axis=1)
    r = np.asarray(
        [Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude() for a, b in zip(truth, pose, strict=True)]
    )
    g = np.clip(opening, 0, 1) - case["true_opening"][: len(pose)]
    return dict(
        endpoints=len(pose),
        position_rmse_m=float(np.linalg.norm(p) / np.sqrt(len(p))),
        rotation_rmse_rad=float(np.linalg.norm(r) / np.sqrt(len(r))),
        opening_mae=float(np.mean(np.abs(g))),
        position_errors_m=p,
        rotation_errors_rad=r,
        opening_errors=g,
    )


def aggregate(cases):
    pooled, scores = {}, {}
    for prefix in (8, 16, 40):
        pooled[str(prefix)], scores[str(prefix)] = {}, {}
        for method in METHODS:
            p, r, g = [
                np.concatenate([case["methods"][method][key][:prefix] for case in cases])
                for key in (
                    "position_errors_m",
                    "rotation_errors_rad",
                    "opening_errors",
                )
            ]
            pooled[str(prefix)][method] = dict(
                endpoints=len(p),
                position_rmse_m=float(np.sqrt(np.mean(p * p))),
                rotation_rmse_rad=float(np.sqrt(np.mean(r * r))),
                opening_mae=float(np.mean(np.abs(g))),
            )
            scores[str(prefix)][method] = dict(
                endpoints=len(p), mean_squared_sum=float(np.mean((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2))
            )
    return pooled, scores


def decision(pooled, scores, folds):
    components, short = [], []
    for prefix in (8, 16, 40):
        for metric in ("position_rmse_m", "rotation_rmse_rad", "opening_mae"):
            raw, paired = (pooled[str(prefix)][method][metric] for method in ("raw", "paired"))
            components.append(dict(horizon=prefix, metric=metric, raw=raw, paired=paired, passed=bool(paired <= raw)))
        if prefix in (8, 16):
            raw, paired = (scores[str(prefix)][method]["mean_squared_sum"] for method in ("raw", "paired"))
            controls = {
                method: scores[str(prefix)][method]["mean_squared_sum"]
                for method in ("hold", "constant_velocity", "no_wam", "shuffled")
            }
            short.append(
                dict(
                    horizon=prefix,
                    raw=raw,
                    paired=paired,
                    controls=controls,
                    at_least_ten_percent_raw_improvement=bool(raw > 0 and paired <= 0.9 * raw),
                    strictly_better_than_all_controls=bool(all(paired < value for value in controls.values())),
                )
            )
    episodes = [
        dict(
            held_episode=fold["held_episode"],
            both_short_scores_nonregressing=bool(
                all(
                    fold["normalized_scores"][str(prefix)]["paired"]["mean_squared_sum"]
                    <= fold["normalized_scores"][str(prefix)]["raw"]["mean_squared_sum"]
                    for prefix in (8, 16)
                )
            ),
        )
        for fold in folds
    ]
    passing = sum(row["both_short_scores_nonregressing"] for row in episodes)
    eligible = (
        all(row["passed"] for row in components)
        and all(
            row["at_least_ten_percent_raw_improvement"] and row["strictly_better_than_all_controls"] for row in short
        )
        and passing >= 5
    )
    return dict(
        eligible_for_further_validation=bool(eligible),
        all_prefix_component_checks=components,
        short_score_checks=short,
        episode_checks=episodes,
        nonregressing_both_short_episode_folds=passing,
        required_episode_folds=5,
        added_comparison_tolerance=0,
        full_candidate_excluded=True,
        independent_acceptance_passed=False,
        physical_promotion_authorized=False,
    )


def preparation(cases):
    plans, full_pairs = [], []
    for held in sorted({case["episode"] for case in cases}):
        fit = [case for case in cases if case["episode"] != held]
        test = [case for case in cases if case["episode"] == held]
        fit_pairs, test_pairs = pairs(fit, test)
        assert len({case["episode"] for case in fit}) == 5
        assert all(donor["episode"] != held for _, donor in fit_pairs + test_pairs)
        fit_records, _ = records_and_weights(fit_pairs)
        test_records, _ = records_and_weights(test_pairs, held)
        fit_inputs = {}
        for method in FIT_METHODS:
            arrays, records = fit_arrays(fit_pairs, method)
            fit_inputs[method] = dict(fit_queries=records, **arrays)
        plans.append(
            dict(
                held_episode=held,
                fit_episodes=sorted({case["episode"] for case in fit}),
                fit_pair_records=fit_records,
                test_pair_records=test_records,
                fit_inputs=fit_inputs,
            )
        )
        full_pairs.extend(test_pairs)
    full_arrays, full_records = fit_arrays(full_pairs, "paired")
    for record in full_records:
        record["prefix_origin_held_episode"] = record["episode"]
    assert len(full_pairs) == len({tuple(c["key"]) for c, _ in full_pairs}) == 44
    assert len(full_arrays["endpoint_weights"]) == 1398
    return dict(
        folds=plans,
        full_candidate_fit_queries=full_records,
        full_candidate_fit_inputs=full_arrays,
        full_candidate_training_endpoints=1398,
        expected_fit_count=19,
        eligible_for_further_validation=None,
        full_candidate_prefix_rule="same44 OOF008/018 test donor counts fixed without labels; all-six paired fit only",
    ), full_pairs


def cached_sources():
    config_path = ROOT / "configs/wam_stability_screen_8400_g10_20261021.json"
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_dir"]
    manifest = root / "samples/manifest.json"
    files = [config_path, manifest]
    for sample in json.loads(manifest.read_text())["cases"]:
        files.append(Path(sample["arrays_path"]))
        for seed in (42, 43):
            files.append(
                root
                / "state_queries"
                / config["checkpoints"][0]["name"]
                / f"{sample['name']}_h40_task_only_s{seed}"
                / "response.json"
            )
    return {str(path.resolve()): sha(path) for path in files}


def verify_source(artifact):
    sources = dict(artifact["input_sha256"])
    for path, digest in sources.items():
        assert sha(path) == digest, path
    for relative, commit in (
        (CONFIG, "fab0f78"),
        (PLAN, "fab0f78"),
        ("src/s2a/generated_reference_calibration.py", "f481660"),
        ("scripts/diagnose_generated_reference_calibration.py", "691e3c0"),
    ):
        digest = hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)).hexdigest()
        assert sources[str(ROOT / relative)] == sha(ROOT / relative) == digest
    assert sha(ROOT / "scripts/audit_reference_gains.py") == HELPER_SHA
    inputs = cached_sources()
    assert all(sources[path] == digest for path, digest in inputs.items())
    sources.update(inputs)
    sources[str(ROOT / "scripts/audit_reference_gains.py")] = HELPER_SHA
    return sources


def verify_full(artifact, cases, config, expected_preparation, full_pairs, compare):
    outputs, folds, invocations = [], [], []
    for held, saved in zip(sorted({c["episode"] for c in cases}), artifact["folds"], strict=True):
        fit, test = [c for c in cases if c["episode"] != held], [c for c in cases if c["episode"] == held]
        fit_pairs, test_pairs = pairs(fit, test)
        planned = next(p for p in expected_preparation["folds"] if p["held_episode"] == held)
        for field in ("held_episode", "fit_episodes", "fit_pair_records", "test_pair_records"):
            compare.check(saved[field], planned[field], "fold_metadata")
        models = {}
        assert set(saved["fitted_models"]) == set(FIT_METHODS)
        for method in FIT_METHODS:
            model, expected = augmented_fit(fit_pairs, method, config)
            compare.check(saved["fitted_models"][method], expected, "fit")
            models[method] = model
            invocations.append(dict(held_episode=held, method=method))
        fold_outputs = []
        for (case, donor), actual in zip(test_pairs, saved["test_cases"], strict=True):
            count = min(case["count"], donor["count"])
            own, opening = view(case, donor, count, "paired")
            values = dict(raw=metrics(case, own, opening), identity=metrics(case, own, opening))
            references = {}
            for method, model in models.items():
                source, source_opening = view(case, donor, count, method)
                pose, grip, diagnostics = corrected_reference(case, source, source_opening, model)
                values[method] = metrics(case, pose, grip)
                references[method] = dict(
                    anchor=case["anchor"],
                    anchor_opening=case["anchor_opening"],
                    input_world_poses=source,
                    input_raw_openings=source_opening,
                    corrected_world_poses=pose,
                    corrected_openings=grip,
                    diagnostics=diagnostics,
                )
            values["hold"] = metrics(
                case, np.repeat(case["anchor"][None], count, axis=0), np.full(count, case["anchor_opening"])
            )
            values["constant_velocity"] = metrics(case, case["history"][:count], np.full(count, case["anchor_opening"]))
            expected = dict(
                query=case["name"],
                key=list(case["key"]),
                episode=case["episode"],
                task=case["task"],
                phase=case["phase"],
                shuffled_donor=donor["name"],
                shuffled_donor_episode=donor["episode"],
                valid_endpoints=count,
                original_valid_endpoints=case["count"],
                endpoint_times_seconds=np.arange(1, count + 1) / 20,
                methods=values,
                corrected_references=references,
            )
            compare.check(actual, expected, "heldout_case")
            assert actual["methods"]["raw"] == actual["methods"]["identity"]
            fold_outputs.append(expected)
        pooled, scores = aggregate(fold_outputs)
        compare.check(saved["metrics"], pooled, "fold_metrics")
        compare.check(saved["normalized_scores"], scores, "fold_scores")
        folds.append(dict(held_episode=held, normalized_scores=scores))
        outputs.extend(fold_outputs)
    assert len(outputs) == len({tuple(c["key"]) for c in outputs}) == 44
    pooled, scores = aggregate(outputs)
    assert [pooled[str(p)]["raw"]["endpoints"] for p in (8, 16, 40)] == [352, 704, 1398]
    compare.check(artifact["pooled"], pooled, "pooled")
    compare.check(artifact["normalized_scores"], scores, "scores")
    for field in ("task", "episode", "phase"):
        expected_groups = {}
        for value in sorted({c[field] for c in outputs}, key=str):
            p, s = aggregate([c for c in outputs if c[field] == value])
            expected_groups[str(value)] = dict(metrics=p, normalized_scores=s)
        compare.check(artifact["groups"][field], expected_groups, "reporting_groups")
    model, candidate = augmented_fit(full_pairs, "paired", config)
    candidate.update(
        scope="all-six training candidate; not heldout scored or deployed",
        fit_queries=expected_preparation["full_candidate_fit_queries"],
        prefix_provenance=expected_preparation["full_candidate_fit_queries"],
    )
    compare.check(artifact["full_candidate"], candidate, "full_candidate")
    invocations.append(dict(held_episode=None, method="paired", scope="training-only all-six candidate"))
    assert artifact["fit_invocations"] == invocations and len(invocations) == artifact["fit_count"] == 19
    expected_decision = decision(pooled, scores, folds)
    compare.check(artifact["decision"], expected_decision, "decision")
    assert artifact["eligible_for_further_validation"] is expected_decision["eligible_for_further_validation"]
    return dict(
        registered_fits_verified=19,
        pooled=pooled,
        normalized_scores=scores,
        decision=expected_decision,
        full_candidate_parameters={
            "pose_weights": model["pose_weights"].tolist(),
            "opening_weights": model["opening_weights"].tolist(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preparation", type=Path, help="reconstruct designs and weights with zero fits")
    mode.add_argument("--artifact", type=Path, help="reconstruct the registered completed19 fits")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or any(
        os.environ.get(name) != "" for name in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
    ):
        raise ValueError("run CPU-only in S2A with explicitly empty CUDA and LD")
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("requires a new repository output artifact")
    path = (args.preparation or args.artifact).resolve()
    artifact_sha = sha(path)
    artifact = json.loads(path.read_text())
    preparation_only = args.preparation is not None
    schema = (
        "s2a.generated-reference-calibration-preparation.v1"
        if preparation_only
        else "s2a.generated-reference-calibration-development-diagnostic.v1"
    )
    assert artifact["schema_version"] == schema and artifact["complete"] is True
    assert artifact["registered_commit"] == "fab0f78"
    assert artifact["fit_count"] == (0 if preparation_only else 19)
    assert (
        artifact["new_model_queries"] == artifact["new_physics_steps"] == 0
        and artifact["physical_promotion_authorized"] is False
    )
    config = json.loads((ROOT / CONFIG).read_text())
    assert artifact["config"] == config
    sources = verify_source(artifact)
    cases, aliases = load_cases("8400_g10")
    manifest = json.loads(
        (ROOT / "outputs/wam_stability_20261021/earlier_screening/8400_g10/samples/manifest.json").read_text()
    )
    for case in cases:
        sample = next(s for s in manifest["cases"] if case["name"] == f"{s['name']}_h40_task_only_s{case['key'][2]}")
        case.update(task=sample["task"], phase=sample["phase"])
    expected, full_pairs = preparation(cases)
    expected["duplicate_queries_removed"] = aliases
    compare = Comparison()
    compare.check(artifact["preparation"], expected, "preparation")
    rebuilt = dict(registered_fits_verified=0, eligible_for_further_validation=None)
    if not preparation_only:
        assert (
            artifact["unique_queries"] == 44
            and artifact["distinct_anchors"] == 22
            and artifact["common_h40_endpoints"] == 1398
        )
        assert artifact["pose_identity_bytes_exact"] is True
        rebuilt = verify_full(artifact, cases, config, expected, full_pairs, compare)
    sources.update({str(path): artifact_sha, str(Path(__file__).resolve()): sha(__file__)})
    for source, digest in sources.items():
        assert sha(source) == digest, source
    result = dict(
        schema_version="s2a.generated-reference-calibration-independent-verification.v1",
        complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        preparation_only=preparation_only,
        artifact_path=str(path),
        artifact_sha256=artifact_sha,
        input_sha256=sources,
        input_files_rechecked=len(sources),
        unique_queries=44,
        aliases_removed=4,
        whole_episode_folds=6,
        fit_episodes_per_fold=5,
        common_endpoint_counts=[352, 704, 1398],
        no_heldout_fit_or_donor_leakage=True,
        episode_query_endpoint_weights_verified=True,
        raw_opening_features_and_fit_labels_verified=True,
        full_candidate_same_oof_prefix_only=True,
        maximum_absolute_recompute_difference=compare.maximum,
        recompute_numeric_tolerance=TOLERANCE,
        promotion_comparison_tolerance=0,
        independent_solver="augmented weighted least squares with zero-residual ridge rows, not production normal equations",
        production_fit_apply_metric_decision_imported=False,
        **rebuilt,
        new_model_queries=0,
        new_physics_steps=0,
        physical_promotion_authorized=False,
        scope="Six already-viewed development episodes; reference intervention, no fresh acceptance, raw checkpoint certification or task success evidence.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                path=str(args.output.resolve()),
                sha256=sha(args.output),
                registered_fits_verified=rebuilt["registered_fits_verified"],
            )
        )
    )


if __name__ == "__main__":
    main()
