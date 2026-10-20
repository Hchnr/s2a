"""Independently verify018's nineteen cached angular fits and heldout metrics.

Direct matrix integration and shuffle reconstruction come from the independent
008 auditor. This file imports neither the018 production driver/model nor its
weight, fit, correction, pooling or eligibility functions. No new hypothesis,
model request, physics step or fresh performance is evaluated.
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
from audit_reference_gains import load_cases, metrics, pairs, view
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = "6247a296793b35ef1c998ee0a72677657dbd11d834e083f1891f8cd17dbdbf61"
METHODS = ("raw", "identity", "corrected", "hold", "constant_velocity", "no_wam", "shuffled")
FIT_METHODS = ("paired", "no_wam", "shuffled")
TOLERANCE = 2e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
                query_total_weight=count * weight,
                original_query_endpoints=case["count"],
                donor_endpoints=donor["count"],
                prefix_origin_held_episode=origin,
            )
        )
    np.testing.assert_allclose(np.sum(weights), 1, atol=1e-12, rtol=0)
    for episode in groups:
        np.testing.assert_allclose(
            sum(r["query_total_weight"] for r in records if r["episode"] == episode),
            1 / len(groups),
            atol=1e-12,
            rtol=0,
        )
    return records, np.asarray(weights)


def augmented_solution(residuals, times, weights):
    """Three-column augmented LS, distinct from production's scalar formula."""
    residuals, times, weights = map(np.asarray, (residuals, times, weights))
    assert residuals.shape == (len(times), 3) and weights.shape == times.shape
    assert np.all(weights > 0) and np.all((times > 0) & (times <= 2))
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12, rtol=0)
    design = ((np.sqrt(weights) * times)[:, None, None] * np.eye(3)).reshape(-1, 3)
    design = np.vstack((design, np.sqrt(0.1) * np.eye(3)))
    targets = np.r_[(np.sqrt(weights)[:, None] * residuals).ravel(), np.zeros(3)]
    unconstrained = np.linalg.lstsq(design, targets, rcond=None)[0]
    bias = unconstrained.copy()
    norm = np.linalg.norm(bias)
    if norm > 0.25:
        bias *= 0.25 / norm
    return bias, unconstrained


def reconstruct_fit(paired, method):
    records, weights = records_and_weights(paired)
    residuals, times = [], []
    for (case, donor), record in zip(paired, records, strict=True):
        count = record["common_endpoints"]
        source, _ = view(case, donor, count, method)
        for index in range(count):
            error = source[index, :3, :3] @ case["truth"][index, :3, :3].T
            residual = Rotation.from_matrix(error).as_rotvec()
            assert np.linalg.norm(residual) < np.pi - 1e-6
            residuals.append(residual)
            times.append((index + 1) / 20)
    residuals, times = np.asarray(residuals), np.asarray(times)
    bias, unconstrained = augmented_solution(residuals, times, weights)
    return bias, dict(
        method=method,
        model=dict(
            bias_rad_per_second=bias.tolist(), maximum_bias_norm_rad_per_second=0.25, maximum_horizon_seconds=2.0
        ),
        fit_episodes=sorted({case["episode"] for case, _ in paired}),
        fit_queries=records,
        training_endpoints=len(times),
        endpoint_weights=weights,
        endpoint_times_seconds=times,
        world_left_log_residuals_rad=residuals,
        closed_form_numerator=np.einsum("i,ij->j", weights * times, residuals),
        closed_form_denominator_seconds_squared=float(np.dot(weights, times**2) + 0.1),
        unconstrained_bias_rad_per_second=unconstrained,
        norm_projection_active=bool(np.linalg.norm(unconstrained) > 0.25),
        labels_are_fit_only=True,
    )


def corrected_reference(source, opening, bias):
    output = source.copy()
    for index in range(len(source)):
        correction = Rotation.from_rotvec(-(index + 1) / 20 * bias).as_matrix()
        output[index, :3, :3] = correction @ source[index, :3, :3]
    np.testing.assert_array_equal(output[:, :3, 3], source[:, :3, 3])
    return output, opening.copy()


def pool(values):
    result = {}
    for prefix in (8, 16, 40):
        result[str(prefix)] = {}
        for method in METHODS:
            vectors = [
                np.concatenate([case[method][field][:prefix] for case in values])
                for field in ("position_errors_m", "rotation_errors_rad", "opening_errors")
            ]
            position, rotation, opening = vectors
            result[str(prefix)][method] = dict(
                endpoints=len(position),
                position_rmse_m=float(np.linalg.norm(position) / np.sqrt(len(position))),
                rotation_rmse_rad=float(np.linalg.norm(rotation) / np.sqrt(len(rotation))),
                opening_mae=float(np.sum(np.abs(opening)) / len(opening)),
            )
    return result


def decision(metrics_by_prefix, folds):
    nonregression, short = {}, {}
    for prefix in (8, 16, 40):
        key = str(prefix)
        values = metrics_by_prefix[key]
        raw, corrected = (values[method]["rotation_rmse_rad"] for method in ("raw", "corrected"))
        nonregression[key] = bool(corrected - raw <= 1e-10)
        if prefix in (8, 16):
            controls = {m: values[m]["rotation_rmse_rad"] for m in ("hold", "constant_velocity", "no_wam", "shuffled")}
            count = sum(
                f[key]["corrected"]["rotation_rmse_rad"] - f[key]["raw"]["rotation_rmse_rad"] <= 1e-10 for f in folds
            )
            short[key] = dict(
                at_least_ten_percent_raw_improvement=bool(corrected <= 0.9 * raw),
                control_rotation_rmse_rad=controls,
                strictly_better_than_best_control_by_tolerance=bool(corrected + 1e-10 < min(controls.values())),
                nonregressing_episode_folds=int(count),
                at_least_five_nonregressing_episode_folds=bool(count >= 5),
            )
    eligible = all(nonregression.values()) and all(
        value["at_least_ten_percent_raw_improvement"]
        and value["strictly_better_than_best_control_by_tolerance"]
        and value["at_least_five_nonregressing_episode_folds"]
        for value in short.values()
    )
    return dict(
        eligible_for_further_validation=eligible,
        all_prefixes_nonregression=nonregression,
        short_prefix_criteria=short,
        comparison_tolerance_rad=1e-10,
        full_candidate_excluded=True,
        independent_acceptance_passed=False,
        physical_promotion_authorized=False,
    )


class Comparison:
    def __init__(self):
        self.maximum = {}

    def check(self, saved, expected, kind):
        if isinstance(expected, dict):
            assert isinstance(saved, dict) and set(saved) == set(expected), kind
            for key in expected:
                self.check(saved[key], expected[key], kind)
        elif isinstance(expected, (str, bool)) or expected is None:
            assert saved == expected and type(saved) is type(expected), (kind, saved, expected)
        elif isinstance(expected, list) and (not expected or isinstance(expected[0], (dict, str))):
            assert len(saved) == len(expected), kind
            for actual, desired in zip(saved, expected, strict=True):
                self.check(actual, desired, kind)
        else:
            actual, desired = np.asarray(saved), np.asarray(expected)
            assert actual.shape == desired.shape, kind
            np.testing.assert_allclose(actual, desired, atol=TOLERANCE, rtol=0, err_msg=kind)
            maximum = float(np.max(np.abs(actual - desired))) if actual.size else 0.0
            self.maximum[kind] = max(self.maximum.get(kind, 0.0), maximum)


def cache_inputs():
    config_path = ROOT / "configs/wam_stability_screen_8400_g10_20261021.json"
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_dir"]
    manifest = root / "samples/manifest.json"
    files = [config_path, manifest]
    for sample in json.loads(manifest.read_text())["cases"]:
        files.append(Path(sample["arrays_path"]))
        for seed in (42, 43):
            name = f"{sample['name']}_h40_task_only_s{seed}"
            files.append(root / "state_queries" / config["checkpoints"][0]["name"] / name / "response.json")
    return {str(path.resolve()): sha(path) for path in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or any(
        os.environ.get(name) != "" for name in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
    ):
        raise ValueError("run CPU-only from S2A with empty CUDA and LD")
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("requires a new repository output artifact")
    artifact_sha = sha(args.artifact)
    artifact = json.loads(args.artifact.read_text())
    assert artifact["schema_version"] == "s2a.rotation-drift-development-diagnostic.v1"
    assert artifact["registered_commit"] == "447865f"
    assert artifact["config"] == json.loads((ROOT / "configs/rotation_drift_diagnostic_20261021.json").read_text())
    assert artifact["complete"] and artifact["fit_count"] == 19
    assert (
        artifact["unique_queries"] == 44
        and artifact["distinct_anchors"] == 22
        and artifact["common_h40_endpoints"] == 1398
    )
    assert artifact["world_position_preserved"] and artifact["input_opening_preserved"] and artifact["identity_exact"]
    assert (
        artifact["new_model_queries"] == artifact["new_physics_steps"] == 0
        and not artifact["physical_promotion_authorized"]
    )
    sources = dict(artifact["input_sha256"])
    for path, digest in sources.items():
        assert sha(path) == digest, path
    for relative, commit in (
        ("scripts/diagnose_rotation_drift.py", "4fe738b"),
        ("src/s2a/rotation_drift.py", "c037e77"),
        ("plans/v0.0.1/018_rotation_drift_diagnostic_20261021.md", "447865f"),
        ("configs/rotation_drift_diagnostic_20261021.json", "447865f"),
    ):
        original = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(original).hexdigest()
        assert sources[str(ROOT / relative)] == sha(ROOT / relative) == digest
    assert sha(ROOT / "scripts/audit_reference_gains.py") == HELPER_SHA
    independent_inputs = cache_inputs()
    assert all(sources[path] == digest for path, digest in independent_inputs.items())
    cases, aliases = load_cases("8400_g10")
    assert aliases == artifact["preparation"]["duplicate_queries_removed"]
    assert artifact["preparation"]["expected_fit_count"] == 19
    assert artifact["preparation"]["eligible_for_further_validation"] is None
    assert artifact["preparation"]["full_candidate_training_endpoints"] == 1398
    compare, fold_results, all_methods, full_pairs, tested = Comparison(), [], [], [], []
    invocation_records, fit_results = [], []
    for held, saved_fold, preparation in zip(
        sorted({case["episode"] for case in cases}), artifact["folds"], artifact["preparation"]["folds"], strict=True
    ):
        fit = [case for case in cases if case["episode"] != held]
        test = [case for case in cases if case["episode"] == held]
        fit_pairs, test_pairs = pairs(fit, test)
        fit_records, _ = records_and_weights(fit_pairs)
        test_records, _ = records_and_weights(test_pairs, held)
        fit_episodes = sorted({case["episode"] for case in fit})
        assert len(fit_episodes) == 5 and held not in fit_episodes
        assert all(donor["episode"] in fit_episodes for _, donor in fit_pairs + test_pairs)
        expected_preparation = dict(
            held_episode=held, fit_episodes=fit_episodes, fit_pair_records=fit_records, test_pair_records=test_records
        )
        compare.check(preparation, expected_preparation, "preparation")
        for key, value in expected_preparation.items():
            compare.check(saved_fold[key], value, "fold_metadata")
        biases = {}
        assert set(saved_fold["fitted_models"]) == set(FIT_METHODS)
        for method in FIT_METHODS:
            bias, expected_fit = reconstruct_fit(fit_pairs, method)
            compare.check(saved_fold["fitted_models"][method], expected_fit, "fit")
            biases[method] = bias
            invocation_records.append(dict(held_episode=held, method=method))
            fit_results.append(
                dict(
                    held_episode=held,
                    method=method,
                    bias_rad_per_second=bias.tolist(),
                    endpoints=expected_fit["training_endpoints"],
                )
            )
        fold_methods = []
        for (case, donor), saved in zip(test_pairs, saved_fold["test_cases"], strict=True):
            count = min(case["count"], donor["count"])
            own, opening = view(case, donor, count, "paired")
            opening = np.clip(opening, 0, 1)
            values = dict(raw=metrics(case, own, opening, count), identity=metrics(case, own, opening, count))
            transforms = {}
            for method, bias in biases.items():
                source, source_opening = view(case, donor, count, method)
                clipped = np.clip(source_opening, 0, 1)
                output, output_opening = corrected_reference(source, clipped, bias)
                np.testing.assert_array_equal(output_opening, clipped)
                name = "corrected" if method == "paired" else method
                values[name] = metrics(case, output, output_opening, count)
                transforms[name] = dict(
                    input_rotations=source[:, :3, :3],
                    corrected_rotations=output[:, :3, :3],
                    position_exact=True,
                    opening_exact=True,
                )
            values["hold"] = metrics(
                case, np.repeat(case["anchor"][None], count, axis=0), np.full(count, case["anchor_opening"]), count
            )
            values["constant_velocity"] = metrics(
                case, case["history"][:count], np.full(count, case["anchor_opening"]), count
            )
            expected = dict(
                query=case["name"],
                key=list(case["key"]),
                episode=held,
                shuffled_donor=donor["name"],
                shuffled_donor_episode=donor["episode"],
                valid_endpoints=count,
                original_valid_endpoints=case["count"],
                endpoint_times_seconds=np.arange(1, count + 1) / 20,
                methods=values,
                transformed_references=transforms,
            )
            compare.check(saved, expected, "heldout_case")
            for field in ("position_errors_m", "opening_errors"):
                assert (
                    saved["methods"]["raw"][field]
                    == saved["methods"]["identity"][field]
                    == saved["methods"]["corrected"][field]
                )
            fold_methods.append(values)
            tested.append(case["key"])
        aggregate = pool(fold_methods)
        compare.check(saved_fold["metrics"], aggregate, "fold_aggregate")
        fold_results.append(aggregate)
        all_methods.extend(fold_methods)
        full_pairs.extend(test_pairs)
    assert len(tested) == len(set(tested)) == len(cases) == 44
    assert set(tested) == {case["key"] for case in cases}
    pooled = pool(all_methods)
    assert [pooled[str(p)]["raw"]["endpoints"] for p in (8, 16, 40)] == [352, 704, 1398]
    compare.check(artifact["pooled"], pooled, "pooled")
    expected_decision = decision(pooled, fold_results)
    compare.check(artifact["decision"], expected_decision, "eligibility")
    assert artifact["eligible_for_further_validation"] == expected_decision["eligible_for_further_validation"]
    full_records, full_weights = records_and_weights(full_pairs)
    for record in full_records:
        record["prefix_origin_held_episode"] = record["episode"]
    assert sum(record["common_endpoints"] for record in full_records) == 1398
    compare.check(artifact["preparation"]["full_candidate_fit_queries"], full_records, "full_prefix_provenance")
    compare.check(artifact["preparation"]["full_candidate_endpoint_weights"], full_weights, "full_weights")
    bias, full_fit = reconstruct_fit(full_pairs, "paired")
    full_fit.update(
        scope="all-six training candidate, not deployed, not scored as heldout",
        fit_queries=full_records,
        prefix_provenance=full_records,
    )
    compare.check(artifact["full_candidate"], full_fit, "full_fit")
    invocation_records.append(dict(held_episode=None, method="paired", scope="all-six training candidate only"))
    assert invocation_records == artifact["fit_invocations"] and len(invocation_records) == 19
    sources.update(independent_inputs)
    sources.update(
        {
            str(args.artifact.resolve()): artifact_sha,
            str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / "scripts/audit_reference_gains.py"): HELPER_SHA,
        }
    )
    for path, digest in sources.items():
        assert sha(path) == digest, path
    result = dict(
        schema_version="s2a.rotation-drift-independent-verification.v1",
        complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        artifact_path=str(args.artifact.resolve()),
        artifact_sha256=artifact_sha,
        input_sha256=sources,
        input_files_rechecked=len(sources),
        unique_queries=44,
        distinct_anchors=22,
        aliases_removed=4,
        whole_episode_folds=6,
        fit_episodes_per_fold=5,
        heldout_queries_rebuilt=44,
        paired_no_wam_shuffled_fits_rebuilt=18,
        full_candidate_rebuilt=1,
        registered_fits_verified=19,
        endpoint_counts=[352, 704, 1398],
        no_heldout_fit_or_donor_leakage=True,
        episode_query_endpoint_weights_verified=True,
        full_candidate_same_oof_prefix_only=True,
        fit_solver="Independent three-column augmented least squares with zero-target ridge rows, then isotropic norm projection; log-vector surrogate only",
        fit_parameters=fit_results,
        full_candidate_bias_rad_per_second=bias.tolist(),
        maximum_absolute_recompute_difference=compare.maximum,
        pooled=pooled,
        decision=expected_decision,
        eligible_for_further_validation=expected_decision["eligible_for_further_validation"],
        production_model_driver_geometry_pooling_imported=False,
        new_hypotheses_or_searches=0,
        new_model_queries=0,
        new_physics_steps=0,
        physical_promotion_authorized=False,
        scope="Six already-viewed development episodes. No fresh acceptance, original checkpoint certification or closed-loop success evidence.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha(args.output),
                complete=True,
                eligible_for_further_validation=result["eligible_for_further_validation"],
                maximum_absolute_difference=compare.maximum,
            )
        )
    )


if __name__ == "__main__":
    main()
