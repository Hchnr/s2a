"""Registered018: nineteen fit-only angular drift models on cached8400 states.

No model, physics, fresh fixture or task planner is loaded. All LOEO methods
share008's fixed donor prefixes. --prepare-only authenticates these prefixes
and weights without fitting any bias. Original loaders and outputs stay frozen.
"""

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from diagnose_reference_gains import load_cases, metrics, pooled, shuffled_pairs, view
from summarize_wam_stability_screen import Evidence

from s2a.artifacts import REPO, sha256, write_json
from s2a.rotation_drift import RotationDrift, episode_query_weights, world_rotation_residual

CONFIG = "configs/rotation_drift_diagnostic_20261021.json"
PLAN = "plans/v0.0.1/018_rotation_drift_diagnostic_20261021.md"
REGISTERED_COMMIT = "447865f"
METHODS = ("raw", "identity", "corrected", "hold", "constant_velocity", "no_wam", "shuffled")
SOURCES = (
    "scripts/diagnose_rotation_drift.py",
    "src/s2a/rotation_drift.py",
    "tests/test_rotation_drift.py",
    "tests/test_rotation_drift_diagnostic.py",
    "scripts/diagnose_reference_gains.py",
    "scripts/summarize_wam_earlier_screen.py",
    "scripts/summarize_wam_stability_screen.py",
    "src/s2a/reference_gains.py",
    "src/s2a/geometry.py",
    "src/s2a/checkpoint_audit.py",
    "plans/v0.0.1/008_reference_gain_diagnostic_20261021.md",
)


def endpoint_times(count, config):
    return np.arange(1, count + 1, dtype=float) / config["control_fps"]


def pairs_record(pairs, *, origin_held_episode=None):
    counts = [min(case["count"], donor["count"]) for case, donor in pairs]
    weights = episode_query_weights([case["episode"] for case, _ in pairs], counts)
    offset, records = 0, []
    for (case, donor), count in zip(pairs, counts, strict=True):
        values = weights[offset : offset + count]
        records.append(
            dict(
                query=case["name"],
                query_key=case["key"],
                episode=case["episode"],
                donor=donor["name"],
                donor_episode=donor["episode"],
                common_endpoints=count,
                weight_per_endpoint=float(values[0]),
                query_total_weight=float(values.sum()),
                original_query_endpoints=case["count"],
                donor_endpoints=donor["count"],
                prefix_origin_held_episode=origin_held_episode,
            )
        )
        offset += count
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12, rtol=0)
    return records, weights


def fold_plans(cases, seed):
    """Only whole-episode membership and generated-reference lengths are used."""
    plans = []
    for held in sorted({case["episode"] for case in cases}):
        fit = [case for case in cases if case["episode"] != held]
        test = [case for case in cases if case["episode"] == held]
        assert len({c["episode"] for c in fit}) == 5
        assert {c["episode"] for c in fit}.isdisjoint({c["episode"] for c in test})
        fit_pairs, test_pairs = shuffled_pairs(fit, test, seed)
        fit_records, _ = pairs_record(fit_pairs)
        test_records, _ = pairs_record(test_pairs, origin_held_episode=held)
        assert all(donor["episode"] != held for _, donor in test_pairs)
        plans.append(
            dict(
                held_episode=held,
                fit_episodes=sorted({c["episode"] for c in fit}),
                fit_pairs=fit_pairs,
                test_pairs=test_pairs,
                fit_pair_records=fit_records,
                test_pair_records=test_records,
            )
        )
    assert len(plans) == 6
    return plans


def fit_model(pairs, method, config):
    """Only the provided fit pairs' labels enter the closed-form solver."""
    records, weights = pairs_record(pairs)
    predicted, truth, times = [], [], []
    for (case, donor), record in zip(pairs, records, strict=True):
        count = record["common_endpoints"]
        source, _ = view(case, donor, count, method)
        predicted.append(source[:, :3, :3])
        truth.append(case["truth"][:count, :3, :3])
        times.append(endpoint_times(count, config))
    predicted, truth, times = np.concatenate(predicted), np.concatenate(truth), np.concatenate(times)
    model = RotationDrift.fit(
        predicted,
        truth,
        times,
        weights,
        ridge_seconds_squared=config["ridge_seconds_squared"],
        maximum_bias_norm_rad_per_second=config["maximum_bias_norm_rad_per_second"],
        maximum_horizon_seconds=config["maximum_horizon_seconds"],
    )
    residual = world_rotation_residual(predicted, truth)
    numerator = np.sum((weights * times)[:, None] * residual, axis=0)
    denominator = float(np.sum(weights * times**2) + config["ridge_seconds_squared"])
    return model, dict(
        method=method,
        model=asdict(model),
        fit_episodes=sorted({case["episode"] for case, _ in pairs}),
        fit_queries=records,
        training_endpoints=len(times),
        endpoint_weights=weights,
        endpoint_times_seconds=times,
        world_left_log_residuals_rad=residual,
        closed_form_numerator=numerator,
        closed_form_denominator_seconds_squared=denominator,
        unconstrained_bias_rad_per_second=numerator / denominator,
        norm_projection_active=bool(
            np.linalg.norm(numerator / denominator) > config["maximum_bias_norm_rad_per_second"]
        ),
        labels_are_fit_only=True,
    )


def score_fold(plan, config, fit_counter):
    models, fit_records = {}, {}
    for method in ("paired", "no_wam", "shuffled"):
        fit_counter.append(dict(held_episode=plan["held_episode"], method=method))
        models[method], fit_records[method] = fit_model(plan["fit_pairs"], method, config)
        assert plan["held_episode"] not in fit_records[method]["fit_episodes"]
    outputs = []
    for case, donor in plan["test_pairs"]:
        count = min(case["count"], donor["count"])
        times = endpoint_times(count, config)
        own, opening = view(case, donor, count, "paired")
        # Metric clipping matches008 and is outside the corrector. The module
        # itself preserves any input opening value, including out-of-range ones.
        opening = np.clip(opening, 0, 1)
        identity, identity_opening = RotationDrift().apply(own, opening, times)
        np.testing.assert_array_equal(identity, own)
        np.testing.assert_array_equal(identity_opening, opening)
        values = dict(raw=metrics(case, own, opening, count), identity=metrics(case, identity, identity_opening, count))
        assert values["identity"] == values["raw"]
        transformed_references = {}
        for method, model in models.items():
            source, source_opening = view(case, donor, count, method)
            source_opening = np.clip(source_opening, 0, 1)
            corrected, corrected_opening = model.apply(source, source_opening, times)
            np.testing.assert_array_equal(corrected[:, :3, 3], source[:, :3, 3])
            np.testing.assert_array_equal(corrected_opening, source_opening)
            name = "corrected" if method == "paired" else method
            values[name] = metrics(case, corrected, corrected_opening, count)
            transformed_references[name] = dict(
                input_rotations=source[:, :3, :3],
                corrected_rotations=corrected[:, :3, :3],
                position_exact=True,
                opening_exact=True,
            )
        values["hold"] = metrics(
            case, np.repeat(case["anchor"][None], count, axis=0), np.full(count, case["anchor_opening"]), count
        )
        values["constant_velocity"] = metrics(
            case, case["history"][:count], np.full(count, case["anchor_opening"]), count
        )
        assert set(values) == set(METHODS)
        outputs.append(
            dict(
                query=case["name"],
                key=case["key"],
                episode=case["episode"],
                shuffled_donor=donor["name"],
                shuffled_donor_episode=donor["episode"],
                valid_endpoints=count,
                original_valid_endpoints=case["count"],
                endpoint_times_seconds=times,
                methods=values,
                transformed_references=transformed_references,
            )
        )
    return dict(
        held_episode=plan["held_episode"],
        fit_episodes=plan["fit_episodes"],
        fit_pair_records=plan["fit_pair_records"],
        test_pair_records=plan["test_pair_records"],
        fitted_models=fit_records,
        test_cases=outputs,
        metrics=pooled(outputs, config["prefixes"]),
    )


def eligibility(pooled_metrics, folds, config):
    tolerance = config["comparison_tolerance_rad"]
    nonregression, short = {}, {}
    for prefix in config["prefixes"]:
        p = str(prefix)
        values = pooled_metrics[p]
        corrected = values["corrected"]["rotation_rmse_rad"]
        raw = values["raw"]["rotation_rmse_rad"]
        nonregression[p] = bool(corrected <= raw + tolerance)
        if prefix in (8, 16):
            controls = {
                name: values[name]["rotation_rmse_rad"] for name in ("hold", "constant_velocity", "no_wam", "shuffled")
            }
            nonregressing = sum(
                fold["metrics"][p]["corrected"]["rotation_rmse_rad"]
                <= fold["metrics"][p]["raw"]["rotation_rmse_rad"] + tolerance
                for fold in folds
            )
            short[p] = dict(
                at_least_ten_percent_raw_improvement=bool(
                    corrected <= (1 - config["minimum_short_raw_relative_improvement"]) * raw
                ),
                control_rotation_rmse_rad=controls,
                strictly_better_than_best_control_by_tolerance=bool(min(controls.values()) - corrected > tolerance),
                nonregressing_episode_folds=int(nonregressing),
                at_least_five_nonregressing_episode_folds=bool(
                    nonregressing >= config["minimum_nonregressing_short_episode_folds"]
                ),
            )
    passed = all(nonregression.values()) and all(
        item["at_least_ten_percent_raw_improvement"]
        and item["strictly_better_than_best_control_by_tolerance"]
        and item["at_least_five_nonregressing_episode_folds"]
        for item in short.values()
    )
    return dict(
        eligible_for_further_validation=bool(passed),
        all_prefixes_nonregression=nonregression,
        short_prefix_criteria=short,
        comparison_tolerance_rad=tolerance,
        full_candidate_excluded=True,
        independent_acceptance_passed=False,
        physical_promotion_authorized=False,
    )


def registered_config(path, evidence):
    config = evidence.json(path)
    for relative in (CONFIG, PLAN):
        registered = subprocess.check_output(["git", "show", f"{REGISTERED_COMMIT}:{relative}"], cwd=REPO)
        evidence.check(REPO / relative, hashlib.sha256(registered).hexdigest())
    assert Path(path).resolve() == REPO / CONFIG
    assert config == dict(
        seed=42,
        output_dir="outputs/wam_stability_20261021/rotation_drift_diagnostic",
        checkpoint="8400_g10",
        ridge_seconds_squared=0.1,
        maximum_bias_norm_rad_per_second=0.25,
        control_fps=20,
        maximum_horizon_seconds=2.0,
        prefixes=[8, 16, 40],
        split="leave_one_complete_episode_out",
        deduplicate=["episode", "anchor_frame", "sampling_seed"],
        expected_unique_queries=44,
        expected_common_h40_endpoints=1398,
        registered_fit_count=19,
        minimum_short_raw_relative_improvement=0.1,
        minimum_nonregressing_short_episode_folds=5,
        comparison_tolerance_rad=1e-10,
        new_model_queries=0,
        new_physics_steps=0,
        physical_promotion_authorized=False,
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--output-root", help="fresh artifact directory; never changes the registered fit protocol")
    parser.add_argument(
        "--prepare-only", action="store_true", help="authenticate8400 and record fixed pairings/weights, zero fits"
    )
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("run CPU-only from S2A with empty CUDA/LD")
    evidence = Evidence()
    config = registered_config(args.config, evidence)
    root = (REPO / (args.output_root or config["output_dir"])).resolve()
    if not root.is_relative_to(REPO / "outputs") or root == REPO / "outputs" or root.exists():
        raise ValueError("requires a fresh repository output directory; no overwrite or automatic retry")
    for name in SOURCES:
        evidence.read(REPO / name)
    fit_counter, stage = [], "8400-only loader"
    try:
        cases, aliases = load_cases("8400_g10", REPO / "outputs/wam_stability_20261021", evidence)
        assert len(cases) == config["expected_unique_queries"] and len({c["episode"] for c in cases}) == 6
        stage = "fixed pairings and fit-only weights; no bias fits yet"
        plans = fold_plans(cases, config["seed"])
        full_pairs = [pair for plan in plans for pair in plan["test_pairs"]]
        full_records, full_weights = pairs_record(full_pairs)
        assert len(full_pairs) == 44 and len({tuple(c["key"]) for c, _ in full_pairs}) == 44
        assert sum(item["common_endpoints"] for item in full_records) == config["expected_common_h40_endpoints"]
        for item, (case, _) in zip(full_records, full_pairs, strict=True):
            item["prefix_origin_held_episode"] = case["episode"]
        preparation = dict(
            folds=[{k: v for k, v in plan.items() if k not in ("fit_pairs", "test_pairs")} for plan in plans],
            full_candidate_fit_queries=full_records,
            full_candidate_endpoint_weights=full_weights,
            full_candidate_training_endpoints=1398,
            full_candidate_prefix_rule="same44 out-of-fold008 test donor counts, chosen without labels; all-six paired fit only",
            duplicate_queries_removed=aliases,
            expected_fit_count=19,
            eligible_for_further_validation=None,
        )
        if args.prepare_only:
            evidence.recheck()
            root.mkdir(parents=True, exist_ok=False)
            write_json(
                root / "preparation.json",
                dict(
                    complete=True,
                    fit_count=0,
                    preparation=preparation,
                    input_sha256=evidence.hashes,
                    new_model_queries=0,
                    new_physics_steps=0,
                ),
            )
            print(
                json.dumps(
                    dict(path=str(root / "preparation.json"), sha256=sha256(root / "preparation.json"), fit_count=0)
                )
            )
            return
        stage = "eighteen LOEO fit-only models"
        folds = [score_fold(plan, config, fit_counter) for plan in plans]
        outputs = [case for fold in folds for case in fold["test_cases"]]
        assert len(outputs) == 44 and sum(c["valid_endpoints"] for c in outputs) == 1398
        pooled_metrics = pooled(outputs, config["prefixes"])
        stage = "nineteenth all-six candidate fit; not heldout scoring"
        fit_counter.append(dict(held_episode=None, method="paired", scope="all-six training candidate only"))
        _, full_fit = fit_model(full_pairs, "paired", config)
        assert full_fit["training_endpoints"] == 1398 and len(fit_counter) == config["registered_fit_count"]
        full_fit.update(
            scope="all-six training candidate, not deployed, not scored as heldout",
            fit_queries=full_records,
            prefix_provenance=full_records,
        )
        decision = eligibility(pooled_metrics, folds, config)
        evidence.recheck()
        root.mkdir(parents=True, exist_ok=False)
        output = root / "diagnostic.json"
        write_json(
            output,
            dict(
                schema_version="s2a.rotation-drift-development-diagnostic.v1",
                complete=True,
                scope="Only six already-viewed development episodes; no independent acceptance, pretraining-unseen claim or original-checkpoint certification",
                registered_commit=REGISTERED_COMMIT,
                config=config,
                fit_count=len(fit_counter),
                fit_invocations=fit_counter,
                unique_queries=44,
                distinct_anchors=22,
                common_h40_endpoints=1398,
                opening_metric_rule="clip original/view opening once before the corrector and scoring, identically for raw/identity/corrected; module never clips or changes input opening",
                world_position_preserved=True,
                input_opening_preserved=True,
                identity_exact=True,
                preparation=preparation,
                folds=folds,
                pooled=pooled_metrics,
                full_candidate=full_fit,
                decision=decision,
                eligible_for_further_validation=decision["eligible_for_further_validation"],
                input_sha256=evidence.hashes,
                new_model_queries=0,
                new_physics_steps=0,
                physical_promotion_authorized=False,
            ),
        )
        print(
            json.dumps(
                dict(
                    path=str(output),
                    sha256=sha256(output),
                    fit_count=19,
                    eligible_for_further_validation=decision["eligible_for_further_validation"],
                )
            )
        )
    except Exception as error:
        if not root.exists():
            root.mkdir(parents=True, exist_ok=False)
            write_json(
                root / "execution_rejected.json",
                dict(
                    complete=False,
                    stage=stage,
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    fit_invocations=fit_counter,
                    input_sha256=evidence.hashes,
                    source_verification_not_relaxed=True,
                    new_model_queries=0,
                    new_physics_steps=0,
                ),
            )
        raise


if __name__ == "__main__":
    main()
