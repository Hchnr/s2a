"""023 cached-reference residual calibration: zero-fit preparation or fixed19 fits.

All labels belong to explicitly separated fit/scoring partitions. Deployment
features contain only measured anchor, generated references and endpoint time.
"""

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
from diagnose_reference_gains import load_cases, metrics, pooled, view
from diagnose_rotation_drift import endpoint_times, fold_plans, pairs_record
from summarize_wam_stability_screen import Evidence

from s2a.artifacts import REPO, sha256, write_json
from s2a.generated_reference_calibration import CalibrationConfig, GeneratedReferenceCalibration, reference_features

CONFIG = "configs/generated_reference_calibration_20261021.json"
PLAN = "plans/v0.0.1/023_generated_reference_calibration_20261021.md"
REGISTRATION = "fab0f78"
METHODS = ("raw", "identity", "paired", "hold", "constant_velocity", "no_wam", "shuffled")
FROZEN_HELPERS = {
    "scripts/diagnose_reference_gains.py": "09847245130ca36d4fce66e7bdbfa56a1b127a6b3570190d18f654e6efbe3bc6",
    "scripts/diagnose_rotation_drift.py": "bcc11c46a7852dde38924f20a80aa9eedef042341204d4442b205222673cb414",
}
SOURCES = (
    "scripts/diagnose_generated_reference_calibration.py",
    "src/s2a/generated_reference_calibration.py",
    "tests/test_generated_reference_calibration.py",
    "tests/test_generated_reference_calibration_diagnostic.py",
    "scripts/summarize_wam_earlier_screen.py",
    "scripts/summarize_wam_stability_screen.py",
    "src/s2a/reference_gains.py",
    "src/s2a/rotation_drift.py",
    "src/s2a/geometry.py",
    "src/s2a/checkpoint_audit.py",
    "plans/v0.0.1/008_reference_gain_diagnostic_20261021.md",
    "plans/v0.0.1/018_rotation_drift_diagnostic_20261021.md",
)


def algorithm_config(config):
    return CalibrationConfig(**{field.name: config[field.name] for field in fields(CalibrationConfig)})


def registered_config(path, evidence):
    if Path(path).resolve() != REPO / CONFIG:
        raise ValueError("requires the explicit registered023 configuration")
    for relative in (CONFIG, PLAN):
        expected = hashlib.sha256(
            subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
        ).hexdigest()
        evidence.check(REPO / relative, expected)
    config = evidence.json(path)
    if (
        config["checkpoint"] != "8400_g10"
        or config["registered_fit_count"] != 19
        or config["expected_unique_queries"] != 44
    ):
        raise ValueError("wrong registered model or fit/sample budget")
    if config["new_model_queries"] != 0 or config["new_physics_steps"] != 0 or config["physical_promotion_authorized"]:
        raise ValueError("023 has no model/physics/fresh promotion budget")
    return config


def case_metadata(cases, evidence):
    manifest = evidence.json(REPO / "outputs/wam_stability_20261021/earlier_screening/8400_g10/samples/manifest.json")
    for case in cases:
        sample = next(
            sample
            for sample in manifest["cases"]
            if case["name"] == f"{sample['name']}_h40_task_only_s{case['key'][2]}"
        )
        # These labels are for reporting only. They never enter a feature,
        # ridge solver, donor selection or deployment apply call.
        case.update(task=sample["task"], phase=sample["phase"])


def fit_arrays(pairs, method, config):
    """Materialize one fit partition and its fixed donor-balanced weights; no fit."""
    records, weights = pairs_record(pairs)
    algorithm = algorithm_config(config)
    blocks = {
        name: []
        for name in (
            "pose_features",
            "pose_residuals",
            "opening_features",
            "opening_residuals",
            "endpoint_times_seconds",
        )
    }
    inputs = []
    offset = 0
    for (case, donor), record in zip(pairs, records, strict=True):
        count = record["common_endpoints"]
        source, opening = view(case, donor, count, method)
        times = endpoint_times(count, config)
        generated = reference_features(case["anchor"], case["anchor_opening"], source, opening, times, config=algorithm)
        truth = reference_features(
            case["anchor"],
            case["anchor_opening"],
            case["truth"][:count],
            case["true_opening"][:count],
            times,
            config=algorithm,
        )
        blocks["pose_features"].append(generated["pose_features"])
        blocks["pose_residuals"].append(truth["normalized_local_pose"] - generated["normalized_local_pose"])
        blocks["opening_features"].append(generated["opening_features"])
        blocks["opening_residuals"].append(case["true_opening"][:count] - opening)
        blocks["endpoint_times_seconds"].append(times)
        inputs.append(
            dict(
                query=case["name"],
                donor=donor["name"],
                anchor=case["anchor"].copy(),
                anchor_opening=case["anchor_opening"],
                generated_world_poses=source.copy(),
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


def fit_model(pairs, method, config):
    arrays, records = fit_arrays(pairs, method, config)
    model = GeneratedReferenceCalibration.fit(
        arrays["pose_features"],
        arrays["pose_residuals"],
        arrays["opening_features"],
        arrays["opening_residuals"],
        arrays["endpoint_weights"],
        config=algorithm_config(config),
    )
    x, y, xg, yg, w = (
        arrays[name]
        for name in ("pose_features", "pose_residuals", "opening_features", "opening_residuals", "endpoint_weights")
    )
    return model, dict(
        method=method,
        model=asdict(model),
        fit_episodes=sorted({case["episode"] for case, _ in pairs}),
        fit_queries=records,
        training_endpoints=len(w),
        labels_are_fit_only=True,
        **arrays,
        pose_normal_matrix=x.T @ (w[:, None] * x) + config["ridge"] * np.eye(7),
        pose_right_hand_side=x.T @ (w[:, None] * y),
        opening_normal_matrix=xg.T @ (w[:, None] * xg) + config["ridge"] * np.eye(2),
        opening_right_hand_side=xg.T @ (w * yg),
    )


def shared_metrics(case, pose, opening, count):
    return metrics(case, pose, np.clip(opening, 0, 1), count)


def scores(outputs, prefixes):
    result = {}
    for prefix in prefixes:
        result[str(prefix)] = {}
        for method in METHODS:
            p, r, g = (
                np.concatenate([output["methods"][method][field][:prefix] for output in outputs])
                for field in ("position_errors_m", "rotation_errors_rad", "opening_errors")
            )
            assert len(p) == len(r) == len(g) and len(p)
            result[str(prefix)][method] = dict(
                endpoints=len(p),
                mean_squared_sum=float(np.mean((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2)),
            )
    return result


def score_fold(plan, config, counter):
    models, records = {}, {}
    for method in ("paired", "no_wam", "shuffled"):
        counter.append(dict(held_episode=plan["held_episode"], method=method))
        models[method], records[method] = fit_model(plan["fit_pairs"], method, config)
        assert plan["held_episode"] not in records[method]["fit_episodes"]
    outputs = []
    for case, donor in plan["test_pairs"]:
        count = min(case["count"], donor["count"])
        times = endpoint_times(count, config)
        own, opening = view(case, donor, count, "paired")
        identity, identity_opening, _ = GeneratedReferenceCalibration(config=algorithm_config(config)).apply(
            case["anchor"], case["anchor_opening"], own, opening, times
        )
        assert identity.tobytes() == own.tobytes() and identity.dtype == own.dtype
        np.testing.assert_array_equal(identity_opening, np.clip(opening, 0, 1))
        values = dict(
            raw=shared_metrics(case, own, opening, count),
            identity=shared_metrics(case, identity, identity_opening, count),
        )
        assert values["raw"] == values["identity"]
        corrected_references = {}
        for method, model in models.items():
            source, source_opening = view(case, donor, count, method)
            corrected, corrected_opening, diagnostics = model.apply(
                case["anchor"], case["anchor_opening"], source, source_opening, times
            )
            values[method] = shared_metrics(case, corrected, corrected_opening, count)
            corrected_references[method] = dict(
                anchor=case["anchor"],
                anchor_opening=case["anchor_opening"],
                input_world_poses=source.copy(),
                input_raw_openings=source_opening.copy(),
                corrected_world_poses=corrected,
                corrected_openings=corrected_opening,
                diagnostics=diagnostics,
            )
        values["hold"] = shared_metrics(
            case, np.repeat(case["anchor"][None], count, axis=0), np.full(count, case["anchor_opening"]), count
        )
        values["constant_velocity"] = shared_metrics(
            case, case["history"][:count], np.full(count, case["anchor_opening"]), count
        )
        assert set(values) == set(METHODS)
        outputs.append(
            dict(
                query=case["name"],
                key=case["key"],
                episode=case["episode"],
                task=case["task"],
                phase=case["phase"],
                shuffled_donor=donor["name"],
                shuffled_donor_episode=donor["episode"],
                valid_endpoints=count,
                original_valid_endpoints=case["count"],
                endpoint_times_seconds=times,
                methods=values,
                corrected_references=corrected_references,
            )
        )
    return dict(
        held_episode=plan["held_episode"],
        fit_episodes=plan["fit_episodes"],
        fit_pair_records=plan["fit_pair_records"],
        test_pair_records=plan["test_pair_records"],
        fitted_models=records,
        test_cases=outputs,
        metrics=pooled(outputs, config["prefixes"]),
        normalized_scores=scores(outputs, config["prefixes"]),
    )


def eligibility(pooled_metrics, pooled_scores, folds, config):
    """All registered component and endpoint-score gates, with exactly zero slack."""
    components, short_scores = [], []
    for prefix in config["prefixes"]:
        p = str(prefix)
        for metric in ("position_rmse_m", "rotation_rmse_rad", "opening_mae"):
            raw, paired = (pooled_metrics[p][method][metric] for method in ("raw", "paired"))
            components.append(dict(horizon=prefix, metric=metric, raw=raw, paired=paired, passed=bool(paired <= raw)))
        if prefix in (8, 16):
            raw, paired = (pooled_scores[p][method]["mean_squared_sum"] for method in ("raw", "paired"))
            controls = {
                method: pooled_scores[p][method]["mean_squared_sum"]
                for method in ("hold", "constant_velocity", "no_wam", "shuffled")
            }
            short_scores.append(
                dict(
                    horizon=prefix,
                    raw=raw,
                    paired=paired,
                    controls=controls,
                    at_least_ten_percent_raw_improvement=bool(
                        raw > 0 and paired <= (1 - config["minimum_short_score_relative_improvement"]) * raw
                    ),
                    strictly_better_than_all_controls=bool(all(paired < value for value in controls.values())),
                )
            )
    episode_checks = []
    for fold in folds:
        checks = [
            fold["normalized_scores"][str(prefix)]["paired"]["mean_squared_sum"]
            <= fold["normalized_scores"][str(prefix)]["raw"]["mean_squared_sum"]
            for prefix in (8, 16)
        ]
        episode_checks.append(
            dict(held_episode=fold["held_episode"], both_short_scores_nonregressing=bool(all(checks)))
        )
    passing = sum(item["both_short_scores_nonregressing"] for item in episode_checks)
    passed = (
        all(item["passed"] for item in components)
        and all(
            item["at_least_ten_percent_raw_improvement"] and item["strictly_better_than_all_controls"]
            for item in short_scores
        )
        and passing >= config["minimum_nonregressing_short_episode_folds"]
    )
    return dict(
        eligible_for_further_validation=bool(passed),
        all_prefix_component_checks=components,
        short_score_checks=short_scores,
        episode_checks=episode_checks,
        nonregressing_both_short_episode_folds=passing,
        required_episode_folds=5,
        added_comparison_tolerance=0,
        full_candidate_excluded=True,
        independent_acceptance_passed=False,
        physical_promotion_authorized=False,
    )


def prepare(cases, config):
    plans = fold_plans(cases, config["seed"])
    full_pairs = [pair for plan in plans for pair in plan["test_pairs"]]
    full_records, full_weights = pairs_record(full_pairs)
    assert len(full_pairs) == len({tuple(case["key"]) for case, _ in full_pairs}) == 44
    assert sum(record["common_endpoints"] for record in full_records) == 1398
    for record, (case, _) in zip(full_records, full_pairs, strict=True):
        record["prefix_origin_held_episode"] = case["episode"]
    folds = []
    for plan in plans:
        fit_inputs = {}
        for method in ("paired", "no_wam", "shuffled"):
            arrays, records = fit_arrays(plan["fit_pairs"], method, config)
            fit_inputs[method] = dict(fit_queries=records, **arrays)
        folds.append(
            dict(
                held_episode=plan["held_episode"],
                fit_episodes=plan["fit_episodes"],
                fit_pair_records=plan["fit_pair_records"],
                test_pair_records=plan["test_pair_records"],
                fit_inputs=fit_inputs,
            )
        )
    full_arrays, full_input_records = fit_arrays(full_pairs, "paired", config)
    for record, (case, _) in zip(full_input_records, full_pairs, strict=True):
        record["prefix_origin_held_episode"] = case["episode"]
    np.testing.assert_array_equal(full_weights, full_arrays["endpoint_weights"])
    result = dict(
        folds=folds,
        full_candidate_fit_queries=full_input_records,
        full_candidate_fit_inputs=full_arrays,
        full_candidate_training_endpoints=1398,
        expected_fit_count=19,
        eligible_for_further_validation=None,
        full_candidate_prefix_rule="same44 OOF008/018 test donor counts fixed without labels; all-six paired fit only",
    )
    return plans, full_pairs, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--output-root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument(
        "--fit",
        action="store_true",
        help="explicit fixed19 fit run; requires prior independent source/preparation review",
    )
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("run CPU-only in S2A with explicitly empty CUDA/LD")
    evidence = Evidence()
    config = registered_config(args.config, evidence)
    root = (REPO / (args.output_root or config["output_dir"])).resolve()
    if root.exists() or not root.is_relative_to(REPO / "outputs") or root == REPO / "outputs":
        raise ValueError("output must be a fresh repository directory; no overwrite or retry")
    for path, expected in FROZEN_HELPERS.items():
        evidence.check(REPO / path, expected)
    for relative in SOURCES:
        evidence.read(REPO / relative)
    counter, stage = [], "frozen8400-only cached loader"
    try:
        cases, aliases = load_cases("8400_g10", REPO / "outputs/wam_stability_20261021", evidence)
        assert len(cases) == 44 and len({case["episode"] for case in cases}) == 6
        case_metadata(cases, evidence)
        stage = "fixed folds and reference/label features; zero parameter fitting"
        plans, full_pairs, preparation = prepare(cases, config)
        preparation["duplicate_queries_removed"] = aliases
        if args.prepare_only:
            evidence.recheck()
            root.mkdir(parents=True, exist_ok=False)
            write_json(
                root / "preparation.json",
                dict(
                    schema_version="s2a.generated-reference-calibration-preparation.v1",
                    complete=True,
                    fit_count=0,
                    registered_commit=REGISTRATION,
                    config=config,
                    preparation=preparation,
                    input_sha256=evidence.hashes,
                    new_model_queries=0,
                    new_physics_steps=0,
                    physical_promotion_authorized=False,
                ),
            )
            print(
                json.dumps(
                    dict(path=str(root / "preparation.json"), sha256=sha256(root / "preparation.json"), fit_count=0)
                )
            )
            return
        stage = "eighteen fit-only LOEO model groups"
        folds = [score_fold(plan, config, counter) for plan in plans]
        outputs = [case for fold in folds for case in fold["test_cases"]]
        assert len(outputs) == 44 and sum(case["valid_endpoints"] for case in outputs) == 1398
        pooled_metrics, pooled_scores = pooled(outputs, config["prefixes"]), scores(outputs, config["prefixes"])
        assert [pooled_metrics[str(h)]["raw"]["endpoints"] for h in config["prefixes"]] == [352, 704, 1398]
        stage = "nineteenth all-six fit candidate, excluded from heldout scores"
        counter.append(dict(held_episode=None, method="paired", scope="training-only all-six candidate"))
        _, candidate = fit_model(full_pairs, "paired", config)
        candidate.update(
            scope="all-six training candidate; not heldout scored or deployed",
            fit_queries=preparation["full_candidate_fit_queries"],
            prefix_provenance=preparation["full_candidate_fit_queries"],
        )
        assert len(counter) == config["registered_fit_count"] and candidate["training_endpoints"] == 1398
        decision = eligibility(pooled_metrics, pooled_scores, folds, config)
        groups = {
            field: {
                str(value): dict(
                    metrics=pooled([case for case in outputs if case[field] == value], config["prefixes"]),
                    normalized_scores=scores([case for case in outputs if case[field] == value], config["prefixes"]),
                )
                for value in sorted({case[field] for case in outputs}, key=str)
            }
            for field in ("task", "episode", "phase")
        }
        evidence.recheck()
        root.mkdir(parents=True, exist_ok=False)
        output = root / "diagnostic.json"
        write_json(
            output,
            dict(
                schema_version="s2a.generated-reference-calibration-development-diagnostic.v1",
                complete=True,
                scope="Six already-viewed development episodes; reference intervention, not raw checkpoint or fresh acceptance",
                registered_commit=REGISTRATION,
                config=config,
                unique_queries=44,
                distinct_anchors=22,
                common_h40_endpoints=1398,
                fit_count=len(counter),
                fit_invocations=counter,
                input_opening_rule="unclipped raw opening enters fit/features; shared clip[0,1] at apply output and every metric boundary",
                pose_identity_bytes_exact=True,
                preparation=preparation,
                folds=folds,
                pooled=pooled_metrics,
                normalized_scores=pooled_scores,
                groups=groups,
                full_candidate=candidate,
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
                fit_invocations=counter,
                input_sha256=evidence.hashes,
                new_model_queries=0,
                new_physics_steps=0,
                source_verification_not_relaxed=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
