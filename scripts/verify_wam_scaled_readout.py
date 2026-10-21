"""027 independent CPU checks of train-only scaling and one fixed ridge fit.

Share only frozen026 independent cache/evidence and metric helpers. No producer
fit/apply/gate, Torch, Cosmos, dataloader, raw EP or PNG reader is imported.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

CONFIG = REPO / "configs/wam_train_scale_readout_20261021.json"
CONFIG_SHA = "dd0e0124d256a953af48ea2e548cca6603cd9f777d6bd5e4f620548591a2c06b"
PLAN = REPO / "plans/v0.0.1/027_train_scale_readout_20261021.md"
PLAN_SHA = "586ddf2656d39752efd7979e1877af940004e14b058d900c7ef464661cbf9fb1"
HELPER = REPO / "scripts/verify_wam_flow_readout.py"
HELPER_SHA = "f72ed4567da138a3100c95344f1b8cb7b535493fdca37888f8b0e7d807bdcd29"
NEW_SOURCES = (
    "src/s2a/wam_scaled_readout.py",
    "tests/test_wam_scaled_readout.py",
    "scripts/fit_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_fit.py",
    "scripts/prepare_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_preparation.py",
    "scripts/verify_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_verification.py",
    "scripts/run_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_runner.py",
)


def frozen_helpers():
    if sha256(HELPER) != HELPER_SHA:
        raise ValueError("frozen026 independent helper changed")
    spec = importlib.util.spec_from_file_location("frozen026_scaled_audit_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLOW = frozen_helpers()
OLD = FLOW.OLD
require, numeric, exact, close = FLOW.require, FLOW.numeric, FLOW.exact, FLOW.close


def row_weights(weights, count):
    weights = numeric(weights, "train weights").astype(np.float64)
    require(
        weights.shape == (count,) and np.all(weights > 0) and abs(weights.sum() - 1) <= 1e-12,
        "train weights must be positive and sum to one",
    )
    return weights


def preprocessing(hidden, weights):
    hidden = numeric(hidden, "train hidden").astype(np.float64)
    require(hidden.ndim == 2 and min(hidden.shape) > 0, "train hidden must be a nonempty matrix")
    weights = row_weights(weights, len(hidden))
    center = np.sum(weights[:, None] * hidden, axis=0)
    scale = np.sqrt(np.sum(weights * np.sum((hidden - center) ** 2, axis=1)))
    require(
        np.isfinite(center).all() and np.isfinite(scale) and scale > 0, "train global scale must be finite and positive"
    )
    return center, np.asarray(scale, np.float64)


def scaled_design(hidden, sigma_actual, center, scale):
    hidden, center, scale = (
        numeric(value, name).astype(np.float64)
        for value, name in ((hidden, "hidden"), (center, "center"), (scale, "global scale"))
    )
    require(
        hidden.ndim == 2
        and min(hidden.shape) > 0
        and center.shape == (hidden.shape[1],)
        and scale.shape == ()
        and scale > 0,
        "scaled design requires matching center and one positive global scalar",
    )
    phi = np.column_stack(((hidden - center) / scale, np.ones(len(hidden))))
    return -phi / FLOW.sigmas64(sigma_actual, len(hidden))[:, None]


def augmented_fit(hidden, sigma_actual, target, weights):
    center, scale = preprocessing(hidden, weights)
    design = scaled_design(hidden, sigma_actual, center, scale)
    target, weights = (
        numeric(target, "original velocity residual").astype(np.float64),
        row_weights(weights, len(design)),
    )
    require(target.shape == (len(design), 10), "scaled fit requires original active10 velocity targets")
    x = np.concatenate((np.sqrt(weights)[:, None] * design, np.sqrt(0.1) * np.eye(design.shape[1])))
    y = np.concatenate((np.sqrt(weights)[:, None] * target, np.zeros((design.shape[1], 10))))
    return dict(weights=np.linalg.lstsq(x, y, rcond=None)[0], center=center, scale=scale)


def training_rows(rows, train_episodes, validation_episodes):
    require(not set(train_episodes) & set(validation_episodes), "027 whole-episode train/validation overlap")
    for row in rows:
        require(
            (row["split"] == "train" and row["episode"] in train_episodes)
            or (row["split"] == "validation" and row["episode"] in validation_episodes),
            "027 teacher row entered the wrong whole-episode split",
        )
    selected = np.asarray([r["split"] == "train" for r in rows])
    return selected, OLD.hierarchical_weights([r for r in rows if r["split"] == "train"])


def quantized_velocity(hidden, base, sigma_actual, coefficients, center, scale):
    base, coefficients = numeric(base, "base"), numeric(coefficients, "coefficients")
    design = scaled_design(hidden, sigma_actual, center, scale)
    require(
        base.shape == (len(design), 128) and coefficients.shape == (design.shape[1], 10),
        "scaled deployment tensor shapes differ",
    )
    if not np.any(coefficients):
        return base
    result = base.copy()
    result[:, :10] = OLD.quantize_bfloat16(base[:, :10].astype(np.float64) + design @ coefficients)
    return result


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def forbidden_observations(prepared):
    paths = {str(Path(a[k]).resolve()) for a in prepared["anchors"] for k in ("arrays_path", "image_path")}
    for case in prepared["zero_identity_cases"]:
        paths.update(str(Path(case["sample"][k]).resolve()) for k in ("arrays_path", "image_path"))
    return paths


def cache_only_hashes(mapping, forbidden, evidence):
    require(mapping, "027 empty source/evidence map")
    for filename, digest in mapping.items():
        source = Path(filename).resolve()
        require(
            str(source) not in forbidden and source.suffix.lower() != ".png",
            "027 preparation reopened inherited EP/PNG",
        )
        evidence.check(source, digest)


def source_closure(prepared, flow, path):
    expected = (
        set(flow["source_sha256"]) | {str(REPO / filename) for filename in NEW_SOURCES} | {str(CONFIG), str(PLAN)}
    )
    require(
        len(expected) == 84 and set(prepared["source_sha256"]) == expected,
        "027 required84 source closure is incomplete or changed",
    )
    require(
        all(prepared["source_sha256"].get(k) == v for k, v in flow["source_sha256"].items())
        and len(flow["source_sha256"]) == 72,
        "027 did not preserve all72 original026 sources",
    )
    require(
        prepared["source_sha256"][str(CONFIG)] == CONFIG_SHA
        and prepared["source_sha256"][str(PLAN)] == PLAN_SHA
        and prepared["source_sha256"][str(HELPER)] == HELPER_SHA,
        "027 source closure changed registration or independent helper",
    )
    # write_once sorts JSON object keys after snapshots are numbered. Rebuild
    # the registered construction order, not the new JSON map's sorted order.
    ordered = list(flow["source_sha256"]) + [str(REPO / p) for p in NEW_SOURCES] + [str(CONFIG), str(PLAN)]
    snapshots = {
        str(Path(path).resolve().parent / "source_snapshot" / f"{index:03d}_{Path(filename).name}"): prepared[
            "source_sha256"
        ][filename]
        for index, filename in enumerate(ordered)
    }
    require(
        prepared["snapshot_sha256"] == snapshots, "027 required84 snapshot paths/digests do not match their sources"
    )


def verify_preparation(path, evidence):
    prepared = evidence.json(path)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    evidence.check(HELPER, HELPER_SHA)
    require(
        prepared["schema_version"] == "s2a.scaled-readout-preparation.v1"
        and prepared["config"] == config
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration"] == reference(PLAN),
        "027 preparation/config/registration differs",
    )
    require(
        prepared["output_root"] == str(REPO / config["output_dir"]) and prepared["budget"] == config["budget"],
        "027 preparation budget/output differs",
    )
    forbidden = forbidden_observations(prepared)
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        cache_only_hashes(prepared[key], forbidden, evidence)
    ancestors = {}
    for suffix in ("prepared", "preparation_verification", "fit", "fit_verification", "final_peer"):
        expected = dict(path=str(REPO / config[f"source_flow_{suffix}"]), sha256=config[f"source_flow_{suffix}_sha256"])
        require(prepared[f"flow_{suffix}_ref"] == expected, "027 frozen026 ancestor reference changed")
        ancestors[suffix] = evidence.ref(expected)
    for ancestor in ancestors.values():
        if "input_sha256" in ancestor:
            cache_only_hashes(ancestor["input_sha256"], forbidden, evidence)
    for key in ("source_sha256", "snapshot_sha256"):
        cache_only_hashes(ancestors["prepared"][key], forbidden, evidence)
    flow, features = FLOW.verify_preparation(prepared["flow_prepared_ref"]["path"], evidence)
    require(flow == ancestors["prepared"], "027 original026 preparation changed")
    source_closure(prepared, flow, path)
    require(
        all(
            prepared[k] == 0
            for k in (
                "new_model_calls",
                "new_teacher_calls",
                "new_episode_reads",
                "new_png_reads",
                "fit_count",
                "physics_steps",
                "fresh_performance_cases",
            )
        )
        and prepared["full_weight_bytes_rehashed"] is False,
        "027 preparation exceeded its cache-only zero-computation budget",
    )
    admission = prepared["source_admission"]
    require(
        admission["original_flow_source_count"] == 72
        and admission["original_flow_actual_input_count"] == 761
        and admission["original_weight_stat_count"] == 163
        and admission["original025_eligible_for_sampling"] is False
        and admission["original026_eligible_for_sampling"] is False
        and admission["inherited_episode_refs_not_reopened"] is True
        and admission["inherited_png_refs_not_reopened"] is True
        and admission["preprocessing_statistics_computed"] is False
        and admission["historical_initial_physical_sync_certified"] is False
        and admission["pretraining_visibility"] == "unknown",
        "027 changed source scope or original failed admission",
    )
    inherited = (
        "source_prepared_ref",
        "source_features_ref",
        "source_features_verification_ref",
        "source_failed_fit_verification_ref",
        "episodes",
        "anchors",
        "teacher_calls",
        "validation_generations",
        "zero_identity_cases",
        "checkpoint",
        "weight_files",
        "external_cosmos",
        "domain",
        "conservative_source_exclusion",
    )
    require(all(prepared[k] == flow[k] for k in inherited), "027 changed original025 cache/whole-EP/control interface")
    for key in (
        "train_episodes",
        "validation_episodes",
        "sigma",
        "noise_seed",
        "sampling_seeds",
        "horizon",
        "control_fps",
        "sampling_steps",
        "guidance",
        "shift",
        "hidden_size",
        "ridge",
        "teacher_gate",
        "state_gate",
    ):
        require(config[key] == flow["config"][key], "027 changed original split/time/loss or gate")
    prep_verified, fit_verified, final = (
        ancestors[k] for k in ("preparation_verification", "fit_verification", "final_peer")
    )
    require(
        prep_verified["passed"] is True
        and prep_verified["phase"] == "prepare"
        and prep_verified["preparation_verified"] is True
        and prep_verified["prepared"] == prepared["flow_prepared_ref"],
        "027 lacks original026 full preparation admission",
    )
    fit = ancestors["fit"]
    require(
        fit["complete"] is True
        and fit["passed"] is True
        and fit["eligible_for_sampling"] is False
        and fit["teacher_gate"]["passed"] is False
        and fit["producer_fit_groups"] == fit["fit_attempts"] == 2,
        "027 changed original026 failed producer fit",
    )
    require(
        fit_verified["passed"] is True
        and fit_verified["phase"] == "fit"
        and fit_verified["fit_verified"] is True
        and fit_verified["eligible_for_sampling"] is False
        and fit_verified["teacher_gate"] == fit["teacher_gate"]
        and fit_verified["independent_model_reconstructions"] == 2
        and fit_verified["prepared"] == prepared["flow_prepared_ref"],
        "027 lacks original026 failed independent fit admission",
    )
    require(
        fit_verified["input_sha256"].get(prepared["flow_fit_ref"]["path"]) == prepared["flow_fit_ref"]["sha256"]
        and fit["prepared_ref"] == prepared["flow_prepared_ref"],
        "027 reused control is not bound to original026 fit",
    )
    require(
        final["passed"] is True
        and final["eligible_for_sampling"] is False
        and final["fit"] == prepared["flow_fit_ref"]
        and final["fit_verification"] == prepared["flow_fit_verification_ref"]
        and final["actual_budget"]["actual_wam_generations"] == final["actual_budget"]["physics_steps"] == 0,
        "027 erased original026 final failure or unopened budget",
    )
    require(
        prepared["reused_bias_ref"] == fit["models"]["bias"]
        and prepared["reused_bias_predictions_ref"] == fit["teacher_predictions"],
        "027 reused bias/control predictions changed",
    )
    control = admission["actual_control"]
    require(
        control["model_ref"] == prepared["reused_bias_ref"]
        and control["predictions_ref"] == prepared["reused_bias_predictions_ref"]
        and control["control_fit_count"] == 0
        and control["control_rows"] == 5760
        and control["control_prediction_reproduction_required_during_fit"] is True,
        "027 reused control preparation admission differs",
    )
    for ref in (prepared["reused_bias_ref"], prepared["reused_bias_predictions_ref"]):
        evidence.check(ref["path"], ref["sha256"])
    with np.load(
        evidence.check(fit["models"]["bias"]["path"], fit["models"]["bias"]["sha256"]), allow_pickle=False
    ) as archive:
        require(
            archive.files == ["weights"]
            and archive["weights"].shape == (1281, 10)
            and archive["weights"].dtype == np.float64
            and not np.any(archive["weights"][:-1]),
            "027 original time-aware bias parameter identity differs",
        )
    return prepared, features, fit


def verify_fit(prepared, features, old_fit, path, evidence):
    actual = evidence.json(path)
    require(
        actual["schema_version"] == "s2a.scaled-readout-fit.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual,
        "partial/rejected027 fit",
    )
    prepared_path = Path(prepared["output_root"]) / "preparation/prepared.json"
    require(
        actual["prepared_ref"] == reference(prepared_path)
        and actual["flow_fit_ref"] == prepared["flow_fit_ref"]
        and actual["source_prepared_ref"] == prepared["source_prepared_ref"]
        and actual["features_ref"] == prepared["source_features_ref"],
        "027 fit source identity differs",
    )
    require(
        actual["new_candidate_fits"]
        == actual["candidate_fits"]
        == actual["producer_fit_groups"]
        == actual["fit_attempts"]
        == actual["reused_controls"]
        == 1,
        "027 exceeded one candidate fit or refitted its control",
    )
    require(
        actual["cuda_visible_devices"] == actual["ld_library_path"] == ""
        and all(actual[k] == 0 for k in ("new_model_calls", "new_teacher_calls", "new_episode_reads", "physics_steps")),
        "027 used undeclared model/EP/physics budget",
    )
    require(
        actual["ridge"] == 0.1
        and actual["deployment_dtype"] == "torch.bfloat16"
        and actual["eligible_for_sampling"] is False,
        "027 precision/prior or sampling authorization changed",
    )
    require(
        actual["learned_numbers"] == prepared["config"]["total_learned_numbers"] == 14091
        and actual["parameterization"] == prepared["config"]["parameterization"]
        and actual["original026_eligible_for_sampling"] is False
        and actual["runtime_qualification_executed"] is False
        and actual["reused_control_quantized_predictions_exact"] is True,
        "027 prior/capacity or unopened runtime scope differs",
    )
    require(
        actual["models"]["bias"] == actual["reused_control_ref"] == prepared["reused_bias_ref"]
        and actual["reused_teacher_predictions_ref"] == prepared["reused_bias_predictions_ref"],
        "027 fit replaced original026 control",
    )
    arrays, rows = FLOW.cached_rows(prepared, features, evidence)
    require(
        actual["teacher_rows"] == rows
        and actual["fitted_rows"] == actual["validation_rows"] == 2880
        and actual["training_episodes"] == prepared["config"]["train_episodes"]
        and actual["validation_episodes"] == prepared["config"]["validation_episodes"],
        "027 fit split or original teacher rows changed",
    )
    train, weights = training_rows(
        rows, prepared["config"]["train_episodes"], prepared["config"]["validation_episodes"]
    )
    residual = arrays["target"][train, :10].astype(np.float64) - arrays["base"][train, :10].astype(np.float64)
    reconstructed = augmented_fit(arrays["hidden"][train], arrays["sigma_actual"][train], residual, weights)
    model_ref = actual["models"]["adapt"]
    with np.load(evidence.check(model_ref["path"], model_ref["sha256"]), allow_pickle=False) as archive:
        require(
            set(archive.files) == {"weights", "center", "scale"}, "027 model omitted preprocessing or added parameters"
        )
        saved = {k: numeric(archive[k], k).copy() for k in archive.files}
    require(
        all(v.dtype == np.float64 for v in saved.values())
        and saved["weights"].shape == (1281, 10)
        and saved["center"].shape == (1280,)
        and saved["scale"].shape == ()
        and saved["scale"] > 0,
        "027 model precision/shape differs",
    )
    differences = {}
    for key in ("weights", "center", "scale"):
        close(saved[key], reconstructed[key], f"027 independent train-only {key}")
        differences[key] = float(np.max(np.abs(saved[key] - reconstructed[key])))
    scaled_hidden = (arrays["hidden"][train].astype(np.float64) - saved["center"]) / saved["scale"]
    norm_squared = float(np.sum(weights * np.sum(scaled_hidden**2, axis=1)))
    close(norm_squared, 1.0, "027 train global normalized energy")
    metadata = actual["preprocessing"]
    require(
        metadata["definition"] == prepared["config"]["preprocessing"]
        and metadata["train_rows"] == 2880
        and metadata["training_episodes"] == prepared["config"]["train_episodes"],
        "027 preprocessing provenance changed",
    )
    close(metadata["weighted_centered_norm_squared"], norm_squared, "027 recorded preprocessing energy")
    predictions = dict(
        raw=arrays["base"],
        adapt=quantized_velocity(
            arrays["hidden"], arrays["base"], arrays["sigma_actual"], saved["weights"], saved["center"], saved["scale"]
        ),
    )
    bias_ref = prepared["reused_bias_ref"]
    with np.load(evidence.check(bias_ref["path"], bias_ref["sha256"]), allow_pickle=False) as archive:
        bias = numeric(archive["weights"], "old026 bias").copy()
    predictions["bias"] = quantized_velocity(
        arrays["hidden"], arrays["base"], arrays["sigma_actual"], bias, saved["center"], saved["scale"]
    )
    exact(
        predictions["bias"],
        FLOW.quantized_velocity(arrays["hidden"], arrays["base"], arrays["sigma_actual"], bias),
        "027 preprocessing cannot change the reused time-aware bias",
    )
    old_predictions_ref = old_fit["teacher_predictions"]
    with np.load(
        evidence.check(old_predictions_ref["path"], old_predictions_ref["sha256"]), allow_pickle=False
    ) as archive:
        for key in ("raw", "bias"):
            exact(
                predictions[key].view(np.uint32),
                archive[key].view(np.uint32),
                f"027 original026 actual {key} full128 bits",
            )
        exact(archive["training_weights"], weights, "027 old026 training weights")
        exact(archive["sigma_actual"], arrays["sigma_actual"], "027 old026 actual sigma")
    predictions_ref = actual["teacher_predictions"]
    with np.load(evidence.check(predictions_ref["path"], predictions_ref["sha256"]), allow_pickle=False) as archive:
        require(
            set(archive.files) == {"raw", "adapt", "bias", "training_weights", "sigma_actual"},
            "027 teacher prediction roster differs",
        )
        exact(archive["training_weights"], weights, "027 unchanged hierarchical weights")
        require(archive["sigma_actual"].dtype == np.float64, "027 actual FP32 sigma must expand to FP64")
        exact(archive["sigma_actual"], arrays["sigma_actual"], "027 unchanged actual sigma")
        for name, values in predictions.items():
            require(
                archive[name].dtype == np.float32 and archive[name].shape == (5760, 128),
                "027 teacher must retain full FP32 BF16 expansion",
            )
            exact(
                archive[name].view(np.uint32),
                values.view(np.uint32),
                f"027 actual quantized teacher {name} full128 bits",
            )
            exact(archive[name][:, 10:], arrays["base"][:, 10:], f"027 inactive118 {name}")
    metrics = OLD.all_teacher_metrics(predictions, arrays["target"], rows)
    OLD.compare_metrics(actual["teacher_metrics"], metrics)
    gate = OLD.independent_teacher_gate(metrics)
    require(
        actual["teacher_gate"] == gate
        and actual["eligible_for_runtime_qualification"] is gate["passed"]
        and actual["eligible_for_sampling"] is False,
        "027 changed quality thresholds or authorized unregistered sampling",
    )
    return dict(
        fit_verified=True,
        independent_candidate_reconstructions=1,
        reused_controls_verified=1,
        max_parameter_abs_difference=differences["weights"],
        max_preprocessing_abs_difference=dict(center=differences["center"], scale=differences["scale"]),
        weighted_centered_norm_squared=norm_squared,
        teacher_metrics=metrics,
        teacher_gate=gate,
        eligible_for_runtime_qualification=gate["passed"],
        eligible_for_sampling=False,
        raw_checkpoint_task_success_proven=False,
        adapted_state_or_task_capability_proven=False,
    )


def phase_arguments(args):
    require((args.fit is not None) is (args.phase == "fit"), "027 phase requires exactly its fit evidence argument")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "027 verifier requires CPU emptyCUDA/LD and seed42",
    )
    phase_arguments(args)
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "027 receipt must be new under outputs; refuse overwrite",
    )
    evidence = OLD.Evidence()
    receipt = dict(
        schema_version="s2a.scaled-readout-independent-verification.v1",
        phase=args.phase,
        passed=False,
        new_model_calls=0,
        new_candidate_fits=0,
        new_episode_reads=0,
        new_physics_steps=0,
    )
    error = None
    try:
        prepared, features, old_fit = verify_preparation(args.prepared, evidence)
        receipt["prepared"] = reference(args.prepared)
        if args.phase == "prepare":
            receipt.update(
                preparation_verified=True,
                old_features_reused=144,
                old026_eligible=False,
                reused_controls_verified=1,
                eligible_for_sampling=False,
            )
        else:
            receipt.update(verify_fit(prepared, features, old_fit, args.fit, evidence))
        evidence.recheck()
        receipt["passed"] = True
    except Exception as caught:
        error = caught
        receipt["error"] = dict(type=type(caught).__name__, message=str(caught))
    receipt["input_sha256"] = evidence.hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=receipt["passed"], output=reference(output))))
    if error is not None:
        raise SystemExit(f"027 independent audit rejected: {error}")


if __name__ == "__main__":
    main()
