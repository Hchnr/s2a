"""027 one CPU candidate solve and exact reuse of the admitted026 control."""

import argparse
import json
import os
from pathlib import Path

import fit_wam_readout as original
import numpy as np
from fit_wam_flow_readout import actual_sigmas

from s2a.artifacts import REPO
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext
from s2a.wam_scaled_readout import ScaledFlowReadout

CONFIG = REPO / "configs/wam_train_scale_readout_20261021.json"
CONFIG_SHA = "dd0e0124d256a953af48ea2e548cca6603cd9f777d6bd5e4f620548591a2c06b"
PLAN = REPO / "plans/v0.0.1/027_train_scale_readout_20261021.md"
PLAN_SHA = "586ddf2656d39752efd7979e1877af940004e14b058d900c7ef464661cbf9fb1"


def exact_bits(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    original.require(
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.tobytes(order="C") == expected.tobytes(order="C"),
        f"027 exact source/prediction differs: {label}",
    )


def reused_control(prepared, flow_fit, arrays, rows, weights, sigma, evidence):
    """Validate the complete old control before the sole new candidate solve."""
    original.require(
        flow_fit["complete"] is True
        and flow_fit["passed"] is True
        and flow_fit["eligible_for_sampling"] is False
        and flow_fit["teacher_gate"]["passed"] is False,
        "027 must preserve026 completed failed candidate",
    )
    original.require(
        flow_fit["features_ref"] == prepared["source_features_ref"] and flow_fit["teacher_rows"] == rows,
        "027 old control teacher source or row order differs",
    )
    for key in ("training_episodes", "validation_episodes"):
        original.require(
            flow_fit[key]
            == prepared["config"]["train_episodes" if key == "training_episodes" else "validation_episodes"],
            "027 old control whole-episode split differs",
        )
    original.require(
        flow_fit["ridge"] == 0.1 and flow_fit["deployment_dtype"] == "torch.bfloat16",
        "027 original control prior/dtype differs",
    )
    reference = flow_fit["models"]["bias"]
    original.require(
        prepared["reused_bias_ref"] == reference
        and prepared["reused_bias_predictions_ref"] == flow_fit["teacher_predictions"],
        "027 prepared control refs differ from026",
    )
    with np.load(evidence.check(reference["path"], reference["sha256"]), allow_pickle=False) as archive:
        original.require(archive.files == ["weights"], "027 reused control contains unregistered parameters")
        coefficients = archive["weights"].copy()
    original.require(
        coefficients.shape == (1281, 10)
        and coefficients.dtype == np.float64
        and np.isfinite(coefficients).all()
        and not np.any(coefficients[:-1]),
        "027 reused bias must retain zero hidden coefficients",
    )
    predictions_ref = flow_fit["teacher_predictions"]
    with np.load(evidence.check(predictions_ref["path"], predictions_ref["sha256"]), allow_pickle=False) as archive:
        exact_bits(archive["training_weights"], weights, "026 original training weights")
        exact_bits(archive["sigma_actual"], sigma, "026 actual sigma rows")
        exact_bits(archive["raw"], arrays["base"], "026 raw quantized base")
        predictions = archive["bias"].copy()
    return coefficients, predictions


def execute(args):
    original.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "" and args.seed == 42,
        "027 fit requires CPU emptyCUDA/LD and seed42",
    )
    evidence = original.Evidence()
    prepared = evidence.json(args.prepared, args.prepared_sha256)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    original.require(
        prepared["schema_version"] == "s2a.scaled-readout-preparation.v1"
        and prepared["config"] == config
        and prepared["config_ref"] == original.reference(CONFIG)
        and prepared["registration"] == original.reference(PLAN),
        "027 registered preparation/config identity differs",
    )
    for key in ("source_sha256", "input_sha256", "snapshot_sha256"):
        for path, digest in prepared[key].items():
            evidence.check(path, digest)
    original.require(
        str(Path(__file__).resolve()) in prepared["source_sha256"], "027 preparation omitted actual fit source"
    )
    for path, item in prepared["weight_files"].items():
        stat = Path(path).stat()
        original.require(
            (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]), "027 frozen8400 weight stat differs"
        )
    old = evidence.json_ref(prepared["source_prepared_ref"])
    features = evidence.json_ref(prepared["source_features_ref"])
    flow_prepared = evidence.json_ref(prepared["flow_prepared_ref"])
    flow_fit = evidence.json_ref(prepared["flow_fit_ref"])
    flow_checked = evidence.json_ref(prepared["flow_fit_verification_ref"])
    flow_final = evidence.json_ref(prepared["flow_final_peer_ref"])
    original.require(
        flow_prepared["source_prepared_ref"] == prepared["source_prepared_ref"]
        and flow_prepared["source_features_ref"] == prepared["source_features_ref"],
        "027 inherited026 feature identity differs",
    )
    original.require(
        flow_checked["passed"] is True
        and flow_checked["fit_verified"] is True
        and flow_checked["eligible_for_sampling"] is False
        and flow_checked["teacher_gate"]["passed"] is False
        and flow_checked["input_sha256"].get(prepared["flow_fit_ref"]["path"]) == prepared["flow_fit_ref"]["sha256"]
        and flow_final["passed"] is True,
        "027 requires admitted026 fit and unchanged negative gate",
    )
    for key in ("anchors", "episodes", "teacher_calls"):
        original.require(prepared[key] == old[key], "027 changed original whole-episode/anchor split")
    output = Path(prepared["output_root"]) / "fit"
    original.require(
        output.parent == (REPO / config["output_dir"]).resolve() and output.parent.is_relative_to(REPO / "outputs"),
        "027 fit output root differs",
    )
    output.mkdir(exist_ok=False)
    result = dict(
        schema_version="s2a.scaled-readout-fit.v1",
        complete=False,
        passed=False,
        prepared_ref=original.reference(args.prepared),
        source_prepared_ref=prepared["source_prepared_ref"],
        features_ref=prepared["source_features_ref"],
        flow_fit_ref=prepared["flow_fit_ref"],
        models={},
        candidate_fits=0,
        new_candidate_fits=0,
        producer_fit_groups=0,
        fit_attempts=0,
        reused_controls=0,
        eligible_for_runtime_qualification=False,
        eligible_for_sampling=False,
        new_model_calls=0,
        new_teacher_calls=0,
        new_episode_reads=0,
        physics_steps=0,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        ld_library_path=os.environ["LD_LIBRARY_PATH"],
    )
    try:
        import torch

        arrays, rows = original.load_rows(old, features, evidence, torch)
        sigma = actual_sigmas(features, evidence)
        train = np.asarray([row["split"] == "train" for row in rows])
        weights = original.hierarchical_weights([row for row in rows if row["split"] == "train"])
        bias_weights, old_bias_predictions = reused_control(prepared, flow_fit, arrays, rows, weights, sigma, evidence)
        target = arrays["target"][train, :10].astype(np.float64) - arrays["base"][train, :10].astype(np.float64)
        result["fit_attempts"] = 1
        model = ScaledFlowReadout.fit(arrays["hidden"][train], sigma[train], target, weights)
        result.update(candidate_fits=1, new_candidate_fits=1, producer_fit_groups=1)
        model_path = output / "adapt.npz"
        with model_path.open("xb") as stream:
            np.savez_compressed(
                stream, weights=model.weights, center=model.center, scale=np.asarray(model.scale, np.float64)
            )
        result["models"] = dict(adapt=original.reference(model_path), bias=flow_fit["models"]["bias"])
        control = ScaledFlowReadout(bias_weights, model.center, model.scale)
        hidden = torch.from_numpy(arrays["hidden"]).to(torch.bfloat16)
        base = torch.from_numpy(arrays["base"]).to(torch.bfloat16)
        ids = torch.zeros(len(base), dtype=torch.long)
        context = DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)
        predictions = dict(raw=base.float().numpy())
        with torch.inference_mode():
            for name, fitted in (("adapt", model), ("bias", control)):
                predictions[name] = fitted.apply_torch(hidden, base, context, ids, sigma).float().numpy()
                exact_bits(predictions[name][:, 10:], arrays["base"][:, 10:], f"{name} inactive118")
        exact_bits(predictions["bias"], old_bias_predictions, "026 reused bias through027 actual deployment path")
        result["reused_controls"] = 1
        artifact = output / "teacher_predictions.npz"
        with artifact.open("xb") as stream:
            np.savez_compressed(stream, **predictions, training_weights=weights, sigma_actual=sigma)
        metrics = original.teacher_metrics(predictions, arrays["target"], rows)
        gate = original.teacher_gate(metrics)
        centered = (arrays["hidden"][train].astype(np.float64) - model.center) / model.scale
        result.update(
            complete=True,
            passed=True,
            eligible_for_runtime_qualification=gate["passed"],
            eligible_for_sampling=False,
            teacher_gate=gate,
            teacher_metrics=metrics,
            teacher_rows=rows,
            teacher_predictions=original.reference(artifact),
            reused_control_ref=flow_fit["models"]["bias"],
            reused_teacher_predictions_ref=flow_fit["teacher_predictions"],
            reused_control_quantized_predictions_exact=True,
            preprocessing=dict(
                definition=config["preprocessing"],
                train_rows=int(train.sum()),
                training_episodes=config["train_episodes"],
                weighted_centered_norm_squared=float(np.sum(weights * np.sum(centered**2, axis=1))),
            ),
            fitted_rows=int(train.sum()),
            validation_rows=int((~train).sum()),
            training_episodes=config["train_episodes"],
            validation_episodes=config["validation_episodes"],
            learned_numbers=int(model.weights.size + model.center.size + 1),
            ridge=0.1,
            torch_version=torch.__version__,
            deployment_dtype="torch.bfloat16",
            parameterization=config["parameterization"],
            fit_input_contract="train-only mu/global L2 RMS; A=-[(h-mu)/s,1]/actual_sigma; Y=active10(target-base); unchanged weights",
            original026_eligible_for_sampling=False,
            runtime_qualification_executed=False,
        )
        evidence.recheck()
        for path, item in prepared["weight_files"].items():
            stat = Path(path).stat()
            original.require(
                (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]),
                "027 frozen8400 stat changed during CPU fit",
            )
    except BaseException as error:
        result.update(
            complete=False, passed=False, eligible_for_runtime_qualification=False, eligible_for_sampling=False
        )
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result["input_sha256"] = evidence.hashes
        original.write_new(output / "fit_result.json", result)
    return output / "fit_result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    path = execute(parser.parse_args())
    print(json.dumps(dict(result=original.reference(path))))


if __name__ == "__main__":
    main()
