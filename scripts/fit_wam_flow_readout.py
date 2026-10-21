"""026 fixed two-fit CPU stage, consuming only admitted025 cached features."""

import argparse
import json
import os
from pathlib import Path

import fit_wam_readout as original
import numpy as np

from s2a.artifacts import REPO
from s2a.wam_flow_readout import FlowResidualReadout
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext

CONFIG = REPO / "configs/wam_flow_consistent_readout_20261021.json"
CONFIG_SHA = "2d0e45b3ae02ecec4ee45bcaa1928596e347417f945f8903416f0df314a68534"
PLAN = REPO / "plans/v0.0.1/026_flow_consistent_readout_20261021.md"
PLAN_SHA = "42c0435773306852d4aa79be4d319f9de9cc71ff7823b7ad830286c2c41fc2fa"
GPU_UUID = "7f176867-170c-b508-5c89-1cd819ad3b8e"


def actual_sigmas(features, evidence):
    """Read each saved FP32 value; JSON's nominal sigma is only a grid label."""
    result = []
    for call in features["teacher_calls"]:
        meta = evidence.json_ref(call["bundle"])
        with np.load(evidence.check(meta["arrays"]["path"], meta["arrays"]["sha256"]), allow_pickle=False) as archive:
            sigma = archive["sigma_action"]
        original.require(
            sigma.dtype == np.float32 and sigma.shape == (40, 1) and np.isfinite(sigma).all() and np.all(sigma > 0),
            "026 actual cached sigma_action must be40×1 positive FP32",
        )
        original.require(
            np.array_equal(sigma, np.full((40, 1), call["sigma"], dtype=np.float32)),
            "026 cached sigma does not match its original teacher call",
        )
        result.append(sigma[:, 0].astype(np.float64))
    result = np.concatenate(result)
    original.require(result.shape == (5760,), "026 requires the original5760 cached sigma rows")
    return result


def precision_admission(reference, prepared_ref, core_sha, evidence):
    result = evidence.json_ref(reference)
    original.require(
        result["schema_version"] == "s2a.flow-readout-synthetic-cpu-cuda-precision.v1"
        and result["complete"] is True
        and result["passed"] is True
        and result["prepared_ref"] == prepared_ref,
        "026 actual synthetic precision admission is required before fit",
    )
    original.require(
        result["core_sha256"] == core_sha
        and result["physical_gpu"] == 6
        and result["actual_gpu_uuid"].replace("GPU-", "") == GPU_UUID
        and result["cpu_cuda_bitwise_equal"] is True,
        "026 precision admission does not certify this core/physicalGPU6",
    )
    original.require(
        all(
            result[key] == 0
            for key in ("model_weights_loaded", "wam_denoise_calls", "fit_groups", "episode_reads", "physics_steps")
        ),
        "026 synthetic precision admission exceeded its zero scientific-work budget",
    )
    exit_record = evidence.json_ref(result["exit_ref"])
    release = evidence.json_ref(result["release_ref"])
    gpu = evidence.json_ref(result["gpu_after_ref"])
    original.require(
        exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0),
        "026 precision worker has no normal actual wait",
    )
    original.require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "026 precision worker was not normally reaped",
    )
    original.require(
        gpu["physical_gpu"] == 6 and gpu["memory_mib"] == 0 and gpu["released"] is True,
        "026 precision GPU6 release is unproven",
    )
    if "actual_outputs" in result:
        evidence.check(result["actual_outputs"]["path"], result["actual_outputs"]["sha256"])
    return result


def execute(args):
    original.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "" and args.seed == 42,
        "026 fit requires CPU emptyCUDA/LD and seed42",
    )
    evidence = original.Evidence()
    prepared = evidence.json(args.prepared, args.prepared_sha256)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    original.require(
        prepared["schema_version"] == "s2a.flow-readout-preparation.v1"
        and prepared["config"] == config
        and prepared["config_ref"] == original.reference(CONFIG)
        and prepared["registration"] == original.reference(PLAN),
        "026 registered preparation/config identity differs",
    )
    for key in ("source_sha256", "input_sha256", "snapshot_sha256"):
        for path, digest in prepared[key].items():
            evidence.check(path, digest)
    original.require(
        str(Path(__file__).resolve()) in prepared["source_sha256"], "026 preparation omitted actual fit source"
    )
    for path, item in prepared["weight_files"].items():
        stat = Path(path).stat()
        original.require(
            (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]), "026 frozen8400 weight stat changed"
        )
    old = evidence.json_ref(prepared["source_prepared_ref"])
    features = evidence.json_ref(prepared["source_features_ref"])
    verified = evidence.json_ref(prepared["source_features_verification_ref"])
    failed_fit = evidence.json_ref(prepared["source_failed_fit_verification_ref"])
    original.require(
        verified["passed"] is True
        and verified["phase"] == "features"
        and verified["prepared"] == prepared["source_prepared_ref"]
        and verified["input_sha256"].get(prepared["source_features_ref"]["path"])
        == prepared["source_features_ref"]["sha256"],
        "026 requires exact independently admitted025 features",
    )
    original.require(
        failed_fit["passed"] is True
        and failed_fit["eligible_for_sampling"] is False
        and failed_fit["teacher_gate"]["passed"] is False,
        "026 must preserve original025 failed teacher gate",
    )
    for key in ("anchors", "episodes", "teacher_calls"):
        original.require(prepared[key] == old[key], "026 changed original whole-episode/anchor split")
    quantization_ref = dict(path=str(args.quantization_admission.resolve()), sha256=args.quantization_admission_sha256)
    precision_admission(
        quantization_ref,
        original.reference(args.prepared),
        prepared["source_sha256"][str(REPO / "src/s2a/wam_flow_readout.py")],
        evidence,
    )
    output = Path(prepared["output_root"]) / "fit"
    original.require(
        output.parent == (REPO / config["output_dir"]).resolve() and output.parent.is_relative_to(REPO / "outputs"),
        "026 fit output root differs",
    )
    output.mkdir(exist_ok=False)
    result = dict(
        schema_version="s2a.flow-readout-fit.v1",
        complete=False,
        passed=False,
        prepared_ref=original.reference(args.prepared),
        features_ref=prepared["source_features_ref"],
        source_prepared_ref=prepared["source_prepared_ref"],
        quantization_ref=quantization_ref,
        models={},
        producer_fit_groups=0,
        fit_attempts=0,
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

        # The reused reader opens only the144 existing feature bundles and zero
        # response JSONs. It never follows source EP arrays/PNG references.
        arrays, rows = original.load_rows(old, features, evidence, torch)
        sigma = actual_sigmas(features, evidence)
        train = np.asarray([row["split"] == "train" for row in rows])
        weights = original.hierarchical_weights([row for row in rows if row["split"] == "train"])
        target = arrays["target"][train, :10].astype(np.float64) - arrays["base"][train, :10].astype(np.float64)
        models = {}
        for name, fit in (("adapt", FlowResidualReadout.fit), ("bias", FlowResidualReadout.fit_bias)):
            result["fit_attempts"] += 1
            model = fit(arrays["hidden"][train], sigma[train], target, weights)
            result["producer_fit_groups"] += 1
            path = output / f"{name}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, weights=model.weights)
            models[name] = model
            result["models"][name] = original.reference(path)
        hidden = torch.from_numpy(arrays["hidden"]).to(torch.bfloat16)
        base = torch.from_numpy(arrays["base"]).to(torch.bfloat16)
        ids = torch.zeros(len(base), dtype=torch.long)
        context = DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)
        predictions = dict(raw=base.float().numpy())
        with torch.inference_mode():
            for name, model in models.items():
                predictions[name] = model.apply_torch(hidden, base, context, ids, sigma).float().numpy()
                original.require(
                    np.array_equal(predictions[name][:, 10:], arrays["base"][:, 10:]), "026 changed inactive118 slots"
                )
        artifact = output / "teacher_predictions.npz"
        with artifact.open("xb") as stream:
            np.savez_compressed(stream, **predictions, training_weights=weights, sigma_actual=sigma)
        metrics = original.teacher_metrics(predictions, arrays["target"], rows)
        gate = original.teacher_gate(metrics)
        result.update(
            complete=True,
            passed=True,
            eligible_for_sampling=gate["passed"],
            teacher_gate=gate,
            teacher_metrics=metrics,
            teacher_rows=rows,
            teacher_predictions=original.reference(artifact),
            fitted_rows=int(train.sum()),
            validation_rows=int((~train).sum()),
            training_episodes=config["train_episodes"],
            validation_episodes=config["validation_episodes"],
            torch_version=torch.__version__,
            deployment_dtype="torch.bfloat16",
            ridge=0.1,
            parameterization=config["parameterization"],
            fit_input_contract="A=-phi/actual_FP32_sigma_expanded_to_FP64; Y=active10(target-base); unchanged velocity weights; regularized bias",
            prior_units="W is clean-action delta_x0; regularization prior differs from025 velocity residual",
            original025_eligible_for_sampling=False,
        )
        evidence.recheck()
        for path, item in prepared["weight_files"].items():
            stat = Path(path).stat()
            original.require(
                (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]),
                "026 frozen8400 stat changed during fit",
            )
    except BaseException as error:
        result.update(complete=False, passed=False, eligible_for_sampling=False)
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
    parser.add_argument("--quantization-admission", type=Path, required=True)
    parser.add_argument("--quantization-admission-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    path = execute(parser.parse_args())
    print(json.dumps(dict(result=original.reference(path))))


if __name__ == "__main__":
    main()
