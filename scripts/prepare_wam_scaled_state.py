"""Prepare028's frozen027 model and evidence without replaying any fit.

Preparation is CPU-only and cache-only. Source episode arrays and PNGs remain
inherited references; the new precision/model stages belong to the owner.
"""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
from prepare_wam_flow_readout import Evidence, external_identity, reference, require, weight_stats
from prepare_wam_scaled_readout import (
    FLOW_REFS,
    INHERITED,
    cache_only_hashes,
    forbidden_observation_paths,
    owned_cpu_exits,
)

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_scaled_readout_state_20261021.json"
CONFIG_SHA = "183b4a9407bbfaea8e82c1d7c052f5417e32def31805528102e3bd293a6cea26"
PLAN = REPO / "plans/v0.0.1/028_scaled_readout_state_20261021.md"
PLAN_SHA = "cfe26406b98ada3fb82632e067bf1990af71342c055b672c5d3148fad11af451"
NEW_SOURCES = (
    "scripts/run_wam_scaled_state.py",
    "tests/test_wam_scaled_state_runner.py",
    "scripts/probe_wam_scaled_readout_quantization.py",
    "tests/test_wam_scaled_readout_quantization.py",
    "scripts/sample_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_sampling.py",
    "scripts/prepare_wam_scaled_state.py",
    "tests/test_wam_scaled_state_preparation.py",
    "scripts/verify_wam_scaled_state.py",
    "tests/test_wam_scaled_state_verification.py",
)
SCALED_REFS = (
    ("source_scaled_prepared", "scaled_prepared_ref"),
    ("source_scaled_fit", "scaled_fit_ref"),
    ("source_scaled_fit_verification", "scaled_fit_verification_ref"),
    ("source_scaled_final_peer", "scaled_final_peer_ref"),
    ("source_scaled_owner", "scaled_owner_ref"),
)
INHERITED_PROVENANCE = (
    INHERITED
    + tuple(name for _, name in FLOW_REFS)
    + (
        "reused_bias_ref",
        "reused_bias_predictions_ref",
    )
)


def scaled_admission(scaled, fit, checked, final, owner, refs):
    require(
        scaled["schema_version"] == "s2a.scaled-readout-preparation.v1"
        and len(scaled["source_sha256"]) == len(scaled["snapshot_sha256"]) == 84
        and len(scaled["input_sha256"]) == 874,
        "027 original84 sources/874 actual inputs differ",
    )
    require(
        fit["schema_version"] == "s2a.scaled-readout-fit.v1"
        and fit["complete"] is True
        and fit["passed"] is True
        and fit["prepared_ref"] == refs["scaled_prepared_ref"]
        and fit["features_ref"] == scaled["source_features_ref"]
        and fit["candidate_fits"] == fit["fit_attempts"] == fit["producer_fit_groups"] == 1
        and fit["reused_controls"] == 1
        and fit["fitted_rows"] == fit["validation_rows"] == 2880
        and fit["eligible_for_runtime_qualification"] is True
        and fit["eligible_for_sampling"] is False,
        "027 actual single-fit teacher qualification or original sampling False differs",
    )
    gate = fit["teacher_gate"]
    require(
        gate["passed"] is True
        and set(gate["checks"]) == {"pooled_improvement", "beats_constant_bias", "all_sigmas_nonregressing"}
        and all(v is True for v in gate["checks"].values()),
        "028 requires every original027 teacher gate",
    )
    require(
        checked["passed"] is True
        and checked["phase"] == "fit"
        and checked["fit_verified"] is True
        and checked["prepared"] == refs["scaled_prepared_ref"]
        and checked["independent_candidate_reconstructions"] == checked["reused_controls_verified"] == 1
        and checked["teacher_gate"] == gate
        and checked["eligible_for_runtime_qualification"] is True
        and checked["eligible_for_sampling"] is False
        and checked["input_sha256"].get(refs["scaled_fit_ref"]["path"]) == refs["scaled_fit_ref"]["sha256"],
        "027 independent single reconstruction/teacher identity differs",
    )
    require(
        final["passed"] is True
        and final["prepared"] == refs["scaled_prepared_ref"]
        and final["fit"] == refs["scaled_fit_ref"]
        and final["fit_verification"] == refs["scaled_fit_verification_ref"]
        and final["fit_owner"] == refs["scaled_owner_ref"]
        and final["source_before_after_unchanged"] is True
        and final["frozen_sources"] == final["source_snapshots"] == 84
        and final["weight_stat_count"] == 163
        and final["teacher_gate"] == gate
        and final["eligible_for_runtime_qualification"] is True
        and final["eligible_for_sampling"] is False
        and final["reused026_full128_raw_and_bias_prediction_bits_exact"] is True
        and final["reused026_training_weights_and_actual_sigmas_exact"] is True,
        "027 final peer/unchanged original sampling authorization differs",
    )
    require(
        owner["complete"] is True
        and owner["passed"] is True
        and owner["prepared_ref"] == refs["scaled_prepared_ref"]
        and owner["producer_result"] == refs["scaled_fit_ref"]
        and owner["eligible_for_runtime_qualification"] is True
        and owner["eligible_for_sampling"] is False
        and owner["automatic_retry"] is False
        and len(owner["owned_commands"]) == 2,
        "027 actual owned CPU stage did not complete uniquely",
    )
    require(
        final["actual_budget"]["candidate_fits"] == final["actual_budget"]["independent_candidate_reconstructions"] == 1
        and all(
            final["actual_budget"][k] == 0
            for k in (
                "new_control_fits",
                "gpu_jobs",
                "new_teacher_calls",
                "wam_denoise_calls",
                "complete_generations",
                "state_label_scoring",
                "physics_steps",
                "fresh_cases",
            )
        ),
        "027 prior evidence exceeded its CPU-only budget",
    )
    for actual, proof in zip(owner["owned_commands"], final["owned_cpu_children"], strict=True):
        require(all(actual[k] == proof[k] for k in ("launch", "exit", "release")), "027 owned exit refs disagree")


def model_admission(fit, checked, final, evidence):
    model_ref, predictions_ref = fit["models"]["adapt"], fit["teacher_predictions"]
    for ref in (model_ref, predictions_ref):
        require(checked["input_sha256"].get(ref["path"]) == ref["sha256"], "027 audit omitted actual model/predictions")
        evidence.check(ref["path"], ref["sha256"])
    with np.load(model_ref["path"], allow_pickle=False) as archive:
        shapes = dict(weights=(1281, 10), center=(1280,), scale=())
        require(set(archive.files) == set(shapes), "027 model parameter roster differs")
        for name, shape in shapes.items():
            value = archive[name]
            require(
                value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(),
                "027 frozen model shape/FP64/finite contract differs",
            )
        require(
            float(archive["scale"]) > 0 and fit["learned_numbers"] == 14091,
            "027 global scalar scale/parameter count differs",
        )
        scale = float(archive["scale"])
    require(
        fit["ridge"] == 0.1
        and fit["deployment_dtype"] == "torch.bfloat16"
        and fit["reused_control_quantized_predictions_exact"] is True,
        "027 actual deployment/control identity differs",
    )
    return dict(
        adapted_model_ref=model_ref,
        teacher_predictions_ref=predictions_ref,
        learned_numbers=14091,
        parameter_shapes={k: list(v) for k, v in shapes.items()},
        parameter_dtype="float64",
        scale=scale,
        independent_max_parameter_abs_difference=checked["max_parameter_abs_difference"],
        independent_max_preprocessing_abs_difference=checked["max_preprocessing_abs_difference"],
        original_final_peer_control_exact=final["reused026_full128_raw_and_bias_prediction_bits_exact"],
        new_parameter_reconstructions=0,
    )


def source_closure(scaled, evidence):
    sources = dict(scaled["source_sha256"])
    for path in [REPO / p for p in NEW_SOURCES] + [CONFIG, PLAN]:
        sources[str(path.resolve())] = sha256(evidence.check(path))
    require(len(sources) == 96, "028 requires original84 plus new10 sources and plan/config")
    return sources


def prepare():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "028 preparation requires S2A cwd and explicitly empty CUDA/LD",
    )
    evidence = Evidence()
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    root = REPO / config["output_dir"]
    require(root.is_relative_to(REPO / "outputs") and not root.exists(), "028 refuses existing/non-output root")
    refs, ancestors = {}, {}
    for source, name in SCALED_REFS:
        path = REPO / config[source]
        ancestors[name] = evidence.json(path, config[f"{source}_sha256"])
        refs[name] = reference(path)
    scaled, fit, checked, final, owner = (ancestors[name] for _, name in SCALED_REFS)
    scaled_admission(scaled, fit, checked, final, owner, refs)
    forbidden = forbidden_observation_paths(scaled)
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        cache_only_hashes(scaled[key], forbidden, evidence)
    for ancestor in (fit, checked, final):
        cache_only_hashes(ancestor["input_sha256"], forbidden, evidence)
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
        "domain",
        "checkpoint",
        "cosmos_repo",
        "parameterization",
        "residual_precision",
    ):
        require(config[key] == scaled["config"][key], f"028 changed inherited027 contract: {key}")
    weight_stats(scaled["weight_files"])
    require(external_identity(config["cosmos_repo"]) == scaled["external_cosmos"], "027 external clean8ed differs")
    model = model_admission(fit, checked, final, evidence)
    exits = owned_cpu_exits(final, evidence)
    sources = source_closure(scaled, evidence)
    root.mkdir()
    directory = root / "preparation"
    directory.mkdir()
    try:
        snapshots = directory / "source_snapshot"
        snapshots.mkdir()
        copies = {}
        for i, (path, digest) in enumerate(sources.items()):
            target = snapshots / f"{i:03d}_{Path(path).name}"
            shutil.copyfile(path, target)
            require(sha256(target) == digest, "028 source snapshot differs")
            copies[str(target)] = digest
        evidence.recheck()
        weight_stats(scaled["weight_files"])
        require(external_identity(config["cosmos_repo"]) == scaled["external_cosmos"], "external source changed")
        result = dict(
            schema_version="s2a.scaled-state-preparation.v1",
            config=config,
            config_ref=reference(CONFIG),
            registration=reference(PLAN),
            output_root=str(root),
            **refs,
            **{key: copy.deepcopy(scaled[key]) for key in INHERITED_PROVENANCE},
            adapted_model_ref=model["adapted_model_ref"],
            scaled_teacher_predictions_ref=model["teacher_predictions_ref"],
            source_admission=dict(
                original_source_count=84,
                original_actual_input_count=874,
                original_weight_stat_count=163,
                original027_teacher_passed=True,
                original027_runtime_qualification_eligible=True,
                original027_eligible_for_sampling=False,
                inherited_episode_refs_not_reopened=True,
                inherited_png_refs_not_reopened=True,
                preprocessing_statistics_recomputed=False,
                model=model,
                prior_owned_cpu_exits=exits,
                new_precision_required=True,
                historical_initial_physical_sync_certified=False,
                pretraining_visibility="unknown",
            ),
            budget=config["budget"],
            source_sha256=sources,
            snapshot_sha256=copies,
            input_sha256=evidence.hashes,
            new_model_calls=0,
            new_teacher_calls=0,
            new_episode_reads=0,
            new_png_reads=0,
            fit_count=0,
            independent_parameter_reconstructions=0,
            physics_steps=0,
            fresh_performance_cases=0,
            full_weight_bytes_rehashed=False,
        )
        path = directory / "prepared.json"
        write_once(path, result)
        return path
    except BaseException as error:
        write_once(directory / "error.json", dict(type=type(error).__name__, message=str(error), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(args.seed == 42, "028 requires registered seed42")
    path = prepare()
    print(json.dumps(dict(prepared=reference(path), model_calls=0, fits=0, episode_reads=0)))


if __name__ == "__main__":
    main()
