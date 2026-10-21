"""028 independent CPU audit of frozen scaled-readout runtime and states.

No producer mathematics, Torch, Cosmos or dataloader is imported. Preparation
and precision never open source EP/PNG files. Only complete, released sampling
may open the original six validation state/mask archives, after all99 contracts.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

CONFIG = REPO / "configs/wam_scaled_readout_state_20261021.json"
CONFIG_SHA = "183b4a9407bbfaea8e82c1d7c052f5417e32def31805528102e3bd293a6cea26"
PLAN = REPO / "plans/v0.0.1/028_scaled_readout_state_20261021.md"
PLAN_SHA = "cfe26406b98ada3fb82632e067bf1990af71342c055b672c5d3148fad11af451"
HELPER = REPO / "scripts/verify_wam_scaled_readout.py"
HELPER_SHA = "694e6b125fd84c9aef9859bb15ea087b29a1fbc1f4408f28b282c2575f5af84f"
GPU_UUID = "GPU-7f176867-170c-b508-5c89-1cd819ad3b8e"
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


def frozen_helpers():
    if sha256(HELPER) != HELPER_SHA:
        raise ValueError("frozen027 independent helper changed")
    spec = importlib.util.spec_from_file_location("frozen027_state_audit_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCALED = frozen_helpers()
FLOW, OLD = SCALED.FLOW, SCALED.OLD
require, numeric, exact, close = SCALED.require, SCALED.numeric, SCALED.exact, SCALED.close


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def cache_only_hashes(mapping, forbidden, evidence):
    SCALED.cache_only_hashes(mapping, forbidden, evidence)


def source_closure(prepared, scaled, path):
    ordered = list(scaled["source_sha256"]) + [str(REPO / p) for p in NEW_SOURCES] + [str(CONFIG), str(PLAN)]
    require(
        len(set(ordered)) == len(ordered) == 96 and set(prepared["source_sha256"]) == set(ordered),
        "028 required96 source closure is incomplete or changed",
    )
    require(
        len(scaled["source_sha256"]) == 84
        and all(prepared["source_sha256"].get(k) == v for k, v in scaled["source_sha256"].items()),
        "028 changed one of84 frozen027 sources",
    )
    require(
        prepared["source_sha256"][str(CONFIG)] == CONFIG_SHA
        and prepared["source_sha256"][str(PLAN)] == PLAN_SHA
        and prepared["source_sha256"][str(HELPER)] == HELPER_SHA,
        "028 source registration/helper identity changed",
    )
    # JSON publication sorts mappings; snapshot numbering precedes that sort.
    snapshots = {
        str(Path(path).resolve().parent / "source_snapshot" / f"{index:03d}_{Path(filename).name}"): prepared[
            "source_sha256"
        ][filename]
        for index, filename in enumerate(ordered)
    }
    require(prepared["snapshot_sha256"] == snapshots, "028 required96 snapshot paths/digests differ from sources")


def read_model(ref, evidence):
    with np.load(evidence.check(ref["path"], ref["sha256"]), allow_pickle=False) as archive:
        require(set(archive.files) == {"weights", "center", "scale"}, "028 adapted model omitted frozen preprocessing")
        model = {k: numeric(archive[k], k).copy() for k in archive.files}
    require(
        all(v.dtype == np.float64 for v in model.values())
        and model["weights"].shape == (1281, 10)
        and model["center"].shape == (1280,)
        and model["scale"].shape == ()
        and model["scale"] > 0,
        "028 actual027 coefficient/preprocessing identity differs",
    )
    return model


def owned_exit(record, evidence, *, gpu):
    launch, exit_record, release = (evidence.ref(record[k]) for k in ("launch", "exit", "release"))
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="6" if gpu else "", LD_LIBRARY_PATH=""),
        "028 actual child CPU/GPU environment differs",
    )
    require(
        exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0),
        "028 owned child lacks actual normal wait",
    )
    require(
        all(release[k] == launch["child"][k] for k in ("pid", "start_ticks"))
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "028 owned child normal wait/reap identity differs",
    )
    return dict(pid=release["pid"], start_ticks=release["start_ticks"], reaped=True, signal_sent=False)


def verify_preparation(path, evidence):
    prepared, config = evidence.json(path), evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    evidence.check(HELPER, HELPER_SHA)
    require(
        prepared["schema_version"] == "s2a.scaled-state-preparation.v1"
        and prepared["config"] == config
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration"] == reference(PLAN),
        "028 preparation/config/plan differs",
    )
    require(
        prepared["output_root"] == str(REPO / config["output_dir"]) and prepared["budget"] == config["budget"],
        "028 prepared output/budget differs",
    )
    forbidden = SCALED.forbidden_observations(prepared)
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        cache_only_hashes(prepared[key], forbidden, evidence)
    ancestors = {}
    for suffix in ("prepared", "fit", "fit_verification", "final_peer", "owner"):
        expected = dict(
            path=str(REPO / config[f"source_scaled_{suffix}"]), sha256=config[f"source_scaled_{suffix}_sha256"]
        )
        require(prepared[f"scaled_{suffix}_ref"] == expected, "028 changed frozen027 ancestor ref")
        ancestors[suffix] = evidence.ref(expected)
    scaled, fit, verified, final, owner = (
        ancestors[k] for k in ("prepared", "fit", "fit_verification", "final_peer", "owner")
    )
    for ancestor in ancestors.values():
        if "input_sha256" in ancestor:
            cache_only_hashes(ancestor["input_sha256"], forbidden, evidence)
    for key in ("source_sha256", "snapshot_sha256"):
        cache_only_hashes(scaled[key], forbidden, evidence)
    source_closure(prepared, scaled, path)
    inherited = (
        "source_prepared_ref",
        "source_features_ref",
        "source_features_verification_ref",
        "source_failed_fit_verification_ref",
        "flow_prepared_ref",
        "flow_preparation_verification_ref",
        "flow_fit_ref",
        "flow_fit_verification_ref",
        "flow_final_peer_ref",
        "reused_bias_ref",
        "reused_bias_predictions_ref",
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
    require(all(prepared[k] == scaled[k] for k in inherited), "028 changed original source/roster/control interface")
    for key in (
        "seed",
        "cosmos_repo",
        "checkpoint",
        "domain",
        "train_episodes",
        "validation_episodes",
        "anchors_per_episode",
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
        "parameterization",
        "residual_precision",
    ):
        require(config[key] == scaled["config"][key], "028 changed original clock/domain/loss/gate")
    old = evidence.ref(prepared["source_prepared_ref"])
    old_verified = evidence.ref(prepared["source_features_verification_ref"])
    FLOW.inherited_rosters(prepared, old, config, old_verified["input_sha256"])
    require(len(prepared["weight_files"]) == 163, "028 original8400 weight roster differs")
    for filename, saved in prepared["weight_files"].items():
        stat = Path(filename).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (saved["bytes"], saved["mtime_ns"]),
            "028 frozen8400 weight stat changed",
        )
    require(
        fit["schema_version"] == "s2a.scaled-readout-fit.v1"
        and fit["complete"] is True
        and fit["passed"] is True
        and fit["prepared_ref"] == prepared["scaled_prepared_ref"]
        and fit["candidate_fits"] == fit["fit_attempts"] == fit["producer_fit_groups"] == 1
        and fit["reused_controls"] == 1,
        "028 original027 single-fit evidence differs",
    )
    require(
        verified["passed"] is True
        and verified["phase"] == "fit"
        and verified["fit_verified"] is True
        and verified["independent_candidate_reconstructions"] == 1
        and verified["reused_controls_verified"] == 1
        and verified["prepared"] == prepared["scaled_prepared_ref"]
        and verified["input_sha256"].get(prepared["scaled_fit_ref"]["path"]) == prepared["scaled_fit_ref"]["sha256"],
        "028 lacks complete independent027 parameter admission",
    )
    OLD.compare_metrics(fit["teacher_metrics"], verified["teacher_metrics"])
    gate = OLD.independent_teacher_gate(verified["teacher_metrics"])
    require(
        gate["passed"] is True
        and fit["teacher_gate"] == verified["teacher_gate"] == gate
        and fit["eligible_for_runtime_qualification"] is verified["eligible_for_runtime_qualification"] is True
        and fit["eligible_for_sampling"] is verified["eligible_for_sampling"] is False,
        "028 changed original027 teacher decision or falsely upgraded027 sampling",
    )
    require(
        final["passed"] is True
        and final["fit"] == prepared["scaled_fit_ref"]
        and final["fit_verification"] == prepared["scaled_fit_verification_ref"]
        and final["eligible_for_runtime_qualification"] is True
        and final["eligible_for_sampling"] is False
        and final["reused026_full128_raw_and_bias_prediction_bits_exact"] is True,
        "028 original027 final peer differs",
    )
    require(
        owner["complete"] is True
        and owner["passed"] is True
        and owner["automatic_retry"] is False
        and owner["producer_result"] == prepared["scaled_fit_ref"]
        and owner["prepared_ref"] == prepared["scaled_prepared_ref"]
        and owner["eligible_for_runtime_qualification"] is True
        and owner["eligible_for_sampling"] is False
        and len(owner["owned_commands"]) == 2,
        "028 original027 CPU owner evidence differs",
    )
    for record in owner["owned_commands"]:
        owned_exit(record, evidence, gpu=False)
    require(prepared["adapted_model_ref"] == fit["models"]["adapt"], "028 replaced the qualified027 candidate")
    require(
        prepared["scaled_teacher_predictions_ref"] == fit["teacher_predictions"],
        "028 replaced the original027 teacher artifact",
    )
    model = read_model(prepared["adapted_model_ref"], evidence)
    require(
        all(
            prepared[k] == 0
            for k in (
                "new_model_calls",
                "new_teacher_calls",
                "new_episode_reads",
                "new_png_reads",
                "fit_count",
                "independent_parameter_reconstructions",
                "physics_steps",
                "fresh_performance_cases",
            )
        )
        and prepared["full_weight_bytes_rehashed"] is False,
        "028 preparation used an undeclared execution budget",
    )
    admission = prepared["source_admission"]
    require(
        admission["original_source_count"] == 84
        and admission["original_actual_input_count"] == 874
        and admission["original_weight_stat_count"] == 163
        and admission["original027_teacher_passed"] is True
        and admission["original027_runtime_qualification_eligible"] is True
        and admission["original027_eligible_for_sampling"] is False
        and admission["inherited_episode_refs_not_reopened"] is True
        and admission["inherited_png_refs_not_reopened"] is True
        and admission["preprocessing_statistics_recomputed"] is False
        and admission["new_precision_required"] is True
        and admission["historical_initial_physical_sync_certified"] is False
        and admission["pretraining_visibility"] == "unknown",
        "028 altered historical/source-scope or runtime qualification evidence",
    )
    return prepared, model


def public_lifecycle(actual, prepared, stage, evidence):
    raw = evidence.json(Path(prepared["output_root"]) / stage / "worker_result.json")
    require(all(actual.get(k) == v for k, v in raw.items()), "028 owner changed actual worker evidence")
    directory = Path(prepared["output_root"]) / "orchestration" / stage
    record = dict(
        launch=reference(directory / "model/launch.json"), exit=actual["exit_ref"], release=actual["release_ref"]
    )
    identity = owned_exit(record, evidence, gpu=True)
    gpu = evidence.ref(actual["gpu_after_ref"])
    require(
        gpu["physical_gpu"] == 6
        and gpu["uuid"] == GPU_UUID
        and gpu["released"] is True
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == [],
        "028 GPU6 release is unproven before independent state scoring",
    )
    path = actual["resource_observations_ref"]
    observations = [json.loads(line) for line in evidence.check(path["path"], path["sha256"]).read_text().splitlines()]
    require(
        observations
        and all(o["physical_gpu"] == 6 and o["uuid"] == GPU_UUID for o in observations)
        and all(
            observations[i]["seconds_since_actual_owned_wait"] <= observations[i + 1]["seconds_since_actual_owned_wait"]
            for i in range(len(observations) - 1)
        ),
        "028 release observations are missing or unordered",
    )
    require(
        0
        <= observations[0]["seconds_since_actual_owned_wait"]
        <= observations[-1]["seconds_since_actual_owned_wait"]
        <= 120
        and all(gpu[k] == v for k, v in observations[-1].items()),
        "028 release was not observed within recorded120seconds",
    )
    return identity


def precision_cases(model, sigmas):
    """Independent registered fixtures; never import the producer probe."""
    half, epsilon, spacing = 2**-8, 2**-30, 2**-133
    positive = [
        half - epsilon,
        half,
        half + epsilon,
        -half + epsilon,
        -half,
        -half - epsilon,
        3 * half - epsilon,
        3 * half,
        3 * half + epsilon,
        0.0,
    ]
    tiny = [
        spacing / 2 - 2**-160,
        spacing / 2,
        spacing / 2 + 2**-160,
        -spacing / 2 + 2**-160,
        -spacing / 2,
        -spacing / 2 - 2**-160,
        1.5 * spacing,
        2.5 * spacing,
        2**-126 - spacing / 2,
        -0.0,
    ]
    for index, sigma in enumerate(sigmas):
        for name, scalar, residual in (
            ("positive", 1.0, positive),
            ("negative", -1.0, -np.asarray(positive)),
            ("subnormal", 0.0, tiny),
        ):
            hidden = np.zeros((2, 1280), np.float32)
            hidden[1, :2] = [0.5, -0.25]
            base = np.full((2, 128), scalar, np.float32)
            base[:, 10:] = np.arange(118)
            base[:, 10] = -0.0
            weights = np.zeros((1281, 10), np.float64)
            weights[-1] = -np.asarray(residual, np.float64) * sigma
            weights[0, 0], weights[1, 1] = 2**-20, -(2**-21)
            center = np.zeros(1280, np.float64)
            center[:2] = [0.5, -0.25]
            yield dict(
                name=f"sigma{index}_{name}",
                hidden=hidden,
                base=base,
                weights=weights,
                center=center,
                scale=np.asarray(3.0, np.float64),
                sigma_actual=np.full((2, 1), sigma, np.float64),
            )
    rng = np.random.default_rng(42)
    for index, sigma in enumerate(sigmas):
        yield dict(
            name=f"sigma{index}_candidate",
            hidden=rng.integers(-32, 33, (2, 1280)).astype(np.float32) / 128,
            base=rng.integers(-16, 17, (2, 128)).astype(np.float32) / 16,
            weights=model["weights"],
            center=model["center"],
            scale=model["scale"],
            sigma_actual=np.full((2, 1), sigma, np.float64),
        )


def precision_arrays(arrays, cases, model, sigmas):
    expected = list(precision_cases(model, sigmas))
    names = [case["name"] for case in expected]
    require(
        [case["name"] for case in cases] == names and len(names) == 20,
        "028 precision coverage is not fifteen synthetic plus five actual candidate cases",
    )
    keys = ("hidden", "base", "weights", "center", "scale", "sigma_actual")
    required = {f"{name}_{key}" for name in names for key in (*keys, "cpu_bits", "cuda_bits")}
    require(
        set(arrays) == required | {"primitive_values", "primitive_cpu_bits", "primitive_cuda_bits"},
        "028 precision tensor roster changed",
    )
    for fixture, record in zip(expected, cases, strict=True):
        name = fixture["name"]
        require(
            set(record)
            == {
                "name",
                "cpu_cuda_bitwise_equal",
                "actual_hook_matches_apply",
                "zero_original_object",
                "other_domain_original_object",
                "other118_preserved",
                "input_tensors_unchanged",
            }
            and all(v is True for k, v in record.items() if k != "name"),
            "028 actual synthetic hook/bypass/input evidence missing",
        )
        for key in keys:
            actual, value = arrays[f"{name}_{key}"], fixture[key]
            require(actual.dtype == value.dtype, f"028 precision {key} dtype differs")
            bit_type = np.uint32 if value.dtype == np.float32 else np.uint64
            exact(actual.view(bit_type), value.view(bit_type), f"028 registered {name} {key} input bits")
        output = SCALED.quantized_velocity(
            fixture["hidden"],
            fixture["base"],
            fixture["sigma_actual"],
            fixture["weights"],
            fixture["center"],
            fixture["scale"],
        )
        bits = (output.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
        for device in ("cpu", "cuda"):
            actual = arrays[f"{name}_{device}_bits"]
            require(actual.dtype == np.int16, "028 actual BF16 bits must be int16")
            exact(actual, bits, f"028 independent {name} {device} full128 BF16 bits")
    primitive = np.asarray(
        [
            0.0,
            -0.0,
            2**-134 - 2**-160,
            2**-134,
            2**-134 + 2**-160,
            -(2**-134) + 2**-160,
            -(2**-134),
            -(2**-134) - 2**-160,
            1 + 2**-8 + 2**-30,
            -1 - 2**-8 - 2**-30,
        ],
        np.float64,
    )
    require(arrays["primitive_values"].dtype == np.float64, "028 primitive FP64 input changed")
    exact(arrays["primitive_values"].view(np.uint64), primitive.view(np.uint64), "028 signed primitive input bits")
    for device in ("cpu", "cuda"):
        require(arrays[f"primitive_{device}_bits"].dtype == np.int16, "028 primitive output is not BF16 bits")
        exact(arrays[f"primitive_{device}_bits"], FLOW.bf16_bits(primitive), f"028 independent {device} primitive RNE")


def verify_precision(prepared, model, path, evidence):
    actual = evidence.json(path)
    require(
        actual["schema_version"] == "s2a.scaled-readout-synthetic-cpu-cuda-precision.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual
        and actual["prepared_ref"] == reference(Path(prepared["output_root"]) / "preparation/prepared.json"),
        "partial/rejected028 precision or preparation mismatch",
    )
    require(
        actual["candidate_model_ref"] == prepared["adapted_model_ref"] and actual["candidate_models_loaded"] == 1,
        "028 precision did not exercise actual027 candidate",
    )
    require(
        actual["seed"] == 42
        and actual["physical_gpu"] == 6
        and actual["cuda_visible_devices"] == "6"
        and actual["actual_cuda_index"] == 0
        and actual["actual_gpu_uuid"].removeprefix("GPU-").lower() == GPU_UUID.removeprefix("GPU-").lower()
        and actual["cpu_cuda_bitwise_equal"] is True,
        "028 actual CPU/CUDA physicalGPU6 identity differs",
    )
    require(
        all(
            actual[k] == 0
            for k in ("model_weights_loaded", "wam_denoise_calls", "fit_groups", "episode_reads", "physics_steps")
        ),
        "028 precision used an undeclared WAM/fit/EP/physics budget",
    )
    core, worker = str(REPO / "src/s2a/wam_scaled_readout.py"), actual["worker_source"]
    require(
        actual["core_sha256"] == prepared["source_sha256"][core]
        and worker["path"] == str(REPO / "scripts/probe_wam_scaled_readout_quantization.py")
        and worker["sha256"] == prepared["source_sha256"][worker["path"]],
        "028 precision used an unfrozen core/probe",
    )
    evidence.check(core, actual["core_sha256"])
    evidence.check(worker["path"], worker["sha256"])
    sigmas = np.asarray(prepared["config"]["precision_actual_sigmas"], np.float64)
    exact(
        sigmas,
        np.asarray([0.2, 0.5, 0.8, 0.12809006869792938, 0.9990000128746033], np.float32).astype(np.float64),
        "028 registered actual FP32 sigma values",
    )
    with np.load(
        evidence.check(actual["actual_outputs"]["path"], actual["actual_outputs"]["sha256"]), allow_pickle=False
    ) as archive:
        precision_arrays(
            {key: numeric(archive[key], key).copy() for key in archive.files}, actual["cases"], model, sigmas
        )
    public_lifecycle(actual, prepared, "precision", evidence)
    return dict(
        precision_verified=True,
        precision=reference(path),
        cases_verified=20,
        primitive_values_verified=10,
        actual_candidate_cases_verified=5,
        actual_cpu_cuda_full128_bits_verified=True,
        owned_normal_exit_and_release_verified=True,
        eligible_for_sampling=True,
        original027_eligible_for_sampling=False,
    )


def precision_admission(prepared, model, path, verification_path, evidence):
    actual, verified = evidence.json(path), evidence.json(verification_path)
    require(
        actual["complete"] is True
        and actual["passed"] is True
        and actual["candidate_model_ref"] == prepared["adapted_model_ref"]
        and verified["passed"] is True
        and verified["phase"] == "precision"
        and verified["precision_verified"] is True
        and verified["eligible_for_sampling"] is True
        and verified["original027_eligible_for_sampling"] is False
        and verified["cases_verified"] == 20
        and verified["actual_candidate_cases_verified"] == 5,
        "028 actual precision admission is absent or rejected",
    )
    prepared_ref = reference(Path(prepared["output_root"]) / "preparation/prepared.json")
    require(
        actual["prepared_ref"] == verified["prepared"] == prepared_ref
        and verified["precision"] == reference(path)
        and verified["input_sha256"].get(str(Path(path).resolve())) == sha256(path),
        "028 precision receipt is bound to another candidate/preparation",
    )
    cache_only_hashes(verified["input_sha256"], SCALED.forbidden_observations(prepared), evidence)
    owner = evidence.json(Path(prepared["output_root"]) / "orchestration/precision/result.json")
    require(
        owner["complete"] is True
        and owner["passed"] is True
        and owner["stage"] == "precision"
        and owner["automatic_retry"] is False
        and owner["producer_result"] == reference(path)
        and len(owner["owned_commands"]) == 2,
        "028 prior precision owner did not complete its GPU worker and CPU independent audit",
    )
    for index, record in enumerate(owner["owned_commands"]):
        owned_exit(record, evidence, gpu=index == 0)
    return actual


def sampling_execution_contract(actual, prepared):
    require(
        actual["schema_version"] == "s2a.scaled-state-sampling.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual,
        "partial/rejected028 sampling",
    )
    require(
        actual["fit_ref"] == prepared["scaled_fit_ref"]
        and actual["fit_verification_ref"] == prepared["scaled_fit_verification_ref"]
        and actual["adapted_model_ref"] == prepared["adapted_model_ref"],
        "028 sampling replaced the frozen027 candidate/admission",
    )
    require(
        actual["all_original_parameters_frozen"] is True
        and actual["original_parameters_require_grad_false"] is True
        and actual["head_call_counts"] == dict(zero=105, raw=1680, adapt=1680),
        "028 original model/head freeze contract differs",
    )
    require(
        actual["physics_steps"]
        == actual["fresh_cases"]
        == actual["new_teacher_calls"]
        == actual["new_candidate_fits"]
        == actual["new_episode_reads"]
        == 0
        and actual["wam_model_initializations"] == 1,
        "028 sampling exceeded its fixed budget",
    )
    require(
        actual["denoise_calls"] == 3465 and len(actual["zero_identity"]) == 3 and len(actual["generations"]) == 96,
        "028 sampling fixed99/3465 budget is incomplete",
    )
    require(
        actual["runtime"]["checkpoint_path"] == prepared["checkpoint"] and actual["runtime"]["ema_weights"] is False,
        "028 sampled a different WAM checkpoint",
    )
    labels = [f"zero:{i}" for i in range(3)] + [f"sampling:{i}" for i in range(96)]
    expected = [
        dict(index=i * 35 + j, label=label, completed=True) for i, label in enumerate(labels) for j in range(35)
    ]
    require(actual["denoise_ledger"] == expected, "028 actual denoise ledger is partial/repeated/out of order")


def sampling_records(prepared, actual, evidence):
    source = evidence.ref(prepared["source_prepared_ref"])
    config_path = str(Path(prepared["config"]["checkpoint"]["run"]) / "config.yaml")
    spec = OLD.yaml.safe_load(evidence.check(config_path, source["input_sha256"][config_path]).read_text())["model"][
        "config"
    ]["action128_spec"]
    require(
        spec["registry"]["domains"].index(OLD.DOMAIN) == 31 and spec["registry"]["domain_projection_ids"][31] == 0,
        "028 normalization31/projection0 registry changed",
    )
    transform = spec["transforms"][31]
    for planned, recorded in zip(prepared["zero_identity_cases"], actual["zero_identity"], strict=True):
        require(all(recorded[k] == v for k, v in planned.items()), "028 three zero roster/source changed")
        evidence.check(planned["sample"]["image_path"], planned["sample"]["image_sha256"])
        response, _, metadata, _ = FLOW.condition_signature(
            recorded, planned, planned["request"]["prompt"], prepared, transform, evidence
        )
        entry = dict(
            name=planned["name"],
            generation_index=-planned["task_index"] - 1,
            pair_id=f"zero{planned['task_index']}",
            variant="zero",
            sampling_seed=42,
        )
        OLD.initial_record_identity(metadata, entry)
        cached = evidence.json(planned["cache"]["response_path"], planned["cache"]["response_sha256"])
        current, prior = response["outputs"][0]["content"], cached["outputs"][0]["content"]
        exact(current["action"], prior["action"], "028 zero full128 raw identity")
        exact(
            current["action128"]["normalized_action"],
            prior["action128"]["normalized_action"],
            "028 zero full128 normalized identity",
        )
    anchors = {a["name"]: a for a in prepared["anchors"]}
    pairs, generated = {}, []
    for planned, recorded in zip(prepared["validation_generations"], actual["generations"], strict=True):
        require(all(recorded[k] == v for k, v in planned.items()), "028 paired generation roster/source changed")
        anchor = anchors[planned["anchor_name"]]
        require(anchor["split"] == "validation", "028 training source entered validation generation")
        evidence.check(anchor["image_path"], anchor["image_sha256"])
        _, actions, metadata, signature = FLOW.condition_signature(
            recorded, planned, anchor["task"], prepared, transform, evidence
        )
        OLD.initial_record_identity(metadata, planned)
        pair_id = planned["pair_id"]
        if planned["variant"] == "raw":
            require(pair_id not in pairs, "028 duplicate raw generation pair")
            pairs[pair_id] = signature
        else:
            require(pair_id in pairs, "028 adapted generation lacks raw pair")
            FLOW.require_same_initial(signature, pairs[pair_id])
        generated.append((planned, anchor, actions))
    require(len(pairs) == 48, "028 sampling lacks48 exact initial pairs")
    return generated


def score_states(prepared, generated, evidence):
    labels, integrate, rows = FLOW.state_labels(prepared, evidence), OLD.frozen_integrator(evidence), []
    for planned, anchor, actions in generated:
        count, errors, _ = integrate(labels[anchor["episode"]], anchor, actions)
        require(count == 40, "028 state scoring lost common H40 endpoints")
        rows.append(
            dict(
                name=planned["name"],
                episode=anchor["episode"],
                task_index=anchor["task_index"],
                sampling_seed=planned["sampling_seed"],
                variant=planned["variant"],
                errors=errors,
            )
        )
    groups = dict(pooled=OLD.pool_state_rows(rows))
    for label, field in (("by_episode", "episode"), ("by_task", "task_index"), ("by_seed", "sampling_seed")):
        groups[label] = {
            str(value): OLD.pool_state_rows([r for r in rows if r[field] == value])
            for value in sorted({r[field] for r in rows})
        }
    require(
        len(groups["by_episode"]) == 6 and set(groups["by_seed"]) == {"42", "43"},
        "028 state scoring episode/seed coverage differs",
    )
    return groups, OLD.independent_state_gate(groups)


def verify_sampling(prepared, model, precision_path, precision_verification_path, path, evidence):
    actual = evidence.json(path)
    require(
        actual["prepared_ref"] == reference(Path(prepared["output_root"]) / "preparation/prepared.json"),
        "028 sampling used another preparation",
    )
    sampling_execution_contract(actual, prepared)
    require(
        actual["precision_ref"] == reference(precision_path)
        and actual["precision_verification_ref"] == reference(precision_verification_path),
        "028 sampling precision admission refs differ",
    )
    precision_admission(prepared, model, precision_path, precision_verification_path, evidence)
    public_lifecycle(actual, prepared, "sampling", evidence)
    generated = sampling_records(prepared, actual, evidence)
    # All99 outputs, identities, original sources, actual solver traces and the
    # model's normal exit/resource release precede opening any true future GT.
    groups, gate = score_states(prepared, generated, evidence)
    return dict(
        sampling_verified=True,
        zero_identity_generations_verified=3,
        generations_verified=96,
        actual_denoise_calls_verified=3465,
        paired_initial_states_verified=48,
        actual_solver_sigma_traces_verified=99,
        validation_episode_states_read=6,
        sampling_state_metrics=groups,
        sampling_gate=gate,
        eligible_for_development_physics_registration=gate["passed"],
        physics_authorized=False,
        physics_tasks_executed=0,
        raw_checkpoint_task_success_proven=False,
        adapted_task_capability_proven=False,
    )


def phase_arguments(args):
    present = (args.precision is not None, args.precision_verification is not None, args.sampling is not None)
    require(
        present
        == dict(prepare=(False, False, False), precision=(True, False, False), sampling=(True, True, True))[args.phase],
        "028 evidence arguments do not match explicit phase",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "precision", "sampling"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--precision", type=Path)
    parser.add_argument("--precision-verification", type=Path)
    parser.add_argument("--sampling", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "028 verifier requires emptyCUDA/LD and seed42",
    )
    phase_arguments(args)
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "028 receipt must be new under outputs; refuse overwrite",
    )
    evidence = OLD.Evidence()
    receipt = dict(
        schema_version="s2a.scaled-state-independent-verification.v1",
        phase=args.phase,
        passed=False,
        new_model_calls=0,
        new_candidate_fits=0,
        independent_candidate_reconstructions=0,
        new_physics_steps=0,
    )
    error = None
    try:
        prepared, model = verify_preparation(args.prepared, evidence)
        receipt["prepared"] = reference(args.prepared)
        if args.phase == "prepare":
            receipt.update(
                preparation_verified=True,
                old027_teacher_gate=True,
                old027_eligible_for_sampling=False,
                eligible_for_sampling=False,
            )
        elif args.phase == "precision":
            receipt.update(verify_precision(prepared, model, args.precision, evidence))
        else:
            receipt.update(
                verify_sampling(prepared, model, args.precision, args.precision_verification, args.sampling, evidence)
            )
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
        raise SystemExit(f"028 independent audit rejected: {error}")


if __name__ == "__main__":
    main()
