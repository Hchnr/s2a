"""026 independent cached-feature, flow-design, quantization and gate audit.

Only frozen025 independent evidence/NumPy helpers are shared. No production
readout/fit, Torch, Cosmos, dataloader or episode tensor reader is imported.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO

CONFIG = REPO / "configs/wam_flow_consistent_readout_20261021.json"
CONFIG_SHA = "2d0e45b3ae02ecec4ee45bcaa1928596e347417f945f8903416f0df314a68534"
PLAN = REPO / "plans/v0.0.1/026_flow_consistent_readout_20261021.md"
PLAN_SHA = "42c0435773306852d4aa79be4d319f9de9cc71ff7823b7ad830286c2c41fc2fa"
HELPER = REPO / "scripts/verify_wam_readout.py"
HELPER_SHA = "50e486d82bc7ea543c7e973c3c7d9d1ce469ea49bdab12947c135484b2fd1347"


def frozen_helpers():
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("frozen025 independent helper changed")
    spec = importlib.util.spec_from_file_location("s2a_frozen025_flow_audit_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = frozen_helpers()
require, numeric, exact, close = OLD.require, OLD.numeric, OLD.exact, OLD.close


def sigmas64(values, count):
    result = numeric(values, "actual sigma").astype(np.float64)
    if result.ndim == 0:
        result = np.full(count, result.item(), dtype=np.float64)
    if result.shape == (count, 1):
        result = result[:, 0]
    require(result.shape == (count,) and np.all(result > 0), "actual sigma must be finite and positive for every row")
    return result


def flow_design(hidden, sigma_actual):
    """Build -phi/sigma before matrix multiplication, in FP64."""
    design = OLD.phi(hidden)
    return -design / sigmas64(sigma_actual, len(design))[:, None]


def augmented_fit(hidden, sigma_actual, target, weights):
    design = flow_design(hidden, sigma_actual)
    target, weights = (
        numeric(target, "velocity residual").astype(np.float64),
        numeric(weights, "weights").astype(np.float64),
    )
    require(
        target.shape == (len(design), 10) and weights.shape == (len(design),), "flow fit row/active10 dimensions differ"
    )
    require(np.all(weights > 0) and abs(weights.sum() - 1) <= 1e-12, "weights must be positive and sum to one")
    x = np.concatenate((np.sqrt(weights)[:, None] * design, np.sqrt(0.1) * np.eye(design.shape[1])))
    y = np.concatenate((np.sqrt(weights)[:, None] * target, np.zeros((design.shape[1], 10))))
    adapt = np.linalg.lstsq(x, y, rcond=None)[0]
    bias = np.zeros_like(adapt)
    column = design[:, -1]
    bias[-1] = ((weights * column) @ target) / (np.dot(weights, column**2) + 0.1)
    return dict(adapt=adapt, bias=bias)


def quantized_velocity(hidden, base, sigma_actual, coefficients):
    hidden, base, coefficients = (
        numeric(x, name) for x, name in ((hidden, "hidden"), (base, "base"), (coefficients, "coefficients"))
    )
    design = flow_design(hidden, sigma_actual)
    require(
        base.shape == (len(hidden), 128) and coefficients.shape == (design.shape[1], 10),
        "flow deployment tensor dimensions differ",
    )
    if not np.any(coefficients):
        return base
    result = base.copy()
    result[:, :10] = OLD.quantize_bfloat16(base[:, :10].astype(np.float64) + design @ coefficients)
    return result


def inherited_rosters(prepared, source, config, admitted_inputs):
    """Check frozen source identities without opening inherited EP/PNG bytes."""
    for key in (
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
    ):
        require(prepared[key] == source[key], f"026 changed original025 {key}")
    episodes, anchors = source["episodes"], source["anchors"]
    require(
        len(episodes) == 12 and len(anchors) == 48 and len({a["name"] for a in anchors}) == 48,
        "old development roster differs",
    )
    require(
        [e["episode"] for e in episodes] == config["train_episodes"] + config["validation_episodes"],
        "026 changed whole-episode order",
    )
    require(
        len({e["source_episode_id"] for e in episodes}) == 12
        and not set(config["train_episodes"]) & set(config["validation_episodes"]),
        "026 validation entered training split",
    )
    for index, episode in enumerate(episodes):
        split = "train" if index < 6 else "validation"
        members = [a for a in anchors if a["episode"] == episode["episode"]]
        require(episode["split"] == split and len(members) == 4, "026 whole-episode split changed")
        frames = episode["frames"]
        require(
            len(frames) == len(set(frames)) == 4 and frames == [j * frames[-1] // 3 for j in range(4)],
            "026 inherited four uniform anchors differ",
        )
        require(
            [a["anchor_frame"] for a in members] == frames
            and [a["name"] for a in members] == episode["anchors"]
            and [a["anchor_index"] for a in members] == list(range(4)),
            "026 anchor order differs",
        )
        for anchor in members:
            require(
                anchor["split"] == split
                and anchor["source_episode_id"] == episode["source_episode_id"]
                and anchor["task_index"] == episode["task_index"]
                and anchor["valid_future_steps"] == 40
                and anchor["state_mask128"] == OLD.ACTIVE.tolist(),
                "026 inherited source/mask differs",
            )
            require(
                anchor["original_split"] == ("diagnostic" if split == "train" else "study_heldout"),
                "026 split provenance changed",
            )
            require(
                source["input_sha256"].get(anchor["arrays_path"]) == anchor["arrays_sha256"]
                and admitted_inputs.get(anchor["image_path"]) == anchor["image_sha256"],
                "026 inherited EP/PNG ref lacks original admission",
            )
            state = numeric(anchor["measured_state128"], "inherited measured state")
            require(state.shape == (128,), "026 inherited current state shape differs")
    expected = [
        dict(call_index=i * 3 + j, anchor_name=a["name"], split=a["split"], sigma=s, noise_seed=42)
        for i, a in enumerate(anchors)
        for j, s in enumerate((0.2, 0.5, 0.8))
    ]
    require(prepared["teacher_calls"] == expected, "026 original144 teacher roster differs")
    require(
        prepared["validation_generations"] == OLD.validation_manifest(anchors), "026 fixed paired sampling grid changed"
    )
    require(
        len(source["zero_identity_cases"]) == 3
        and [v["task_index"] for v in source["zero_identity_cases"]] == [0, 1, 2],
        "026 original three zero roster differs",
    )


def cached_contract(arrays, metadata, call):
    require(
        arrays["sigma_action"].shape == (40, 1) and arrays["sigma_action"].dtype == np.float32,
        "026 teacher sigma must originate from stored FP32",
    )
    OLD.check_feature_arrays(arrays, metadata, arrays["x0_normalized"], call["sigma"])
    exact(
        sigmas64(arrays["sigma_action"], 40),
        np.full(40, float(np.float32(call["sigma"]))),
        "026 actual cached sigma provenance",
    )


def original_resource_admission(prepared, source, evidence):
    resource = prepared["source_admission"]["resource"]
    require(
        resource["original_owner_passed"] is False
        and resource["original_owner_tool_exit"] == 1
        and resource["original_model_actual_wait_exit"] == 0
        and resource["original_model_reaped"] is True
        and resource["original_model_signal_sent"] is False,
        "026 erased original025 owner failure or model exit",
    )
    require(
        len(resource["original_births"]) == 2
        and all(v["original_birth_terminal"] is True for v in resource["original_births"]),
        "026 original owned births were not terminal",
    )
    require(
        resource["historical_poll_detail_persisted"] is False
        and resource["historical_release_timeout_root_cause"] == "unknown",
        "026 upgraded missing historical release detail",
    )
    peer = evidence.ref(resource["independent_release_ref"])
    launch = evidence.ref(resource["actual_launch_proof_ref"])
    tool = evidence.ref(resource["tool_exit_ref"])
    require(
        peer["passed"] is True
        and peer["current_resource_release_verified"] is True
        and peer["original_owner_passed"] is False
        and tool["tool_handle_terminal"] is True
        and tool["tool_reported_exit_code"] == 1
        and launch["passed"] is True,
        "026 original025 resource proofs differ",
    )
    root = Path(source["output_root"]) / "orchestration/features"
    owner, start, end, release = (
        evidence.json(root / name)
        for name in ("result.json", "model/launch.json", "model/exit.json", "model/ownership/release.json")
    )
    require(
        owner["complete"] is False
        and owner["passed"] is False
        and tool["original_owner_result"]
        == dict(path=str(root / "result.json"), sha256=OLD.digest(root / "result.json")),
        "026 original owner False changed",
    )
    require(
        end == dict(actual_wait=True, automatic_retry=False, returncode=0)
        and all(start["child"][k] == release[k] for k in ("pid", "start_ticks"))
        and release["returncode"] == 0
        and release["reaped"] is True
        and release["signal_sent"] is False
        and release["original_error"] is None,
        "026 original model wait/reap provenance differs",
    )
    current = resource["current_resource"]
    require(
        current["physical_gpu"] == 6 and current["memory_mib"] == 0 and current["compute_apps"] == [],
        "026 saved GPU6 release admission differs",
    )


def verify_preparation(path, evidence):
    prepared = evidence.json(path)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    evidence.check(HELPER, HELPER_SHA)
    require(
        prepared["schema_version"] == "s2a.flow-readout-preparation.v1" and prepared["config"] == config,
        "wrong026 preparation/config",
    )
    require(
        prepared["config_ref"] == dict(path=str(CONFIG), sha256=CONFIG_SHA)
        and prepared["registration"] == dict(path=str(PLAN), sha256=PLAN_SHA),
        "026 registration identity differs",
    )
    require(
        prepared["output_root"] == str(REPO / config["output_dir"]) and prepared["budget"] == config["budget"],
        "026 output/budget differs",
    )
    inherited = {a[key] for a in prepared["anchors"] for key in ("arrays_path", "image_path")}
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        require(prepared[key], f"empty026 {key}")
        require(not inherited & set(prepared[key]), "026 preparation reopened inherited episode or PNG files")
        for source, expected in prepared[key].items():
            evidence.check(source, expected)
    references = {}
    for name in ("prepared", "features", "features_verification", "failed_fit_verification"):
        ref = dict(path=str(REPO / config[f"source_{name}"]), sha256=config[f"source_{name}_sha256"])
        require(prepared[f"source_{name}_ref"] == ref, f"026 source {name} reference changed")
        references[name] = evidence.ref(ref)
    source, features = references["prepared"], references["features"]
    verified, failed = references["features_verification"], references["failed_fit_verification"]
    require(
        verified["passed"] is True
        and verified["phase"] == "features"
        and verified["teacher_calls_verified"] == 144
        and verified["actual_denoise_calls_verified"] == 249,
        "025 complete original-feature admission missing",
    )
    require(
        failed["passed"] is True
        and failed["phase"] == "fit"
        and failed["fit_verified"] is True
        and failed["eligible_for_sampling"] is False
        and failed["teacher_gate"]["passed"] is False,
        "025 original failed candidate evidence changed",
    )
    require(
        features["complete"] is True
        and features["passed"] is True
        and len(features["teacher_calls"]) == 144
        and features["denoise_calls"] == 249
        and features["all_original_parameters_frozen"] is True,
        "025 original feature cache is partial or changed",
    )
    # The old maps include raw EP/PNG files. Inspect their admitted references;
    # do not reopen them during preparation or cached teacher verification.
    for proof in (verified, failed):
        for name in ("prepared", "features"):
            ref = prepared[f"source_{name}_ref"]
            require(
                proof["input_sha256"].get(ref["path"]) == ref["sha256"],
                "025 independent proof did not bind the original inputs",
            )
    require(
        verified["prepared"] == failed["prepared"] == features["prepared_ref"] == prepared["source_prepared_ref"],
        "old025 inputs do not share one immutable preparation",
    )
    inherited_rosters(prepared, source, config, verified["input_sha256"])
    for key in ("train_episodes", "validation_episodes"):
        require(config[key] == source["config"][key] and len(config[key]) == 6, "026 changed whole-episode split")
    require(
        source["domain"]["name"] == OLD.DOMAIN
        and source["domain"]["normalization_id"] == 31
        and source["domain"]["projection_id"] == 0,
        "026 full-domain identity differs",
    )
    require(source["domain"]["active_mask"] == OLD.ACTIVE.tolist(), "026 active10 mask changed")
    inherited = {a[key] for a in source["anchors"] for key in ("arrays_path", "image_path")}
    require(not inherited & set(prepared["input_sha256"]), "026 preparation reopened inherited episode or PNG files")
    admission = prepared["source_admission"]
    require(
        admission["original_source_count"] == 53
        and admission["original_weight_stat_count"] == 163
        and admission["old_fit_eligible_for_sampling"] is False
        and admission["verified_teacher_cache_count"] == 144
        and admission["inherited_episode_refs_not_reopened"] is True
        and admission["inherited_png_refs_not_reopened"] is True
        and admission["ancestor_input_maps_rehashed_in_full"] is False
        and admission["historical_initial_physical_sync_certified"] is False
        and admission["pretraining_visibility"] == "unknown",
        "026 scope/source admission differs",
    )
    require(
        all(
            prepared[k] == 0
            for k in (
                "new_model_calls",
                "new_teacher_calls",
                "new_episode_reads",
                "fit_count",
                "physics_steps",
                "fresh_performance_cases",
            )
        )
        and prepared["full_weight_bytes_rehashed"] is False
        and prepared["quantization_admission_required_after_preparation"] is True,
        "026 preparation used an undeclared execution budget",
    )
    require(
        len(source["source_sha256"]) == 53
        and all(prepared["source_sha256"].get(k) == v for k, v in source["source_sha256"].items()),
        "026 did not preserve all53 original frozen sources",
    )
    for source_path, expected in source["snapshot_sha256"].items():
        evidence.check(source_path, expected)
    old_fit_path = Path(source["output_root"]) / "fit/fit_result.json"
    old_fit = evidence.json(old_fit_path, failed["input_sha256"][str(old_fit_path)])
    require(
        old_fit["complete"] is True
        and old_fit["passed"] is True
        and old_fit["eligible_for_sampling"] is False
        and old_fit["teacher_gate"]["passed"] is False,
        "026 changed actual025 failed fit",
    )
    original_resource_admission(prepared, source, evidence)
    require(len(prepared["weight_files"]) == 163, "original8400 file roster differs")
    for filename, saved in prepared["weight_files"].items():
        stat = Path(filename).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (saved["bytes"], saved["mtime_ns"]), "original8400 file stat changed"
        )
    for planned, actual in zip(prepared["teacher_calls"], features["teacher_calls"], strict=True):
        require({key: actual[key] for key in planned} == planned, "026 original144 cache order differs")
        metadata, arrays = OLD.bundle(actual["bundle"], evidence)
        require(
            verified["input_sha256"][str(Path(actual["bundle"]["path"]).resolve())] == actual["bundle"]["sha256"],
            "025 feature verification did not bind this original bundle",
        )
        require(
            verified["input_sha256"].get(metadata["arrays"]["path"]) == metadata["arrays"]["sha256"],
            "025 feature verification did not bind this original archive",
        )
        require(all(metadata[k] == v for k, v in planned.items()), "026 cached metadata source differs")
        cached_contract(arrays, metadata, planned)
        for key in ("conditioning_ref", "rng_before", "rng_after"):
            ref = metadata[key]
            require(
                verified["input_sha256"].get(ref["path"]) == ref["sha256"],
                "026 cached condition/RNG lost original audit binding",
            )
            evidence.ref(ref)
    return prepared, features


def cached_rows(prepared, features, evidence):
    anchors = {a["name"]: a for a in prepared["anchors"]}
    columns = {name: [] for name in ("hidden", "base", "target", "sigma_actual")}
    rows = []
    for call in features["teacher_calls"]:
        metadata, arrays = OLD.bundle(call["bundle"], evidence)
        anchor = anchors[call["anchor_name"]]
        for name, shape in dict(
            hidden=(40, 1280), base=(40, 128), target=(40, 128), sigma_action=(40, 1), valid=(40, 128)
        ).items():
            require(arrays[name].shape == shape, f"original cached {name} shape differs")
        cached_contract(arrays, metadata, call)
        sigma_actual = sigmas64(arrays["sigma_action"], 40)
        exact(sigma_actual, np.full(40, float(np.float32(call["sigma"]))), "026 actual cached sigma provenance")
        exact(arrays["valid"], np.broadcast_to(OLD.ACTIVE, (40, 128)), "026 unchanged teacher active10")
        for name in ("hidden", "base"):
            require(metadata["original_dtypes"][name] == "torch.bfloat16", "026 teacher original BF16 changed")
            exact(arrays[name], OLD.quantize_bfloat16(arrays[name]), f"026 cached {name} lossless BF16")
        for name in ("hidden", "base", "target"):
            columns[name].append(arrays[name])
        columns["sigma_actual"].append(sigma_actual)
        rows.extend(
            dict(
                episode=anchor["episode"],
                anchor_name=anchor["name"],
                sigma=call["sigma"],
                split=anchor["split"],
                task_index=anchor["task_index"],
                endpoint=i,
            )
            for i in range(40)
        )
    require(
        len(rows) == 5760
        and all(sum(row["split"] == split for row in rows) == 2880 for split in ("train", "validation")),
        "026 cached teacher endpoint split differs",
    )
    return {key: np.concatenate(value) for key, value in columns.items()}, rows


def bf16_bits(values):
    rounded = OLD.quantize_bfloat16(values)
    return (rounded.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)


def precision_arrays(arrays, cases):
    """Rebuild the registered synthetic coverage and all actual hardware bits."""
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
    subnormal = [
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
    sigmas = np.asarray([0.2, 0.5, 0.8, 0.12809006869792938, 0.9990000128746033], np.float32).astype(np.float64)
    names = [f"sigma{i}_{name}" for i in range(5) for name in ("positive", "negative", "subnormal")]
    require([c["name"] for c in cases] == names, "026 synthetic precision coverage changed")
    required = {
        f"{name}_{key}"
        for name in names
        for key in ("hidden", "base", "weights", "sigma_actual", "cpu_bits", "cuda_bits")
    }
    require(
        set(arrays) == required | {"primitive_values", "primitive_cpu_bits", "primitive_cuda_bits"},
        "026 synthetic tensor roster changed",
    )
    for index, sigma in enumerate(sigmas):
        for scenario, scalar, residual in (
            ("positive", 1.0, positive),
            ("negative", -1.0, -np.asarray(positive)),
            ("subnormal", 0.0, subnormal),
        ):
            name = f"sigma{index}_{scenario}"
            record = cases[names.index(name)]
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
                "026 synthetic hook/bypass/input evidence missing",
            )
            hidden = np.zeros((2, 1280), np.float32)
            hidden[1, :2] = [0.5, -0.25]
            base = np.full((2, 128), scalar, np.float32)
            base[:, 10:] = np.arange(118)
            base[:, 10] = -0.0
            weights = np.zeros((1281, 10), np.float64)
            weights[-1] = -np.asarray(residual, np.float64) * sigma
            weights[0, 0], weights[1, 1] = 2**-20, -(2**-21)
            values = dict(hidden=hidden, base=base, weights=weights, sigma_actual=np.full((2, 1), sigma))
            for key, expected in values.items():
                actual = numeric(arrays[f"{name}_{key}"], name)
                require(actual.dtype == expected.dtype, f"026 synthetic {key} precision changed")
                if key in ("base", "weights"):
                    exact(
                        actual.view(np.uint32 if actual.dtype == np.float32 else np.uint64),
                        expected.view(np.uint32 if actual.dtype == np.float32 else np.uint64),
                        f"026 registered {name} {key} bits",
                    )
                else:
                    exact(actual, expected, f"026 registered {name} {key}")
            expected = quantized_velocity(hidden, base, values["sigma_actual"], weights)
            expected = (expected.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
            for device in ("cpu", "cuda"):
                require(arrays[f"{name}_{device}_bits"].dtype == np.int16, "026 actual BF16 bits must be int16")
                exact(arrays[f"{name}_{device}_bits"], expected, f"026 independent {name} {device} full128 bits")
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
            1 + half + epsilon,
            -1 - half - epsilon,
        ],
        np.float64,
    )
    require(arrays["primitive_values"].dtype == np.float64, "026 primitive FP64 input changed")
    exact(arrays["primitive_values"].view(np.uint64), primitive.view(np.uint64), "026 signed primitive input bits")
    for device in ("cpu", "cuda"):
        require(arrays[f"primitive_{device}_bits"].dtype == np.int16, "026 primitive BF16 bits must be int16")
        exact(arrays[f"primitive_{device}_bits"], bf16_bits(primitive), f"026 independent {device} primitive RNE")


def verify_quantization(reference, prepared, evidence):
    actual = evidence.ref(reference)
    prepared_path = Path(prepared["output_root"]) / "preparation/prepared.json"
    require(
        actual["schema_version"] == "s2a.flow-readout-synthetic-cpu-cuda-precision.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual
        and actual["prepared_ref"] == dict(path=str(prepared_path), sha256=OLD.digest(prepared_path)),
        "026 synthetic precision admission is partial or bound to another preparation",
    )
    require(
        actual["seed"] == 42
        and actual["physical_gpu"] == 6
        and actual["cuda_visible_devices"] == "6"
        and actual["actual_cuda_index"] == 0
        and actual["actual_gpu_uuid"].removeprefix("GPU-").lower() == "7f176867-170c-b508-5c89-1cd819ad3b8e"
        and actual["cpu_cuda_bitwise_equal"] is True,
        "026 actual synthetic physicalGPU6 identity differs",
    )
    require(
        all(
            actual[k] == 0
            for k in ("model_weights_loaded", "wam_denoise_calls", "fit_groups", "episode_reads", "physics_steps")
        ),
        "026 synthetic precision exceeded its scientific-work budget",
    )
    core = str(REPO / "src/s2a/wam_flow_readout.py")
    require(actual["core_sha256"] == prepared["source_sha256"][core], "026 precision tested another flow core")
    evidence.check(core, actual["core_sha256"])
    worker = actual["worker_source"]
    require(
        worker["path"] == str(REPO / "scripts/probe_wam_flow_readout_quantization.py")
        and prepared["source_sha256"].get(worker["path"]) == worker["sha256"],
        "026 precision worker was not frozen",
    )
    evidence.check(worker["path"], worker["sha256"])
    raw = evidence.json(Path(prepared["output_root"]) / "precision/worker_result.json")
    require(all(actual.get(k) == v for k, v in raw.items()), "026 precision owner rewrote actual worker evidence")
    with np.load(
        evidence.check(actual["actual_outputs"]["path"], actual["actual_outputs"]["sha256"]), allow_pickle=False
    ) as archive:
        precision_arrays({key: numeric(archive[key], key).copy() for key in archive.files}, actual["cases"])
    require(
        evidence.ref(actual["exit_ref"]) == dict(actual_wait=True, automatic_retry=False, returncode=0),
        "026 precision lacks actual normal wait",
    )
    release = evidence.ref(actual["release_ref"])
    directory = Path(prepared["output_root"]) / "orchestration/precision"
    launch = evidence.json(directory / "model/launch.json")
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH=""),
        "026 actual precision worker environment differs",
    )
    require(
        all(release[k] == launch["child"][k] for k in ("pid", "start_ticks"))
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "026 precision owned child did not normally exit/reap",
    )
    gpu = evidence.ref(actual["gpu_after_ref"])
    require(
        gpu["physical_gpu"] == 6
        and gpu["uuid"] == "GPU-7f176867-170c-b508-5c89-1cd819ad3b8e"
        and gpu["memory_mib"] == 0
        and gpu["released"] is True
        and gpu["compute_apps"] == [],
        "026 precision GPU6 memory/process release is unproven",
    )
    observations = [
        json.loads(line)
        for line in evidence.check(
            actual["resource_observations_ref"]["path"], actual["resource_observations_ref"]["sha256"]
        )
        .read_text()
        .splitlines()
    ]
    require(
        observations
        and all(o["physical_gpu"] == 6 and o["uuid"] == gpu["uuid"] for o in observations)
        and all(
            observations[i]["seconds_since_actual_owned_wait"] <= observations[i + 1]["seconds_since_actual_owned_wait"]
            for i in range(len(observations) - 1)
        ),
        "026 precision resource polls missing or unordered",
    )
    require(
        0
        <= observations[0]["seconds_since_actual_owned_wait"]
        <= observations[-1]["seconds_since_actual_owned_wait"]
        <= 120
        and all(gpu[k] == v for k, v in observations[-1].items()),
        "026 precision released outside its recorded120second budget",
    )
    owner = evidence.json(directory / "result.json")
    require(
        owner["complete"] is True
        and owner["passed"] is True
        and owner["stage"] == "precision"
        and owner["producer_result"] == reference
        and owner["automatic_retry"] is False
        and len(owner["owned_commands"]) == 1,
        "026 synthetic owner did not complete its unique actual worker",
    )
    require(
        owner["owned_commands"][0]["exit"] == actual["exit_ref"]
        and owner["owned_commands"][0]["release"] == actual["release_ref"],
        "026 precision owner/worker lifecycle refs differ",
    )
    return dict(
        cases_verified=15,
        primitive_values_verified=10,
        actual_cpu_cuda_full128_bits_verified=True,
        owned_normal_exit_and_release_verified=True,
    )


def verify_fit(prepared, features, path, evidence):
    actual = evidence.json(path)
    require(
        actual["schema_version"] == "s2a.flow-readout-fit.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual,
        "partial/rejected026 fit",
    )
    prepared_path = Path(prepared["output_root"]) / "preparation/prepared.json"
    require(
        actual["prepared_ref"] == dict(path=str(prepared_path), sha256=OLD.digest(prepared_path)),
        "026 fit bound to another preparation",
    )
    require(
        actual["producer_fit_groups"] == actual["fit_attempts"] == 2
        and actual["new_model_calls"] == actual["physics_steps"] == 0,
        "026 fixed two-fit/model budget differs",
    )
    require(
        actual["cuda_visible_devices"] == actual["ld_library_path"] == ""
        and actual["ridge"] == 0.1
        and actual["deployment_dtype"] == "torch.bfloat16",
        "026 CPU/regularization/precision contract differs",
    )
    require(
        actual["features_ref"] == prepared["source_features_ref"]
        and actual["source_prepared_ref"] == prepared["source_prepared_ref"]
        and actual["original025_eligible_for_sampling"] is False
        and actual["new_teacher_calls"] == actual["new_episode_reads"] == 0
        and actual["parameterization"] == prepared["config"]["parameterization"],
        "026 fit changed source/target provenance or scope",
    )
    quantization = verify_quantization(actual["quantization_ref"], prepared, evidence)
    arrays, rows = cached_rows(prepared, features, evidence)
    require(
        actual["teacher_rows"] == rows and actual["fitted_rows"] == actual["validation_rows"] == 2880,
        "026 fit teacher provenance differs",
    )
    require(
        actual["training_episodes"] == prepared["config"]["train_episodes"]
        and actual["validation_episodes"] == prepared["config"]["validation_episodes"],
        "026 fit whole-episode split changed",
    )
    selected = np.asarray([row["split"] == "train" for row in rows])
    weights = OLD.hierarchical_weights([row for row in rows if row["split"] == "train"])
    target = arrays["target"][selected, :10].astype(np.float64) - arrays["base"][selected, :10].astype(np.float64)
    reconstructed = augmented_fit(arrays["hidden"][selected], arrays["sigma_actual"][selected], target, weights)
    coefficients, parameter_differences = {}, {}
    for name in ("adapt", "bias"):
        reference = actual["models"][name]
        with np.load(evidence.check(reference["path"], reference["sha256"]), allow_pickle=False) as archive:
            require(archive.files == ["weights"], "026 saved model has extra parameters")
            coefficients[name] = numeric(archive["weights"], name).copy()
        require(
            coefficients[name].shape == (1281, 10) and coefficients[name].dtype == np.float64,
            "026 coefficients must be1281×10 FP64",
        )
        close(coefficients[name], reconstructed[name], f"026 independent augmented-LS {name}")
        parameter_differences[name] = float(np.abs(coefficients[name] - reconstructed[name]).max())
    predictions = dict(raw=arrays["base"])
    predictions.update(
        {
            name: quantized_velocity(arrays["hidden"], arrays["base"], arrays["sigma_actual"], value)
            for name, value in coefficients.items()
        }
    )
    reference = actual["teacher_predictions"]
    with np.load(evidence.check(reference["path"], reference["sha256"]), allow_pickle=False) as archive:
        require(
            set(archive.files) == {"raw", "adapt", "bias", "training_weights", "sigma_actual"},
            "026 teacher prediction roster differs",
        )
        exact(archive["training_weights"], weights, "026 unchanged hierarchical weights")
        require(archive["sigma_actual"].dtype == np.float64, "026 recorded actual sigma is not FP64")
        exact(archive["sigma_actual"], arrays["sigma_actual"], "026 actual FP32 sigma expanded exactly")
        for name, values in predictions.items():
            exact(archive[name], values, f"026 actual quantized teacher {name}")
            exact(archive[name][:, 10:], arrays["base"][:, 10:], f"026 inactive118 {name}")
    metrics = OLD.all_teacher_metrics(predictions, arrays["target"], rows)
    OLD.compare_metrics(actual["teacher_metrics"], metrics)
    gate = OLD.independent_teacher_gate(metrics)
    require(
        actual["teacher_gate"] == gate and actual["eligible_for_sampling"] is gate["passed"],
        "026 unchanged strict teacher gate differs",
    )
    return dict(
        fit_verified=True,
        independent_model_reconstructions=2,
        max_parameter_abs_difference=parameter_differences,
        teacher_metrics=metrics,
        teacher_gate=gate,
        eligible_for_sampling=gate["passed"],
        actual_teacher_sigmas=sorted(set(arrays["sigma_actual"].tolist())),
        quantization_verified=quantization,
        raw_checkpoint_task_success_proven=False,
        adapted_state_or_task_capability_proven=False,
    )


def solver_schedule():
    """Reconstruct frozen UniPC/35/shift5 independently, without solver imports."""
    initial = (1 - np.linspace(1, 0.001, 1000)[::-1]).astype(np.float32)
    grid = np.linspace(float(initial[0]), float(initial[-1]), 36)[:-1]
    shifted = 5 * grid / (1 + 4 * grid)
    return np.r_[shifted, 0].astype(np.float32), (shifted * 1000).astype(np.int64)


def sigma_trace(trace, name, seed, prepared):
    require(
        trace["complete"] is True and trace["name"] == name and trace["seed"] == seed and trace["sampler_calls"] == 1,
        "026 actual solver trace identity differs",
    )
    require(
        trace["solver_sigma_dtype"] == "torch.float32"
        and trace["terminal_sigma"] == 0
        and trace["terminal_neural_calls"] == 0,
        "026 terminal sigma invoked a neural evaluation",
    )
    expected, times = solver_schedule()
    exact(trace["sigma_values"], expected.astype(np.float64), "026 actual solver FP32 sigma grid")
    exact(trace["scheduler_timesteps"], times, "026 original integer timestep grid")
    require(
        len(trace["steps"]) == 35
        and trace["parameters_before"] == trace["parameters_after"]
        and trace["parameters_before"]["count"] > 0,
        "026 solver trace original parameters changed",
    )
    require(
        trace["actual_sampler_arguments"] == dict(num_steps=35, shift=5.0, seed=[seed]),
        "026 actual sampler arguments changed",
    )
    require(
        trace["source_sha256"]
        and all(prepared["source_sha256"].get(k) == v for k, v in trace["source_sha256"].items()),
        "026 actual solver source is not frozen",
    )
    for index, step in enumerate(trace["steps"]):
        require(
            step
            == dict(
                step_index=index,
                solver_step_index_before=None if index == 0 else index,
                solver_sigma=float(expected[index]),
                solver_timestep=int(times[index]),
                velocity_delegations=1,
                head_calls=1,
            ),
            "026 actual sigma/step/single velocity delegation differs",
        )


def sampling_execution_contract(actual, prepared, fit):
    require(
        actual["schema_version"] == "s2a.flow-readout-sampling.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual,
        "partial/rejected026 sampling",
    )
    require(
        actual["all_original_parameters_frozen"] is True
        and actual["head_call_counts"] == dict(zero=105, raw=1680, adapt=1680),
        "026 sampling original parameters/head counts differ",
    )
    require(
        actual["physics_steps"] == actual["fresh_cases"] == actual["new_teacher_calls"] == 0,
        "026 sampling exceeded fixed budget",
    )
    require(
        actual["denoise_calls"] == 3465 and len(actual["zero_identity"]) == 3 and len(actual["generations"]) == 96,
        "026 sampling complete99/3465 budget differs",
    )
    require(
        actual["adapted_model_ref"] == fit["models"]["adapt"]
        and actual["runtime"]["checkpoint_path"] == prepared["checkpoint"]
        and actual["runtime"]["ema_weights"] is False,
        "026 sampling checkpoint/candidate identity differs",
    )
    labels = [f"zero:{i}" for i in range(3)] + [f"sampling:{i}" for i in range(96)]
    expected = [
        dict(index=i * 35 + j, label=label, completed=True) for i, label in enumerate(labels) for j in range(35)
    ]
    require(actual["denoise_ledger"] == expected, "026 actual denoise ledger is partial/repeated/out of order")


def condition_signature(recorded, planned, task, prepared, transform, evidence):
    response, condition = evidence.ref(recorded["response"]), evidence.ref(recorded["conditioning"])
    OLD.context_check(condition["context"])
    require(
        condition["inference_only_current_image"] is True
        and condition["model_runtime"]["checkpoint_path"] == prepared["checkpoint"]
        and condition["model_runtime"]["ema_weights"] is False,
        "026 actual per-request observation/checkpoint contract differs",
    )
    caption = json.loads(condition["caption"])
    require(
        caption["task_title"] == caption["command"] == task
        and caption.get("task_plan", []) == []
        and "current_subtask" not in caption,
        "026 actual condition is not task-only",
    )
    require(
        response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"]
        == hashlib.sha256(condition["caption"].encode()).hexdigest(),
        "026 response/condition caption binding differs",
    )
    actions = OLD.response_actions(response, planned["request"], prepared["checkpoint"], transform)
    metadata, arrays = OLD.bundle(recorded["initial"], evidence)
    require(
        set(arrays) == {"initial_noise", "condition_mask", "first_condition_latent"},
        "026 initial tensor roster differs",
    )
    require(
        isinstance(metadata["caption"], str) and isinstance(metadata["text_tokens"], list),
        "026 actual text input is missing",
    )
    signature = OLD.initial_signature(metadata, arrays)
    require(signature["caption"] == condition["caption"], "026 initial text differs from response condition")
    sigma_trace(evidence.ref(recorded["actual_sigma_trace"]), planned["name"], planned["request"]["seed"], prepared)
    return response, actions, metadata, signature


def require_same_initial(actual, original):
    require(
        actual["caption"] == original["caption"]
        and actual["text_tokens"] == original["text_tokens"]
        and set(actual["arrays"]) == set(original["arrays"]),
        "026 paired text/initial tensor roster differs",
    )
    for name, value in actual["arrays"].items():
        exact(value, original["arrays"][name], f"026 raw/adapt actual initial {name}")


def state_labels(prepared, evidence):
    """Only after complete generation, read the original six validation states."""
    labels = {}
    for episode in prepared["config"]["validation_episodes"]:
        anchors = [a for a in prepared["anchors"] if a["episode"] == episode]
        require(
            len(anchors) == 4 and all(a["split"] == "validation" for a in anchors),
            "026 state scoring attempted a training EP",
        )
        reference = anchors[0]
        with np.load(
            evidence.check(reference["arrays_path"], reference["arrays_sha256"]), allow_pickle=False
        ) as archive:
            states, mask = (
                numeric(archive["states"], "validation states").astype(np.float64),
                archive["state_mask"].copy(),
            )
        require(
            states.ndim == 2 and states.shape[1] == 128 and mask.shape == states.shape and mask.dtype == np.bool_,
            "026 state label/mask geometry differs",
        )
        require(
            [a["anchor_frame"] for a in anchors] == OLD.source_anchors(len(states)),
            "026 state scoring source length/clock differs",
        )
        for anchor in anchors:
            frame = anchor["anchor_frame"]
            require(
                anchor["arrays_path"] == reference["arrays_path"]
                and anchor["arrays_sha256"] == reference["arrays_sha256"],
                "026 EP anchors reference different labels",
            )
            exact(states[frame], anchor["measured_state128"], "026 actual current measured anchor")
            exact(mask[frame : frame + 41], np.broadcast_to(OLD.ACTIVE, (41, 128)), "026 state future valid mask")
        labels[episode] = states
    return labels


def verify_sampling(prepared, fit_path, fit_verification_path, path, evidence):
    actual, fit, verified = evidence.json(path), evidence.json(fit_path), evidence.json(fit_verification_path)
    require(
        fit["complete"] is True
        and fit["passed"] is True
        and fit["eligible_for_sampling"] is True
        and verified["phase"] == "fit"
        and verified["passed"] is True
        and verified["fit_verified"] is True
        and verified["eligible_for_sampling"] is True,
        "026 teacher gate did not authorize sampling",
    )
    fit_ref = dict(path=str(Path(fit_path).resolve()), sha256=OLD.digest(fit_path))
    verification_ref = dict(path=str(Path(fit_verification_path).resolve()), sha256=OLD.digest(fit_verification_path))
    prepared_path = Path(prepared["output_root"]) / "preparation/prepared.json"
    prepared_ref = dict(path=str(prepared_path), sha256=OLD.digest(prepared_path))
    require(
        actual["fit_ref"] == fit_ref
        and actual["fit_verification_ref"] == verification_ref
        and actual["prepared_ref"] == verified["prepared"] == fit["prepared_ref"] == prepared_ref
        and verified["input_sha256"].get(fit_ref["path"]) == fit_ref["sha256"],
        "026 sampling admission consumed different fit/preparation",
    )
    for filename, expected in verified["input_sha256"].items():
        evidence.check(filename, expected)
    sampling_execution_contract(actual, prepared, fit)
    source = evidence.ref(prepared["source_prepared_ref"])
    config_path = str(Path(prepared["config"]["checkpoint"]["run"]) / "config.yaml")
    spec = OLD.yaml.safe_load(evidence.check(config_path, source["input_sha256"][config_path]).read_text())["model"][
        "config"
    ]["action128_spec"]
    require(
        spec["registry"]["domains"].index(OLD.DOMAIN) == 31 and spec["registry"]["domain_projection_ids"][31] == 0,
        "026 sampling normalization/projection registry changed",
    )
    transform = spec["transforms"][31]
    for planned, recorded in zip(prepared["zero_identity_cases"], actual["zero_identity"], strict=True):
        require(
            recorded["name"] == planned["name"] and recorded["task_index"] == planned["task_index"],
            "026 three zero roster/order differs",
        )
        evidence.check(planned["sample"]["image_path"], planned["sample"]["image_sha256"])
        response, _, metadata, _ = condition_signature(
            recorded, planned, planned["request"]["prompt"], prepared, transform, evidence
        )
        zero_entry = dict(
            name=planned["name"],
            generation_index=-planned["task_index"] - 1,
            pair_id=f"zero{planned['task_index']}",
            variant="zero",
            sampling_seed=42,
        )
        OLD.initial_record_identity(metadata, zero_entry)
        cached = evidence.json(planned["cache"]["response_path"], planned["cache"]["response_sha256"])
        current, prior = response["outputs"][0]["content"], cached["outputs"][0]["content"]
        exact(current["action"], prior["action"], "026 zero full128 raw identity")
        exact(
            current["action128"]["normalized_action"],
            prior["action128"]["normalized_action"],
            "026 zero full128 normalized identity",
        )
    anchors = {a["name"]: a for a in prepared["anchors"]}
    pairs, generated = {}, []
    for planned, recorded in zip(prepared["validation_generations"], actual["generations"], strict=True):
        require({k: recorded[k] for k in planned} == planned, "026 paired generation roster/order differs")
        anchor = anchors[planned["anchor_name"]]
        require(anchor["split"] == "validation", "026 training source entered validation generation")
        evidence.check(anchor["image_path"], anchor["image_sha256"])
        _, actions, metadata, signature = condition_signature(
            recorded, planned, anchor["task"], prepared, transform, evidence
        )
        OLD.initial_record_identity(metadata, planned)
        pair_id = planned["pair_id"]
        if planned["variant"] == "raw":
            require(pair_id not in pairs, "026 duplicate raw generation pair")
            pairs[pair_id] = signature
        else:
            require(pair_id in pairs, "026 adapted generation lacks raw pair")
            require_same_initial(signature, pairs[pair_id])
        generated.append((planned, anchor, actions))
    require(len(pairs) == 48, "026 sampling lacks48 independent initial pairs")
    # All responses, identity controls and actual source/step traces were
    # validated before any true future state file is opened.
    labels, integrate, rows = state_labels(prepared, evidence), OLD.frozen_integrator(evidence), []
    for planned, anchor, actions in generated:
        count, errors, _ = integrate(labels[anchor["episode"]], anchor, actions)
        require(count == 40, "026 state scoring lost common H40 endpoints")
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
        "026 state scoring episode/seed coverage differs",
    )
    gate = OLD.independent_state_gate(groups)
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
        eligible_for_development_physics=gate["passed"],
        raw_checkpoint_task_success_proven=False,
        physics_tasks_executed=0,
    )


def phase_arguments(args):
    fields = (args.fit is not None, args.fit_verification is not None, args.sampling is not None)
    require(
        fields
        == dict(prepare=(False, False, False), fit=(True, False, False), sampling=(True, True, True))[args.phase],
        "026 phase/evidence argument mismatch",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "sampling"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--fit-verification", type=Path)
    parser.add_argument("--sampling", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "026 verifier requires CPU emptyCUDA/LD and seed42",
    )
    phase_arguments(args)
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "026 new receipt must be under outputs; refuse overwrite",
    )
    evidence = OLD.Evidence()
    receipt = dict(
        schema_version="s2a.flow-readout-independent-verification.v1",
        phase=args.phase,
        passed=False,
        new_model_calls=0,
        new_candidate_fits=0,
        new_physics_steps=0,
    )
    error = None
    try:
        prepared, features = verify_preparation(args.prepared, evidence)
        receipt["prepared"] = dict(path=str(args.prepared.resolve()), sha256=OLD.digest(args.prepared))
        if args.phase == "prepare":
            receipt.update(
                preparation_verified=True, old_features_reused=144, old_candidate_eligible=False, new_teacher_calls=0
            )
        elif args.phase == "fit":
            receipt.update(verify_fit(prepared, features, args.fit, evidence))
        else:
            receipt.update(verify_sampling(prepared, args.fit, args.fit_verification, args.sampling, evidence))
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
    print(json.dumps(dict(passed=receipt["passed"], output=str(output), sha256=OLD.digest(output))))
    if error is not None:
        raise SystemExit(f"026 independent audit rejected: {error}")


if __name__ == "__main__":
    main()
