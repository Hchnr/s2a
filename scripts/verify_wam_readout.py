"""Independent CPU audit of025 preparation, features, fits and paired sampling.

No producer fit/apply/gate, Torch, Cosmos or dataloader is imported. The four
phases are explicit; paired sampling scores reference states, not task success.
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from s2a.artifacts import REPO

CONFIG = REPO / "configs/wam_frozen_readout_20261021.json"
CONFIG_SHA = "6534245a114d89167f6d8cd283e55de8685b4a02abca79706436309477860605"
PLAN = REPO / "plans/v0.0.1/025_frozen_wam_readout_20261021.md"
PLAN_SHA = "46b766356ef29c593c895a23130a0415cc83c892afcfc6bb66e07e5bab540dc6"
DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
ACTIVE = np.asarray([True] * 10 + [False] * 118)
FINGERPRINT_SHA = "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8"
INTEGRATOR_SHA = "12329768297939fe6b15b4127aa6b65e144578fd72b0e896dd306f9e88b10069"
ROTATION_SHA = "2c3727f74c54b03a3b089cbbe501fca7dce77dd899a9c23fc3e8f608e6c251c7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def numeric(value, name):
    result = np.asarray(value)
    require(result.dtype.kind in "biuf" and np.isfinite(result).all(), f"{name}: nonfinite/nonnumeric tensor")
    return result


def exact(actual, expected, name):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(actual.shape == expected.shape and np.array_equal(actual, expected), f"{name}: exact mismatch")


def close(actual, expected, name):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(
        actual.shape == expected.shape and np.allclose(actual, expected, rtol=2e-12, atol=2e-12),
        f"{name}: independent numerical reconstruction differs",
    )


class Evidence:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        require(expected is None or actual == expected, f"source SHA changed: {path}")
        require(
            str(path) not in self.hashes or self.hashes[str(path)] == actual, f"source changed during audit: {path}"
        )
        self.hashes[str(path)] = actual
        return path

    def json(self, path, expected=None):
        return json.loads(self.check(path, expected).read_text())

    def ref(self, reference):
        require(set(reference) == {"path", "sha256"}, "invalid evidence reference")
        return self.json(reference["path"], reference["sha256"])

    def recheck(self):
        for path, expected in self.hashes.copy().items():
            self.check(path, expected)


def bundle(reference, evidence):
    metadata = evidence.ref(reference)
    archive = evidence.check(metadata["arrays"]["path"], metadata["arrays"]["sha256"])
    with np.load(archive, allow_pickle=False) as source:
        arrays = {name: numeric(source[name], name).copy() for name in source.files}
    require(set(arrays) == set(metadata["array_index"]), "saved tensor index omitted/added arrays")
    for name, value in arrays.items():
        require(
            metadata["array_index"][name] == dict(shape=list(value.shape), dtype=str(value.dtype)),
            f"saved tensor index differs: {name}",
        )
    return metadata, arrays


def quantize_bfloat16(values):
    """Direct FP64→BF16 ties-to-even, without an intervening FP32 rounding.

    BF16 normal spacing is 2**(floor(log2(abs(x)))-7); subnormal spacing
    is 2**-133. Scaling a binary float by powers of two is exact here.
    """
    value = numeric(values, "BF16 input").astype(np.float64)
    magnitude = np.abs(value)
    _, exponent = np.frexp(magnitude)
    step = np.ldexp(np.ones(value.shape), np.maximum(exponent - 8, -133))
    rounded = np.rint(magnitude / step) * step
    require(np.all(rounded <= np.ldexp(2 - 2**-7, 127)), "BF16 deployment overflow")
    return np.copysign(rounded, value).astype(np.float32)


def phi(hidden):
    hidden = numeric(hidden, "hidden").astype(np.float64)
    require(hidden.ndim == 2 and min(hidden.shape) > 0, "hidden must be a nonempty matrix")
    return np.concatenate((hidden / np.sqrt(hidden.shape[1]), np.ones((len(hidden), 1))), axis=1)


def augmented_fit(hidden, residual, row_weights):
    design = phi(hidden)
    target = numeric(residual, "residual").astype(np.float64)
    weights = numeric(row_weights, "row weights").astype(np.float64)
    require(target.shape == (len(design), 10) and weights.shape == (len(design),), "fit row/active10 dimensions differ")
    require(np.all(weights > 0) and abs(weights.sum() - 1) <= 1e-12, "fit weights must be positive and sum to one")
    x = np.concatenate((np.sqrt(weights)[:, None] * design, np.sqrt(0.1) * np.eye(design.shape[1])))
    y = np.concatenate((np.sqrt(weights)[:, None] * target, np.zeros((design.shape[1], 10))))
    adapt = np.linalg.lstsq(x, y, rcond=None)[0]
    bias = np.zeros_like(adapt)
    bias[-1] = (weights @ target) / (weights.sum() + 0.1)
    return dict(adapt=adapt, bias=bias)


def apply_quantized(hidden, base, coefficients):
    hidden, base, coefficients = (
        numeric(value, name) for value, name in ((hidden, "hidden"), (base, "base"), (coefficients, "coefficients"))
    )
    require(
        base.shape == (len(hidden), 128) and coefficients.shape == (hidden.shape[1] + 1, 10), "readout shape differs"
    )
    result = base.copy()
    if np.any(coefficients):
        result[:, :10] = quantize_bfloat16(base[:, :10].astype(np.float64) + phi(hidden) @ coefficients)
    return result


def source_anchors(length):
    require(type(length) is int and length >= 44, "complete four-anchor episode required")
    result = [j * (length - 41) // 3 for j in range(4)]
    require(len(set(result)) == 4 and result[-1] + 41 == length, "wrong complete source anchor clock")
    return result


def request_check(request, task, image_path):
    require(
        request["prompt"] == request["extra"]["action128"]["task_title"] == task
        and request["vision_path"] == image_path,
        "wrong current image/task request",
    )
    require(
        request["extra"]["action128"]["action_domain"] == DOMAIN
        and request["extra"]["action128"]["action_dim_mask"] == ACTIVE.tolist(),
        "wrong request full domain/mask",
    )
    for key, expected in dict(
        seed=42, model_mode="wam", action_chunk_size=40, num_frames=41, fps=20, guidance=1.0, num_steps=35, shift=5.0
    ).items():
        require(request[key] == expected, f"request {key} changed")


def verify_admission(config, evidence):
    result_path = REPO / config["required_numerics_result"]
    verified = evidence.json(REPO / config["required_numerics_verification"])
    actual = evidence.json(result_path)
    require(
        verified["passed"] is True
        and verified["complete"] is True
        and verified["calls_verified"] == 27
        and verified["independent_losses_verified"] == 18,
        "024 complete independent admission missing",
    )
    require(
        actual["complete"] is True and actual["passed"] is True and "error" not in actual,
        "024 actual result is partial/rejected",
    )
    require(
        verified["input_sha256"][str(result_path)] == digest(result_path),
        "024 verification does not bind actual result",
    )
    for path, expected in verified["input_sha256"].items():
        evidence.check(path, expected)
    folder = result_path.parent.parent / "orchestration"
    launch, exit_record, release = (
        evidence.json(folder / name) for name in ("launch.json", "exit.json", "model_ownership/release.json")
    )
    require(
        exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0), "024 has no successful actual wait"
    )
    require(
        release["pid"] == launch["model"]["pid"]
        and release["start_ticks"] == launch["model"]["start_ticks"]
        and release["returncode"] == 0
        and release["reaped"] is True
        and release["signal_sent"] is False
        and release["original_error"] is None,
        "024 owned model release differs",
    )
    for name in ("owner", "model"):
        stat = Path(f"/proc/{launch[name]['pid']}/stat")
        if stat.exists():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            require(
                int(fields[19]) != launch[name]["start_ticks"] or fields[0] in ("Z", "X"),
                "024 same owned process is still live",
            )


def verify_quantization(evidence):
    folder = REPO / "outputs/wam_stability_20261021/readout_quantization_gpu6"
    result = evidence.json(folder / "result.json", "f4a6953d064570fae9c215464f06dd5e0feb0c9531e708e5ff5d54accb190b74")
    require(
        result["complete"] is True
        and result["passed"] is True
        and result["physical_gpu"] == 6
        and result["cuda_visible_devices"] == "6",
        "synthetic CPU/GPU6 quantization admission missing",
    )
    evidence.check(REPO / "src/s2a/wam_readout.py", result["core_sha256"])
    evidence.check(folder / "worker.py", result["worker_sha256"])
    launch = evidence.json(folder / "launch.json")
    evidence.check(folder / "owner.py", launch["owner_sha256"])
    require(
        evidence.json(folder / "exit.json") == dict(actual_wait=True, automatic_retry=False, returncode=0),
        "synthetic quantization worker did not exit normally",
    )
    release = evidence.json(folder / "ownership/release.json")
    require(
        release["pid"] == launch["child"]["pid"]
        and release["start_ticks"] == launch["child"]["start_ticks"]
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False,
        "synthetic quantization owned release differs",
    )
    reference = result["actual_outputs"]
    with np.load(evidence.check(reference["path"], reference["sha256"]), allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    half, epsilon, spacing = 2**-8, 2**-30, 2**-133
    subhalf, subepsilon = spacing / 2, 2**-160
    cases = (
        (
            "normal_positive",
            1.0,
            [
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
            ],
        ),
        (
            "normal_negative",
            -1.0,
            [
                -half + epsilon,
                -half,
                -half - epsilon,
                half - epsilon,
                half,
                half + epsilon,
                -3 * half + epsilon,
                -3 * half,
                -3 * half - epsilon,
                -0.0,
            ],
        ),
        (
            "subnormal",
            0.0,
            [
                subhalf - subepsilon,
                subhalf,
                subhalf + subepsilon,
                -subhalf + subepsilon,
                -subhalf,
                -subhalf - subepsilon,
                1.5 * spacing,
                2.5 * spacing,
                2**-126 - subhalf,
                -0.0,
            ],
        ),
    )

    def bits(value):
        return (value.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)

    for name, base, residual in cases:
        expected = np.full((2, 128), base, np.float32)
        expected[:, 10:] = np.arange(118)
        expected[:, 10] = -0.0
        expected[:, :10] = quantize_bfloat16(np.float64(base) + np.asarray(residual))
        for device in ("cpu", "cuda_0"):
            exact(arrays[f"{name}_{device}"], bits(expected), f"independent actual {name} {device} BF16 bits")
    primitive = quantize_bfloat16(
        [
            0.0,
            -0.0,
            subhalf - subepsilon,
            subhalf,
            subhalf + subepsilon,
            -subhalf + subepsilon,
            -subhalf,
            -subhalf - subepsilon,
            1 + half + epsilon,
            -1 - half - epsilon,
        ]
    )
    for device in ("cpu", "cuda_0"):
        exact(arrays[f"primitive_{device}"], bits(primitive), f"independent actual {device} primitive BF16 bits")


def verify_preparation(path, evidence):
    prepared = evidence.json(path)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    require(
        prepared["schema_version"] == "s2a.frozen-readout-preparation.v1" and prepared["config"] == config,
        "wrong025 prepared schema/config",
    )
    require(
        prepared["config_ref"] == dict(path=str(CONFIG), sha256=CONFIG_SHA)
        and prepared["registration"] == dict(path=str(PLAN), sha256=PLAN_SHA),
        "wrong025 frozen registration refs",
    )
    require(
        prepared["budget"] == config["budget"]
        and prepared["new_model_calls"]
        == prepared["fit_count"]
        == prepared["physics_steps"]
        == prepared["fresh_performance_cases"]
        == 0,
        "preparation consumed model/fit/physics/fresh budget",
    )
    for name in ("source_sha256", "snapshot_sha256", "input_sha256"):
        require(prepared[name], f"empty {name}")
        for source, expected in prepared[name].items():
            evidence.check(source, expected)
    verify_admission(config, evidence)
    verify_quantization(evidence)
    quantization = prepared["quantization_admission"]
    quantization_root = REPO / "outputs/wam_stability_20261021/readout_quantization_gpu6"
    for key, source in dict(
        result=quantization_root / "result.json",
        actual_outputs=quantization_root / "actual_outputs.npz",
        launch=quantization_root / "launch.json",
        owner_exit=quantization_root / "exit.json",
        child_release=quantization_root / "ownership/release.json",
        actual_child_identity=quantization_root / "actual_child_identity.json",
        core_source=REPO / "src/s2a/wam_readout.py",
        worker_source=quantization_root / "worker.py",
        owner_source=quantization_root / "owner.py",
    ).items():
        require(
            quantization[key] == dict(path=str(source), sha256=digest(evidence.check(source))),
            "prepared synthetic quantization source reference differs",
        )
    require(
        quantization["owner_and_child_births_terminal"] is True
        and quantization["physical_gpu"] == 6
        and quantization["actual_cpu_cuda_gold_bits_verified"] is True
        and quantization["full_readout_cases"] == 3
        and quantization["primitive_values"] == 10,
        "prepared quantization numerical/lifecycle admission differs",
    )
    fingerprint = evidence.json(REPO / config["source_checkpoint_fingerprints"], FINGERPRINT_SHA)
    actual = next(c for c in fingerprint["checkpoints"] if c["name"] == "8400_masked")
    expected_checkpoint = str(
        Path(config["checkpoint"]["run"]) / "checkpoints" / config["checkpoint"]["iteration"] / "model"
    )
    require(
        actual["checkpoint"] == prepared["checkpoint"] == expected_checkpoint
        and actual["files"] == prepared["weight_files"]
        and len(actual["files"]) == 163,
        "wrong8400 identity/file roster",
    )
    for source, saved in actual["files"].items():
        stat = Path(source).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (saved["bytes"], saved["mtime_ns"]), "actual8400 weight stat changed"
        )
    saved = yaml.safe_load(evidence.check(Path(config["checkpoint"]["run"]) / "config.yaml").read_text())["model"][
        "config"
    ]
    registry = saved["action128_spec"]["registry"]
    require(
        registry["domains"].index(DOMAIN) == 31 and registry["domain_projection_ids"][31] == 0,
        "LIBERO full normalization/projection identity differs",
    )
    require(
        prepared["domain"]["name"] == DOMAIN
        and prepared["domain"]["normalization_id"] == 31
        and prepared["domain"]["projection_id"] == 0
        and prepared["domain"]["active_mask"] == ACTIVE.tolist(),
        "prepared full-domain gate differs",
    )
    source_cases = evidence.json(REPO / config["source_samples_manifest"])["cases"]
    prior = evidence.json(REPO / config["source_sampler_prepared"])
    require(len(prepared["zero_identity_cases"]) == 3, "three original zero-identity requests required")
    for task_index, item in enumerate(prepared["zero_identity_cases"]):
        sample = next(s for s in source_cases if s["task_index"] == task_index and s["phase"] == "approach")
        name = f"{sample['name']}_h40_task_only_s42"
        cache = prior["cached_cases"][name]
        require(
            item["sample"] == sample
            and item["cache"] == cache
            and item["name"] == name
            and item["task_index"] == task_index,
            "zero identity source/roster changed",
        )
        request = evidence.json(cache["request_path"], cache["request_sha256"])
        require(item["request"] == request, "zero identity request differs from original011")
        request_check(request, sample["task"], sample["image_path"])
        evidence.json(cache["response_path"], cache["response_sha256"])
    exclusions = prepared["conservative_source_exclusion"]
    fresh = evidence.ref(exclusions["reference"])
    require(
        exclusions["historical_initial_physical_sync_certified"] is False
        and exclusions["pretraining_visibility"] == "unknown",
        "source exclusion evidence upgraded",
    )
    require(len(prepared["episodes"]) == 12 and len(prepared["anchors"]) == 48, "wrong12-episode/48-anchor roster")
    expected_splits = [
        (episode, split)
        for split, key in (("train", "train_episodes"), ("validation", "validation_episodes"))
        for episode in config[key]
    ]
    require(
        [(e["episode"], e["split"]) for e in prepared["episodes"]] == expected_splits, "episode split/order differs"
    )
    require(
        not set(config["train_episodes"]) & set(config["validation_episodes"]), "validation leaked into training split"
    )
    original_sources = set()
    index = {}
    for item, (episode, split) in zip(prepared["episodes"], expected_splits, strict=True):
        sample = next(s for s in source_cases if s["episode"] == episode)
        original_split = "diagnostic" if split == "train" else "study_heldout"
        require(sample["split"] == original_split, "original development source split changed")
        require(sample["source_episode_id"] not in original_sources, "duplicate original source episode")
        original_sources.add(sample["source_episode_id"])
        r1_episode = int(json.loads(sample["source_episode_id"])["source_id"].rsplit(":", 1)[1])
        exclusion = next(e for e in exclusions["records"] if e["r2_episode"] == episode)
        original = next(e for e in fresh["offline_source_exclusion_records"] if e["r1_episode"] == r1_episode)
        require(
            exclusion == dict(original, r2_episode=episode, source_episode_id=sample["source_episode_id"]),
            "conservative source exclusion relationship changed",
        )
        require(
            original["demo"] in fresh["excluded_demos_by_task"][sample["task_index"]]
            and original["initial_sync_certified"] is False,
            "fresh exclusion/synchronization evidence differs",
        )
        require(
            item["source_episode_id"] == sample["source_episode_id"] and item["task_index"] == sample["task_index"],
            "whole-source identity differs",
        )
        with np.load(evidence.check(sample["arrays_path"], sample["arrays_sha256"]), allow_pickle=False) as arrays:
            frames = source_anchors(len(arrays["states"]))
            require(item["frames"] == frames, "nonuniform/shifted source anchors")
            current = [a for a in prepared["anchors"] if a["episode"] == episode]
            require(
                len(current) == 4 and [a["name"] for a in current] == item["anchors"], "episode anchor aliases differ"
            )
            for i, (anchor, frame) in enumerate(zip(current, frames, strict=True)):
                require(
                    anchor["anchor_index"] == i
                    and anchor["anchor_frame"] == frame
                    and anchor["split"] == split
                    and anchor["source_episode_id"] == sample["source_episode_id"],
                    "wrong source anchor identity/split",
                )
                require(
                    anchor["arrays_path"] == sample["arrays_path"]
                    and anchor["arrays_sha256"] == sample["arrays_sha256"]
                    and anchor["task"] == sample["task"]
                    and anchor["task_index"] == sample["task_index"],
                    "wrong source task/arrays",
                )
                for role in ("auxiliary_0", "left_wrist"):
                    require(
                        arrays[role][frame : frame + 41].shape == (41, 256, 256, 3) and arrays[role].dtype == np.uint8,
                        "missing true41-frame camera future",
                    )
                numeric(arrays["states"][frame : frame + 41], "future states")
                numeric(arrays["actions"][frame : frame + 40], "future actions")
                exact(
                    arrays["state_mask"][frame : frame + 41],
                    np.broadcast_to(ACTIVE, (41, 128)),
                    "future state valid mask",
                )
                exact(
                    arrays["action_mask"][frame : frame + 40],
                    np.broadcast_to(ACTIVE, (40, 128)),
                    "future action valid mask",
                )
                exact(arrays["action_valid"][frame : frame + 40], np.ones(40), "future temporal validity")
                exact(anchor["measured_state128"], arrays["states"][frame], "current measured anchor")
                evidence.check(anchor["image_path"], anchor["image_sha256"])
                request = anchor["request"]
                require(anchor["original_split"] == original_split, "anchor original development split changed")
                request_check(request, sample["task"], anchor["image_path"])
                require(
                    request["prompt"] == request["extra"]["action128"]["task_title"] == sample["task"]
                    and request["vision_path"] == anchor["image_path"],
                    "wrong task-only observation route",
                )
                require(
                    request["extra"]["action128"]["action_domain"] == DOMAIN
                    and request["extra"]["action128"]["action_dim_mask"] == ACTIVE.tolist(),
                    "wrong actual request domain/mask",
                )
                require(anchor["name"] not in index, "duplicate anchor name")
                index[anchor["name"]] = anchor
    expected_teacher = [
        dict(call_index=i, anchor_name=a["name"], split=a["split"], sigma=sigma, noise_seed=42)
        for i, (a, sigma) in enumerate((a, s) for a in prepared["anchors"] for s in (0.2, 0.5, 0.8))
    ]
    require(prepared["teacher_calls"] == expected_teacher, "wrong144 teacher roster/order")
    return prepared, index, saved["action128_spec"]


def normalized_source(anchor, transform, evidence):
    with np.load(evidence.check(anchor["arrays_path"], anchor["arrays_sha256"]), allow_pickle=False) as source:
        frame = anchor["anchor_frame"]
        values = source["actions"][frame : frame + 40].astype(np.float32)
    mask = np.asarray(transform["normalization_mask"], bool)
    scale, bias = (np.asarray(transform[name], np.float32) for name in ("scale", "bias"))
    require(mask.shape == scale.shape == bias.shape == (128,), "wrong normalization dimensions")
    values[:, mask] = values[:, mask] * scale[mask] + bias[mask]
    values[:, mask] = np.clip(values[:, mask], transform["clip_min"], transform["clip_max"])
    values[:, ~ACTIVE] = 0
    return values


def context_check(context):
    require(
        context == dict(domain=DOMAIN, normalization_id=31, projection_id=0, action_dim_mask=ACTIVE.tolist()),
        "actual full-domain context differs",
    )


def rng_signature(reference, evidence):
    metadata, arrays = bundle(reference, evidence)
    require(
        metadata["cuda_devices"] == 1 and set(arrays) == {"numpy_keys", "torch_cpu", "torch_cuda_0"},
        "single-device RNG evidence differs",
    )
    content = {key: value for key, value in metadata.items() if key not in ("arrays", "array_index")}
    content["array_bytes_sha256"] = {
        name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()
    }
    return content


def check_feature_arrays(arrays, metadata, normalized, sigma):
    for name, shape in dict(
        hidden=(40, 1280),
        head_base=(40, 128),
        base=(40, 128),
        target=(40, 128),
        xt=(40, 128),
        x0_normalized=(40, 128),
        epsilon=(40, 128),
        valid=(40, 128),
        projection_ids=(40,),
    ).items():
        require(arrays[name].shape == shape, f"actual feature {name} geometry differs")
    for name in ("hidden", "base"):
        require(metadata["original_dtypes"][name] == "torch.bfloat16", f"actual {name} dtype differs")
        exact(arrays[name], quantize_bfloat16(arrays[name]), f"lossless actual {name} BF16 export")
    exact(arrays["head_base"][:, :10], arrays["base"][:, :10], "original head active output versus masked velocity")
    require(not np.any(arrays["base"][:, 10:]), "final inactive velocity is nonzero")
    require(arrays["valid"].dtype == np.bool_, "valid mask is not boolean")
    exact(arrays["valid"], np.broadcast_to(ACTIVE, (40, 128)), "actual teacher active10 valid mask")
    require(arrays["projection_ids"].dtype == np.int64, "head projection IDs are not int64")
    exact(arrays["projection_ids"], np.zeros(40), "actual per-token shared projection IDs")
    exact(arrays["x0_normalized"], normalized, "actual normalized true future labels")
    exact(arrays["condition_action"], np.zeros((40, 1)), "actual no conditioned action steps")
    exact(arrays["sigma_action"], np.full((40, 1), np.float32(sigma)), "actual action sigma clock")
    epsilon = arrays["epsilon"].astype(np.float32)
    target = epsilon - normalized
    target[:, ~ACTIVE] = 0
    xt = quantize_bfloat16(epsilon * np.float32(sigma) + normalized * (np.float32(1) - np.float32(sigma)))
    xt[:, ~ACTIVE] = 0
    # Same registered024 flow tolerances; exact B/C and promotion comparisons
    # never use them. The target and model input remain separately inspected.
    for name, expected in (("target", target), ("xt", xt)):
        require(
            np.allclose(arrays[name], expected, rtol=1e-6, atol=1e-7),
            f"independent teacher {name} flow equation differs",
        )


def verify_features(prepared, anchors, spec, path, evidence):
    actual = evidence.json(path)
    require(
        actual["schema_version"] == "s2a.frozen-readout-features.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual,
        "partial/rejected feature phase",
    )
    prepared_ref = dict(
        path=str(Path(prepared["output_root"]) / "preparation/prepared.json"),
        sha256=digest(Path(prepared["output_root"]) / "preparation/prepared.json"),
    )
    require(actual["prepared_ref"] == prepared_ref, "features bound to another preparation")
    require(
        actual["denoise_calls"] == 249 and len(actual["teacher_calls"]) == 144 and len(actual["zero_identity"]) == 3,
        "wrong phase1 exact249 boundary budget",
    )
    require(
        actual["physics_steps"] == actual["fresh_cases"] == 0 and actual["all_original_parameters_frozen"] is True,
        "model freeze/physical budget differs",
    )
    require(
        actual["runtime"]["checkpoint_path"] == prepared["checkpoint"] and actual["runtime"]["ema_weights"] is False,
        "actual model weights differ",
    )
    labels = [f"zero:{i}" for i in range(3) for _ in range(35)] + [f"teacher:{i}" for i in range(144)]
    require(
        actual["denoise_ledger"] == [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        "phase1 actual denoise boundary ledger differs",
    )
    for reference, planned in zip(actual["zero_identity"], prepared["zero_identity_cases"], strict=True):
        response = evidence.ref(reference)
        original = evidence.json(planned["cache"]["response_path"], planned["cache"]["response_sha256"])
        left, right = (r["outputs"][0]["content"] for r in (response, original))
        for name in ("action", "normalized_action"):
            a = left["action"] if name == "action" else left["action128"][name]
            b = right["action"] if name == "action" else right["action128"][name]
            require(np.asarray(a).shape == (40, 128), "identity must retain full40×128 output")
            exact(a, b, f"zero readout complete {name} reproduction")
        folder = Path(reference["path"]).parent
        require(evidence.json(folder / "request.json") == planned["request"], "zero identity request changed")
        condition = evidence.json(folder / "conditioning.json")
        context_check(condition["context"])
        caption = json.loads(condition["caption"])
        require(
            caption["task_title"] == caption["command"] == planned["sample"]["task"],
            "identity actual task-only caption differs",
        )
    collected = {key: [] for key in ("hidden", "base", "target")}
    rows, before_rng, after_rng = [], None, None
    for planned, recorded in zip(prepared["teacher_calls"], actual["teacher_calls"], strict=True):
        require({key: recorded[key] for key in planned} == planned, "teacher actual roster/order differs")
        anchor = anchors[recorded["anchor_name"]]
        metadata, arrays = bundle(recorded["bundle"], evidence)
        require(all(metadata[key] == recorded[key] for key in planned), "teacher bundle identity differs")
        condition = evidence.ref(metadata["conditioning_ref"])
        context_check(condition["context"])
        require(
            condition["anchor_name"] == anchor["name"]
            and condition["source_frame"] == anchor["anchor_frame"]
            and condition["arrays_ref"] == dict(path=anchor["arrays_path"], sha256=anchor["arrays_sha256"]),
            "teacher conditioning source clock differs",
        )
        caption = json.loads(condition["caption"])
        require(
            caption["task_title"] == caption["command"] == anchor["task"]
            and caption.get("task_plan", []) == []
            and "current_subtask" not in caption,
            "teacher used non-task-only/phase condition",
        )
        normalized = normalized_source(anchor, spec["transforms"][31], evidence)
        check_feature_arrays(arrays, metadata, normalized, recorded["sigma"])
        current_before, current_after = (
            rng_signature(metadata["rng_before"], evidence),
            rng_signature(metadata["rng_after"], evidence),
        )
        if before_rng is None:
            before_rng, after_rng = current_before, current_after
        require(
            current_before == before_rng and current_after == after_rng,
            "teacher seed42 full RNG or consumed stream length differs",
        )
        for key in collected:
            collected[key].append(arrays[key])
        rows.extend(
            dict(
                episode=anchor["episode"],
                anchor_name=anchor["name"],
                sigma=recorded["sigma"],
                split=anchor["split"],
                task_index=anchor["task_index"],
                endpoint=i,
            )
            for i in range(40)
        )
    require(
        len(rows) == 5760 and all(sum(r["split"] == split for r in rows) == 2880 for split in ("train", "validation")),
        "wrong complete teacher row split",
    )
    return dict(
        features_verified=True,
        zero_identity_generations_verified=3,
        teacher_calls_verified=144,
        actual_denoise_calls_verified=249,
    ), dict(
        arrays={key: np.concatenate(value) for key, value in collected.items()},
        rows=rows,
        features_ref=dict(path=str(Path(path).resolve()), sha256=digest(path)),
    )


def hierarchical_weights(rows):
    require(len(rows) > 0 and all(r["split"] == "train" for r in rows), "validation cannot enter fit weights")
    episodes = sorted({r["episode"] for r in rows})
    weights = np.empty(len(rows), np.float64)
    for episode in episodes:
        episode_indices = [i for i, row in enumerate(rows) if row["episode"] == episode]
        anchor_names = sorted({rows[i]["anchor_name"] for i in episode_indices})
        for anchor in anchor_names:
            anchor_indices = [i for i in episode_indices if rows[i]["anchor_name"] == anchor]
            sigmas = sorted({rows[i]["sigma"] for i in anchor_indices})
            for sigma in sigmas:
                indices = [i for i in anchor_indices if rows[i]["sigma"] == sigma]
                require(len({rows[i]["endpoint"] for i in indices}) == len(indices), "duplicate endpoint in fit cell")
                weights[indices] = 1 / (len(episodes) * len(anchor_names) * len(sigmas) * len(indices))
    return weights


def metric_group(predictions, target, selected):
    require(len(selected) > 0, "empty teacher metric group")
    squared = {
        name: (value[selected, :10].astype(np.float64) - target[selected, :10].astype(np.float64)) ** 2
        for name, value in predictions.items()
    }
    return dict(
        rows=len(selected),
        active_elements=10 * len(selected),
        mse={name: float(value.mean()) for name, value in squared.items()},
        per_slot_mse={name: value.mean(axis=0).tolist() for name, value in squared.items()},
    )


def all_teacher_metrics(predictions, target, rows):
    result = {}
    for split in ("train", "validation"):
        indices = [i for i, row in enumerate(rows) if row["split"] == split]
        grouped = dict(pooled=metric_group(predictions, target, indices))
        for label, field in (("by_sigma", "sigma"), ("by_episode", "episode"), ("by_task", "task_index")):
            grouped[label] = {
                str(value): metric_group(predictions, target, [i for i in indices if rows[i][field] == value])
                for value in sorted({rows[i][field] for i in indices})
            }
        result[split] = grouped
    return result


def compare_metrics(saved, recomputed, label="teacher metrics"):
    if isinstance(recomputed, dict):
        require(isinstance(saved, dict) and set(saved) == set(recomputed), f"{label}: group roster differs")
        for key in recomputed:
            compare_metrics(saved[key], recomputed[key], f"{label}.{key}")
    elif isinstance(recomputed, int):
        require(type(saved) is int and saved == recomputed, f"{label}: row/element count differs")
    else:
        close(saved, recomputed, label)


def independent_teacher_gate(metrics):
    validation = metrics["validation"]
    pooled = validation["pooled"]["mse"]
    checks = dict(
        pooled_improvement=pooled["adapt"] <= 0.95 * pooled["raw"],
        beats_constant_bias=pooled["adapt"] < pooled["bias"],
        all_sigmas_nonregressing=all(
            group["mse"]["adapt"] <= 1.05 * group["mse"]["raw"] for group in validation["by_sigma"].values()
        ),
    )
    return dict(passed=all(checks.values()), checks=checks)


def verify_fit(prepared, features, path, evidence):
    actual = evidence.json(path)
    require(
        actual["schema_version"] == "s2a.frozen-readout-fit.v1"
        and actual["complete"] is True
        and actual["passed"] is True
        and "error" not in actual,
        "partial/rejected two-fit stage",
    )
    require(
        actual["features_ref"] == features["features_ref"]
        and actual["producer_fit_groups"] == actual["fit_attempts"] == 2,
        "fit changed feature source or fixed two-fit budget",
    )
    expected_prepared = Path(prepared["output_root"]) / "preparation/prepared.json"
    require(
        actual["prepared_ref"] == dict(path=str(expected_prepared), sha256=digest(expected_prepared)),
        "fit bound to another prepared source",
    )
    require(
        actual["ridge"] == 0.1 and actual["deployment_dtype"] == "torch.bfloat16",
        "fit regularizer/deployment precision changed",
    )
    require(
        actual["new_model_calls"] == actual["physics_steps"] == 0
        and actual["cuda_visible_devices"] == actual["ld_library_path"] == "",
        "fit ran outside CPU-only contract",
    )
    require(
        actual["training_episodes"] == prepared["config"]["train_episodes"]
        and actual["validation_episodes"] == prepared["config"]["validation_episodes"],
        "fit episode split differs",
    )
    arrays, rows = features["arrays"], features["rows"]
    require(
        actual["teacher_rows"] == rows and actual["fitted_rows"] == actual["validation_rows"] == 2880,
        "fit endpoint provenance differs",
    )
    selected = np.asarray([r["split"] == "train" for r in rows])
    weights = hierarchical_weights([r for r in rows if r["split"] == "train"])
    residual = arrays["target"][selected, :10].astype(np.float64) - arrays["base"][selected, :10].astype(np.float64)
    recomputed = augmented_fit(arrays["hidden"][selected], residual, weights)
    coefficients = {}
    for name in ("adapt", "bias"):
        reference = actual["models"][name]
        with np.load(evidence.check(reference["path"], reference["sha256"]), allow_pickle=False) as archive:
            require(archive.files == ["weights"], "saved model has unregistered parameters")
            value = numeric(archive["weights"], name).copy()
        require(value.shape == (1281, 10) and value.dtype == np.float64, "saved coefficients must be1281×10 FP64")
        close(value, recomputed[name], f"independent augmented-LS {name}")
        coefficients[name] = value
    predictions = dict(raw=arrays["base"].copy())
    predictions.update(
        {name: apply_quantized(arrays["hidden"], arrays["base"], value) for name, value in coefficients.items()}
    )
    reference = actual["teacher_predictions"]
    with np.load(evidence.check(reference["path"], reference["sha256"]), allow_pickle=False) as archive:
        require(
            set(archive.files) == {"raw", "adapt", "bias", "training_weights"},
            "teacher prediction artifact roster differs",
        )
        exact(archive["training_weights"], weights, "fit hierarchy weights")
        for name, value in predictions.items():
            exact(archive[name], value, f"actual single-cast BF16 teacher {name}")
            exact(archive[name][:, 10:], arrays["base"][:, 10:], f"{name} inactive118 unchanged")
    metrics = all_teacher_metrics(predictions, arrays["target"], rows)
    compare_metrics(actual["teacher_metrics"], metrics)
    gate = independent_teacher_gate(metrics)
    require(
        actual["teacher_gate"] == gate and actual["eligible_for_sampling"] is gate["passed"],
        "strict teacher promotion gate differs",
    )
    for filename, expected in actual["input_sha256"].items():
        evidence.check(filename, expected)
    episode_sigma = {
        split: {
            str(episode): {
                str(sigma): metric_group(
                    predictions,
                    arrays["target"],
                    [
                        i
                        for i, row in enumerate(rows)
                        if row["split"] == split and row["episode"] == episode and row["sigma"] == sigma
                    ],
                )
                for sigma in (0.2, 0.5, 0.8)
            }
            for episode in sorted({row["episode"] for row in rows if row["split"] == split})
        }
        for split in ("train", "validation")
    }
    return dict(
        independent_model_reconstructions=2,
        fit_verified=True,
        teacher_gate=gate,
        eligible_for_sampling=gate["passed"],
        teacher_metrics=metrics,
        independent_episode_sigma_metrics=episode_sigma,
        raw_checkpoint_task_success_proven=False,
    )


def frozen_integrator(evidence):
    """Share only frozen011's independent SciPy integration and hold/CV rules."""
    evidence.check(REPO / "scripts/audit_wam_sampler_steps.py", INTEGRATOR_SHA)
    evidence.check(REPO / "scripts/audit_wam_earlier_screen.py", ROTATION_SHA)
    script_dir = str(REPO / "scripts")
    sys.path.insert(0, script_dir)
    try:
        return importlib.import_module("audit_wam_sampler_steps").integrate_and_score
    finally:
        sys.path.remove(script_dir)


def response_actions(response, request, checkpoint, transform):
    require(response["status"] == "success" and len(response["outputs"]) == 1, "unsuccessful sampling response")
    content = response["outputs"][0]["content"]
    metadata = content["action128"]
    raw, normalized = (
        numeric(content["action"], "raw sampling actions"),
        numeric(metadata["normalized_action"], "normalized sampling actions"),
    )
    require(raw.shape == normalized.shape == (40, 128), "sampling did not retain full40×128 actions")
    require(not np.any(raw[:, 10:]) and not np.any(normalized[:, 10:]), "sampling inactive118 is nonzero")
    require(
        metadata["checkpoint_path"] == checkpoint
        and metadata["seed"] == request["seed"]
        and metadata["action_domain"] == DOMAIN
        and metadata["action_dim_mask"] == ACTIVE.tolist(),
        "sampling response checkpoint/seed/full-domain/mask differs",
    )
    require(
        metadata["action_units"] == "raw_physical_or_source_units" and metadata["action_fps"] == 20,
        "sampling output units/clock differ",
    )
    exact(metadata["action_timestamps_seconds"], np.arange(40) / 20, "sampling endpoint clock")
    decoded = normalized.astype(np.float32)
    mask = np.asarray(transform["normalization_mask"], bool)
    scale, bias = (np.asarray(transform[key], np.float32) for key in ("scale", "bias"))
    decoded[:, mask] = (decoded[:, mask] - bias[mask]) / scale[mask]
    require(np.allclose(decoded, raw, rtol=0, atol=2e-7), "independent full128 denormalization differs")
    for key in (
        "name",
        "seed",
        "prompt",
        "vision_path",
        "extra",
        "fps",
        "num_frames",
        "num_steps",
        "guidance",
        "shift",
        "action_chunk_size",
        "model_mode",
        "image_size",
        "domain_name",
    ):
        require(response["args"][key] == request[key], f"actual sampling request {key} differs")
    require(
        response["args"]["condition_frame_indexes_vision"] == [0]
        and response["args"]["action_path"] is None
        and response["args"]["native_prompt_upsampling"] is False
        and not response["args"]["negative_prompt"],
        "unexpected sampling conditioning/upsampling branch",
    )
    return raw


def initial_signature(metadata, arrays):
    noise, mask, latent = (arrays[name] for name in ("initial_noise", "condition_mask", "first_condition_latent"))
    require(
        noise.dtype == mask.dtype == latent.dtype == np.float32, "actual inference accumulation/condition is not FP32"
    )
    require(latent.shape == (1, 48, 1, 60, 40), "actual first condition latent geometry differs")
    vision_shape = (1, 48, 11, 60, 40)
    offset = int(np.prod(vision_shape))
    require(noise.shape == mask.shape == (offset + 5120,), "actual flattened noise/condition clock differs")
    expected_mask = np.zeros_like(mask)
    expected_mask[:offset].reshape(vision_shape)[:, :, :1] = 1
    exact(mask, expected_mask, "actual first-frame-only inference condition mask")
    exact(noise[:offset].reshape(vision_shape)[:, :, :1], latent, "initial noise retains actual conditioning prefix")
    require(not np.any(noise[offset:].reshape(40, 128)[:, 10:]), "initial inactive action noise is nonzero")
    return dict(
        arrays={name: value.copy() for name, value in arrays.items()},
        caption=metadata["caption"],
        text_tokens=metadata["text_tokens"],
    )


def pool_state_rows(rows):
    result = {}
    for prefix in (8, 16, 40):
        methods = {}
        for method in ("raw", "adapt", "hold", "constant_velocity"):
            selected = [row for row in rows if row["variant"] == (method if method in ("raw", "adapt") else "raw")]
            field = "wam" if method in ("raw", "adapt") else method
            errors = {
                name: np.concatenate([np.asarray(row["errors"][field][name][:prefix], np.float64) for row in selected])
                for name in ("position", "rotation", "opening")
            }
            p, r, g = (errors[key] for key in ("position", "rotation", "opening"))
            require(p.shape == r.shape == g.shape and len(p) > 0, "state metric valid endpoint group differs")
            methods[method] = dict(
                valid_endpoints=len(p),
                position_rmse_m=float(np.sqrt(np.mean(p**2))),
                rotation_rmse_rad=float(np.sqrt(np.mean(r**2))),
                opening_mae=float(np.abs(g).mean()),
                joint_score=float(np.mean((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2)),
            )
        result[str(prefix)] = methods
    return result


def independent_state_gate(groups):
    pooled = groups["pooled"]
    short_improvement = all(
        pooled[str(h)]["adapt"]["joint_score"] <= 0.90 * pooled[str(h)]["raw"]["joint_score"] for h in (8, 16)
    )
    beats_baselines = all(
        pooled[str(h)]["adapt"]["joint_score"] < pooled[str(h)][baseline]["joint_score"]
        for h in (8, 16)
        for baseline in ("hold", "constant_velocity")
    )
    component_checks = {}
    for h in (8, 16, 40):
        raw, adapted = pooled[str(h)]["raw"], pooled[str(h)]["adapt"]
        for component, additive in (("position_rmse_m", 0.002), ("rotation_rmse_rad", 0.01), ("opening_mae", 0.01)):
            component_checks[f"h{h}_{component}"] = (
                adapted[component] <= 1.05 * raw[component] and adapted[component] <= raw[component] + additive
            )
    episode_passes = {
        episode: all(
            value[str(h)]["adapt"]["joint_score"] <= 1.05 * value[str(h)]["raw"]["joint_score"] for h in (8, 16)
        )
        for episode, value in groups["by_episode"].items()
    }
    no_episode_outlier = all(
        value[str(h)]["adapt"]["joint_score"] <= 1.25 * value[str(h)]["raw"]["joint_score"]
        for value in groups["by_episode"].values()
        for h in (8, 16)
    )
    seeds_pass = all(
        value[str(h)]["adapt"]["joint_score"] <= 1.05 * value[str(h)]["raw"]["joint_score"]
        for value in groups["by_seed"].values()
        for h in (8, 16)
    )
    checks = dict(
        short_joint_improvement=short_improvement,
        strictly_beats_hold_and_constant_velocity=beats_baselines,
        all_components_protected=all(component_checks.values()),
        at_least_five_episodes=sum(episode_passes.values()) >= 5,
        no_episode_short_outlier=no_episode_outlier,
        both_seeds_protected=seeds_pass,
    )
    return dict(
        passed=all(checks.values()), checks=checks, component_checks=component_checks, episode_passes=episode_passes
    )


def sampling_execution_contract(actual, prepared, fit):
    require(actual["schema_version"] == "s2a.frozen-readout-sampling.v1", "wrong sampling execution schema")
    require(
        actual["all_original_parameters_frozen"] is True and actual["head_call_counts"] == dict(raw=1680, adapt=1680),
        "sampling original parameters/readout invocation counts differ",
    )
    require(actual["physics_steps"] == actual["fresh_cases"] == 0, "sampling used physical/fresh performance budget")
    require(
        actual["adapted_model_ref"] == fit["models"]["adapt"], "sampling did not bind the admitted adapted coefficients"
    )
    require(
        actual["runtime"]["checkpoint_path"] == prepared["checkpoint"] and actual["runtime"]["ema_weights"] is False,
        "sampling original checkpoint/EMA contract differs",
    )


def initial_record_identity(metadata, planned):
    for key in ("generation_index", "name", "pair_id", "variant", "sampling_seed"):
        require(metadata[key] == planned[key], f"actual initial {key} is bound to another generation")
    require(metadata["inference_only_current_image"] is True, "initial inference used a noncurrent observation")


def verify_sampling(prepared, anchors, spec, fit_path, fit_verification_path, path, evidence):
    actual = evidence.json(path)
    require(
        actual["complete"] is True and actual["passed"] is True and "error" not in actual,
        "partial/rejected sampling stage",
    )
    require(
        actual["fit_ref"] == dict(path=str(Path(fit_path).resolve()), sha256=digest(fit_path)),
        "sampling consumed another fit",
    )
    fit_verified = evidence.json(fit_verification_path)
    fit = evidence.json(fit_path)
    require(
        fit["complete"] is True and fit["passed"] is True and fit["eligible_for_sampling"] is True,
        "sampling producer teacher gate was not passed",
    )
    sampling_execution_contract(actual, prepared, fit)
    require(
        fit_verified["phase"] == "fit"
        and fit_verified["passed"] is True
        and fit_verified["fit_verified"] is True
        and fit_verified["eligible_for_sampling"] is True,
        "independent teacher gate did not authorize sampling",
    )
    require(
        actual["fit_verification_ref"]
        == dict(path=str(Path(fit_verification_path).resolve()), sha256=digest(fit_verification_path)),
        "sampling fit verification reference differs",
    )
    require(
        fit_verified["input_sha256"].get(str(Path(fit_path).resolve())) == digest(fit_path),
        "sampling teacher admission is not bound to actual fit result",
    )
    for filename, expected in fit_verified["input_sha256"].items():
        evidence.check(filename, expected)
    expected_prepared = Path(prepared["output_root"]) / "preparation/prepared.json"
    require(
        actual["prepared_ref"] == dict(path=str(expected_prepared), sha256=digest(expected_prepared)),
        "sampling bound to another prepared source",
    )
    require(fit_verified["prepared"] == actual["prepared_ref"], "sampling teacher admission used another preparation")
    require(
        actual["denoise_calls"] == 3360 and len(actual["generations"]) == 96, "sampling fixed96/3360 budget differs"
    )
    expected_ledger = [
        dict(index=i * 35 + j, label=f"sampling:{i}", completed=True) for i in range(96) for j in range(35)
    ]
    require(actual["denoise_ledger"] == expected_ledger, "sampling actual denoise boundary ledger differs")
    integrate = frozen_integrator(evidence)
    pairs, rows = {}, []
    for planned, recorded in zip(prepared["validation_generations"], actual["generations"], strict=True):
        require({key: recorded[key] for key in planned} == planned, "actual sampling roster/order changed")
        anchor = anchors[planned["anchor_name"]]
        require(anchor["split"] == "validation", "training source entered held development sampling")
        response, condition = evidence.ref(recorded["response"]), evidence.ref(recorded["conditioning"])
        context_check(condition["context"])
        caption = json.loads(condition["caption"])
        require(
            caption["task_title"] == caption["command"] == anchor["task"]
            and caption.get("task_plan", []) == []
            and "current_subtask" not in caption,
            "sampling actual task-only condition differs",
        )
        require(
            response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"]
            == hashlib.sha256(condition["caption"].encode()).hexdigest(),
            "sampling response/actual caption binding differs",
        )
        actions = response_actions(response, planned["request"], prepared["checkpoint"], spec["transforms"][31])
        initial_meta, initial = bundle(recorded["initial"], evidence)
        initial_record_identity(initial_meta, planned)
        signature = initial_signature(initial_meta, initial)
        require(signature["caption"] == condition["caption"], "actual initial text differs from response condition")
        pair_id = planned["pair_id"]
        if planned["variant"] == "raw":
            require(pair_id not in pairs, "duplicate raw initial pair")
            pairs[pair_id] = signature
        else:
            require(
                pair_id in pairs
                and signature["caption"] == pairs[pair_id]["caption"]
                and signature["text_tokens"] == pairs[pair_id]["text_tokens"],
                "adapt text tokens/caption did not pair with raw",
            )
            require(set(signature["arrays"]) == set(pairs[pair_id]["arrays"]), "pair initial tensor roster differs")
            for name, value in signature["arrays"].items():
                exact(value, pairs[pair_id]["arrays"][name], f"raw/adapt actual initial {name}")
        with np.load(evidence.check(anchor["arrays_path"], anchor["arrays_sha256"]), allow_pickle=False) as source:
            count, errors, _ = integrate(np.asarray(source["states"], np.float64), anchor, actions)
        require(count == 40, "sampling has incomplete common state endpoints")
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
    require(len(pairs) == 48, "wrong48 independent initial observation/seed pairs")
    groups = dict(pooled=pool_state_rows(rows))
    for label, field in (("by_episode", "episode"), ("by_task", "task_index"), ("by_seed", "sampling_seed")):
        groups[label] = {
            str(value): pool_state_rows([row for row in rows if row[field] == value])
            for value in sorted({row[field] for row in rows})
        }
    require(
        len(groups["by_episode"]) == 6 and set(groups["by_seed"]) == {"42", "43"},
        "sampling episode/seed coverage differs",
    )
    gate = independent_state_gate(groups)
    return dict(
        sampling_verified=True,
        generations_verified=96,
        paired_initial_states_verified=48,
        sampling_state_metrics=groups,
        sampling_gate=gate,
        eligible_for_development_physics=gate["passed"],
        raw_checkpoint_task_success_proven=False,
        physics_tasks_executed=0,
    )


def phase_arguments(args):
    fields = ("features", "fit", "fit_verification", "sampling")
    expected = {
        "prepare": (False, False, False, False),
        "features": (True, False, False, False),
        "fit": (True, True, False, False),
        "sampling": (True, True, True, True),
    }
    require(
        tuple(getattr(args, key) is not None for key in fields) == expected[args.phase],
        "actual evidence arguments do not match the explicit audit phase",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "features", "fit", "sampling"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--fit-verification", type=Path)
    parser.add_argument("--sampling", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    phase_arguments(args)
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "025 verifier requires CPU emptyCUDA/LD and seed42",
    )
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "new independent output must be under outputs; refuse overwrite",
    )
    evidence = Evidence()
    receipt = dict(
        schema_version="s2a.frozen-readout-independent-verification.v1",
        phase=args.phase,
        passed=False,
        new_model_calls=0,
        new_physics_steps=0,
        new_candidate_fits=0,
    )
    error = None
    try:
        prepared, anchors, spec = verify_preparation(args.prepared, evidence)
        receipt["prepared"] = dict(path=str(args.prepared.resolve()), sha256=digest(args.prepared))
        if args.phase != "prepare":
            require(args.features is not None, "features/fit phase requires actual features result")
            result, rows = verify_features(prepared, anchors, spec, args.features, evidence)
            receipt.update(result)
            if args.phase in ("fit", "sampling"):
                require(args.fit is not None, "fit phase requires actual fit result")
                if args.phase == "fit":
                    receipt.update(verify_fit(prepared, rows, args.fit, evidence))
                else:
                    require(
                        args.fit_verification is not None and args.sampling is not None,
                        "sampling phase requires immutable fit verification and actual sampling result",
                    )
                    receipt.update(
                        verify_sampling(
                            prepared, anchors, spec, args.fit, args.fit_verification, args.sampling, evidence
                        )
                    )
        else:
            require(args.features is None and args.fit is None, "prepare phase cannot consume model/fit results")
            receipt.update(
                preparation_verified=True,
                episodes_verified=12,
                anchors_verified=48,
                actual_model_results_verified=False,
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
    print(json.dumps(dict(passed=receipt["passed"], output=str(output), sha256=digest(output))))
    if error is not None:
        raise SystemExit(f"025 independent audit rejected: {error}")


if __name__ == "__main__":
    main()
