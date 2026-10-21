"""Freeze025's old-episode readout inputs after the completed024 admission.

CPU preparation reads only already extracted episodes and metadata. It never
imports Cosmos, loads a model, fits parameters or creates a simulator.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from s2a.artifacts import REPO, sha256
from s2a.libero_simulation import canvas
from s2a.worker_handshake import birth, write_once

CONFIG = REPO / "configs/wam_frozen_readout_20261021.json"
CONFIG_SHA = "6534245a114d89167f6d8cd283e55de8685b4a02abca79706436309477860605"
PLAN = REPO / "plans/v0.0.1/025_frozen_wam_readout_20261021.md"
PLAN_SHA = "46b766356ef29c593c895a23130a0415cc83c892afcfc6bb66e07e5bab540dc6"
ROUNDING = REPO / "plans/v0.0.1/025_bfloat16_rounding_addendum_20261021.md"
ROUNDING_SHA = "210983ca27d51aaf4acdb0f6d179ffe991dd34a0ee04fbc9c1c09f930fc9f498"
QUANTIZATION = REPO / "outputs/wam_stability_20261021/readout_quantization_gpu6"
FRESH = REPO / "configs/wam_stability_fresh_reservation_20261021.json"
ACTIVE = np.asarray([True] * 10 + [False] * 118)
DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
LOCAL_SOURCES = (
    "scripts/prepare_wam_readout.py",
    "tests/test_wam_readout_preparation.py",
    "src/s2a/wam_readout.py",
    "tests/test_wam_readout.py",
    "scripts/run_wam_readout_features.py",
    "scripts/fit_wam_readout.py",
    "tests/test_wam_readout_fit.py",
    "scripts/run_wam_readout_sampling.py",
    "tests/test_wam_readout_sampling.py",
    "scripts/run_wam_readout.py",
    "tests/test_wam_readout_runner.py",
    "scripts/verify_wam_readout.py",
    "tests/test_wam_readout_verification.py",
    "scripts/audit_wam_sampler_steps.py",
    "scripts/audit_wam_earlier_screen.py",
    "plans/v0.0.1/025_bfloat16_rounding_addendum_20261021.md",
    "src/s2a/worker_handshake.py",
    "src/s2a/wam_runtime.py",
    "src/s2a/libero_simulation.py",
    "src/s2a/geometry.py",
    "src/s2a/artifacts.py",
    "scripts/audit_wam_current_numerics.py",
    "scripts/verify_wam_current_numerics.py",
)
EXTERNAL_SOURCES = (
    "model/generator/mot/action128.py",
    "model/generator/mot/cosmos3_vfm_network.py",
    "model/generator/mot/action_tower.py",
    "model/generator/mot/small_action_tower.py",
    "data/generator/action/projection_routing.py",
    "data/generator/action/unified_action128.py",
    "data/generator/action/utils/action_processing.py",
    "model/generator/omni_mot_model.py",
    "model/generator/utils/data_and_condition.py",
    "model/generator/reasoner/qwen3_vl/utils.py",
    "model/generator/algorithm/loss/flow_matching.py",
    "model/generator/algorithm/loss/time_weight.py",
    "model/generator/diffusion/rectified_flow.py",
    "model/generator/tokenizers/wan2pt2_vae_4x16x16.py",
    "model/generator/tokenizers/interface.py",
    "data/generator/sequence_packing/static_layout.py",
    "data/generator/sequence_packing/sequence.py",
    "data/generator/sequence_packing/modality.py",
    "inference/action_server.py",
    "inference/action_serving.py",
    "inference/args.py",
    "inference/inference.py",
    "inference/common/init.py",
    "utils/misc.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        value = sha256(path)
        require(expected is None or value == expected, f"frozen source SHA differs: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == value, f"source changed: {path}")
        self.hashes[str(path)] = value
        return path

    def json(self, path, expected=None):
        return json.loads(self.check(path, expected).read_text())

    def recheck(self):
        for path, digest in self.hashes.copy().items():
            self.check(path, digest)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def uniform_anchors(length, horizon=40, count=4):
    require(type(length) is int and type(horizon) is int and type(count) is int, "integer source clock required")
    require(horizon == 40 and count == 4 and length >= horizon + count, "four distinct completeH40 anchors required")
    anchors = [j * (length - horizon - 1) // (count - 1) for j in range(count)]
    require(len(set(anchors)) == count and anchors[-1] + horizon + 1 == length, "source endpoint clock differs")
    return anchors


def split_episodes(config, cases):
    """Whole-source splits, rejecting inconsistent aliases and source reuse."""
    train, validation = config["train_episodes"], config["validation_episodes"]
    require(len(set(train)) == len(train) and len(set(validation)) == len(validation), "duplicate episode split")
    require(not set(train) & set(validation), "train/validation episode overlap")
    selected, sources = [], set()
    for split, episodes, original in (("train", train, "diagnostic"), ("validation", validation, "study_heldout")):
        for episode in episodes:
            aliases = [s for s in cases if s["episode"] == episode]
            require(aliases, "registered whole episode is missing")
            first = aliases[0]
            fields = ("episode", "source_episode_id", "task", "task_index", "arrays_path", "arrays_sha256", "split")
            require(
                all({k: s[k] for k in fields} == {k: first[k] for k in fields} for s in aliases),
                "phase aliases disagree on whole-source identity",
            )
            require(first["split"] == original, "original development split changed")
            require(first["source_episode_id"] not in sources, "whole-source episode leaks between split entries")
            sources.add(first["source_episode_id"])
            selected.append(dict(sample=first, split=split))
    return selected


def validate_window(arrays, frame, horizon=40):
    require(type(frame) is int and frame >= 0 and horizon == 40, "invalid source frame/clock")
    states, mask = arrays["states"], arrays["state_mask"]
    require(states.ndim == 2 and states.shape[1] == 128 and mask.shape == states.shape, "source state128 shape differs")
    count = len(states)
    require(frame + horizon + 1 <= count, "missing complete41-frame future")
    require(np.isfinite(states[frame : frame + horizon + 1]).all(), "nonfinite measured/future state")
    require(
        mask.dtype == np.bool_
        and np.array_equal(mask[frame : frame + horizon + 1], np.broadcast_to(ACTIVE, (41, 128))),
        "missing measured/future active10 state mask",
    )
    action, valid, dimensions = arrays["actions"], arrays["action_valid"], arrays["action_mask"]
    require(
        action.shape == (count, 128) and valid.shape == (count,) and dimensions.shape == action.shape,
        "action clock differs",
    )
    require(valid.dtype == dimensions.dtype == np.bool_, "source masks must be boolean")
    require(valid[frame : frame + horizon].all(), "invalid future action row")
    require(
        np.array_equal(dimensions[frame : frame + horizon], np.broadcast_to(ACTIVE, (40, 128))),
        "future active10 action mask differs",
    )
    require(np.isfinite(action[frame : frame + horizon]).all(), "nonfinite future action")
    for role in ("auxiliary_0", "left_wrist"):
        images = arrays[role]
        require(
            images.shape == (count, 256, 256, 3) and images.dtype == np.uint8, "source camera41 clock/shape differs"
        )
    return dict(measured_state128=states[frame].tolist(), state_mask128=mask[frame].tolist(), valid_future_steps=40)


def validate_domain(config, spec):
    expected = dict(name=DOMAIN, normalization_id=31, projection_id=0, active_slots=10)
    require(config["domain"] == expected and config["hidden_size"] == 1280, "registered LIBERO domain/readout changed")
    registry = spec["registry"]
    require(registry["domains"].index(DOMAIN) == 31, "source normalization domain is not31")
    require(registry["domain_projection_ids"][31] == 0, "LIBERO projection is not sharedslot0")
    return dict(
        **expected,
        other_normalization_domains_sharing_projection0=sum(i == 0 for i in registry["domain_projection_ids"]) - 1,
        active_mask=ACTIVE.tolist(),
        normalization_mask_is_not_active_mask=True,
    )


def source_exclusion(sample, fresh):
    """Reuse conservative historical links without upgrading their evidence."""
    source = json.loads(sample["source_episode_id"])["source_id"]
    require(
        source.startswith("/share/mixed/projset/eai_pwm/datasets/action_processed_r1/libero/v3_eef_droid:"),
        "unknown source identity",
    )
    r1_episode = int(source.rsplit(":", 1)[1])
    records = [r for r in fresh["offline_source_exclusion_records"] if r["r1_episode"] == r1_episode]
    require(
        len(records) == 1 and records[0]["task_index"] == sample["task_index"], "missing conservative source exclusion"
    )
    record = records[0]
    require(record["demo"] in fresh["excluded_demos_by_task"][sample["task_index"]], "linked demo was not excluded")
    require(record["initial_sync_certified"] is False, "historical source evidence unexpectedly changed")
    return dict(record, r2_episode=sample["episode"], source_episode_id=sample["source_episode_id"])


def absent_owned_birth(item):
    try:
        actual = birth(item["pid"], allow_exited=True)
    except (FileNotFoundError, ProcessLookupError):
        return True
    return actual["start_ticks"] != item["start_ticks"] or actual["proc_state"] in ("Z", "X")


def quantization_gold_bits():
    """Exact, already-BF16-representable gold values; no production quantizer."""
    spacing = 2.0**-133
    first10 = {
        "normal_positive": [1, 1, 1 + 2**-7, 1 - 2**-8, 1 - 2**-8, 1 - 2**-8, 1 + 2**-7, 1 + 2**-6, 1 + 2**-6, 1],
        "normal_negative": [
            -1,
            -1,
            -1 - 2**-7,
            -1 + 2**-8,
            -1 + 2**-8,
            -1 + 2**-8,
            -1 - 2**-7,
            -1 - 2**-6,
            -1 - 2**-6,
            -1,
        ],
        "subnormal": [0.0, 0.0, spacing, -0.0, -0.0, -spacing, 2 * spacing, 2 * spacing, 2.0**-126, 0.0],
    }
    gold = {}
    for name, values in first10.items():
        array = np.empty((2, 128), dtype=np.float32)
        array[:, :10] = values
        array[:, 10:] = np.arange(118, dtype=np.float32)
        array[:, 10] = -0.0
        gold[name] = (array.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
    primitive = np.asarray(
        [0.0, -0.0, 0.0, 0.0, spacing, -0.0, -0.0, -spacing, 1 + 2**-7, -1 - 2**-7], dtype=np.float32
    )
    gold["primitive"] = (primitive.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
    return gold


def verify_quantization_arrays(arrays):
    gold = quantization_gold_bits()
    expected = {f"{name}_{device}" for name in gold for device in ("cpu", "cuda_0")}
    require(set(arrays) == expected, "quantization actual output roster differs")
    for name, values in gold.items():
        for device in ("cpu", "cuda_0"):
            actual = arrays[f"{name}_{device}"]
            require(
                actual.dtype == np.int16 and np.array_equal(actual, values),
                f"quantization actual BF16 gold bits differ: {name}/{device}",
            )
    return dict(actual_cpu_cuda_gold_bits_verified=True, full_readout_cases=3, primitive_values=10)


def quantization_admission(evidence):
    """Bind the actual GPU6 synthetic run, current core, gold bits and release."""
    root = QUANTIZATION
    result = evidence.json(root / "result.json")
    require(
        result.get("schema_version") == "s2a.readout-synthetic-cpu-cuda-quantization.v1"
        and result.get("complete") is True
        and result.get("passed") is True,
        "025 requires its completed actual GPU6 quantization result",
    )
    require(
        result["seed"] == 42
        and result["physical_gpu"] == 6
        and result["actual_cuda_index"] == 0
        and result["cuda_visible_devices"] == "6"
        and result["primitive_signed_zero_subnormal_midpoint_passed"] is True,
        "quantization GPU6/seed/numerical contract differs",
    )
    for key in ("model_weights_loaded", "wam_denoise_calls", "fit_groups", "episode_reads", "physics_steps"):
        require(result[key] == 0, f"quantization toy used unregistered real work: {key}")
    flags = {
        "bitwise_cpu_cuda_equal",
        "actual_hook_matches_apply",
        "other118_preserved",
        "zero_original_object",
        "other_domain_original_object",
        "input_tensors_unchanged",
    }
    require(
        [r["name"] for r in result["cases"]] == ["normal_positive", "normal_negative", "subnormal"],
        "quantization case roster differs",
    )
    require(
        all(set(r) == flags | {"name"} and all(r[k] is True for k in flags) for r in result["cases"]),
        "quantization hook/identity protection failed",
    )
    core_path = REPO / "src/s2a/wam_readout.py"
    evidence.check(core_path, result["core_sha256"])
    require(result["actual_outputs"]["path"] == str(root / "actual_outputs.npz"), "quantization output path differs")
    outputs = evidence.check(root / "actual_outputs.npz", result["actual_outputs"]["sha256"])
    with np.load(outputs, allow_pickle=False) as source:
        bits = verify_quantization_arrays({key: source[key] for key in source.files})
    launch = evidence.json(root / "launch.json")
    identity = evidence.json(root / "actual_child_identity.json")
    exit_record = evidence.json(root / "exit.json")
    release = evidence.json(root / "ownership/release.json")
    require(launch["core_sha256"] == result["core_sha256"], "quantization launched another readout core")
    evidence.check(root / "worker.py", result["worker_sha256"])
    require(launch["worker_sha256"] == result["worker_sha256"], "quantization worker source differs")
    evidence.check(root / "owner.py", launch["owner_sha256"])
    expected_environment = dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH="")
    require(launch["selected_environment"] == expected_environment, "quantization actual launch environment differs")
    require(
        len(launch["command"]) == 2 and Path(launch["command"][1]).resolve() == root / "worker.py",
        "quantization did not launch its recorded worker source",
    )
    require(
        all(identity[k] == launch["child"][k] for k in ("pid", "start_ticks", "ppid"))
        and identity["ppid"] == launch["owner"]["pid"]
        and identity["command"] == launch["command"]
        and identity["cwd"] == str(REPO)
        and all(identity["environment"][k] == v for k, v in expected_environment.items()),
        "quantization actual child birth/parent/command/environment differs",
    )
    require(
        exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0),
        "quantization has no actual successful wait",
    )
    require(
        all(release[k] == launch["child"][k] for k in ("pid", "start_ticks"))
        and release["returncode"] == 0
        and release["reaped"] is True
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "quantization owned child was not released normally",
    )
    require(
        absent_owned_birth(launch["owner"]) and absent_owned_birth(launch["child"]),
        "quantization owned GPU6 birth is still live",
    )
    for filename in ("gpu_before.json", "gpu_after.json"):
        gpu = evidence.json(root / filename)
        physical, uuid, memory = (part.strip() for part in gpu["actual_nvidia_smi"].split(","))
        require(
            physical == "6"
            and memory == "0"
            and uuid.removeprefix("GPU-").lower() == result["actual_gpu_uuid"].removeprefix("GPU-").lower(),
            "quantization actual GPU UUID or released memory differs",
        )
        if filename == "gpu_after.json":
            require(gpu["memory_released"] is True, "quantization GPU memory was not released")
    return dict(
        result=reference(root / "result.json"),
        actual_outputs=reference(outputs),
        launch=reference(root / "launch.json"),
        owner_exit=reference(root / "exit.json"),
        child_release=reference(root / "ownership/release.json"),
        actual_child_identity=reference(root / "actual_child_identity.json"),
        core_source=reference(core_path),
        worker_source=reference(root / "worker.py"),
        owner_source=reference(root / "owner.py"),
        owner_and_child_births_terminal=True,
        physical_gpu=6,
        actual_gpu_uuid=result["actual_gpu_uuid"],
        **bits,
    )


def numerics_admission(config, evidence):
    result_path = REPO / config["required_numerics_result"]
    verification_path = REPO / config["required_numerics_verification"]
    result, verification = evidence.json(result_path), evidence.json(verification_path)
    require(
        result.get("complete") is True and result.get("passed") is True and "error" not in result,
        "024 actual27 forwards must complete before025 preparation",
    )
    require(
        verification.get("passed") is True
        and verification.get("complete") is True
        and verification.get("prepare_only") is False
        and verification.get("calls_verified") == 27
        and verification.get("independent_losses_verified") == 18,
        "024 complete independent27-forward verification is required",
    )
    require(
        verification["input_sha256"].get(str(result_path.resolve())) == sha256(result_path),
        "024 verification is not bound to actual result",
    )
    for path, digest in verification["input_sha256"].items():
        evidence.check(path, digest)
    expected = [
        dict(label=f"{t}:{s}:{b}", completed=True) for t in range(3) for s in (0.2, 0.5, 0.8) for b in ("A", "B", "C")
    ]
    require(result["denoise_calls"] == expected, "024 actual denoise budget/order differs")
    require(
        result["physical_gpu"] == 6 and result["sampling_calls"] == result["physics_steps"] == 0,
        "024 execution contract differs",
    )
    orchestration = result_path.parent.parent / "orchestration"
    launch = evidence.json(orchestration / "launch.json")
    exit_record = evidence.json(orchestration / "exit.json")
    release = evidence.json(orchestration / "model_ownership/release.json")
    require(
        exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0),
        "024 owner has no successful actual wait",
    )
    require(
        release["pid"] == launch["model"]["pid"]
        and release["start_ticks"] == launch["model"]["start_ticks"]
        and release["returncode"] == 0
        and release["reaped"] is True
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "024 owned model did not exit and reap normally",
    )
    require(
        absent_owned_birth(launch["owner"]) and absent_owned_birth(launch["model"]),
        "024 owned GPU6 processes are still running",
    )
    require(launch["environment"]["CUDA_VISIBLE_DEVICES"] == "6", "024 was not the registered physicalGPU6 run")
    before, after = launch["external_cosmos"], evidence.json(orchestration / "external_cosmos_after.json")
    require(before == after and before["tracked_source_clean"] is True, "024 external tracked source identity changed")
    return dict(
        result=reference(result_path),
        verification=reference(verification_path),
        owner_exit=reference(orchestration / "exit.json"),
        model_release=reference(orchestration / "model_ownership/release.json"),
        owner_and_model_births_terminal=True,
        external_cosmos=after,
    )


def check_weights(files):
    require(len(files) == 163, "actual8400 weight roster must contain163 files")
    for path, saved in files.items():
        stat = Path(path).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (saved["bytes"], saved["mtime_ns"]), f"actual8400 stat changed: {path}"
        )


def validate_request(request, task, image_path):
    require(request["prompt"] == request["extra"]["action128"]["task_title"] == task, "wrong task-only request")
    require(request["vision_path"] == str(image_path), "request image source differs")
    require(request["extra"]["action128"]["action_domain"] == DOMAIN, "wrong request domain")
    require(request["extra"]["action128"]["action_dim_mask"] == ACTIVE.tolist(), "request active10 mask differs")
    for key, expected in dict(
        model_mode="wam", action_chunk_size=40, num_frames=41, fps=20, guidance=1.0, num_steps=35, shift=5.0
    ).items():
        require(request[key] == expected, f"request {key} differs")
    require(request["seed"] in (42, 43), "unregistered sampling seed")


def build_call_manifests(anchors, config):
    teacher, sampling = [], []
    for anchor in anchors:
        for sigma in config["sigma"]:
            teacher.append(
                dict(
                    call_index=len(teacher),
                    anchor_name=anchor["name"],
                    split=anchor["split"],
                    sigma=sigma,
                    noise_seed=42,
                )
            )
        if anchor["split"] == "validation":
            for seed in config["sampling_seeds"]:
                for variant in ("raw", "adapt"):
                    request = copy.deepcopy(anchor["request"])
                    request["seed"] = seed
                    sampling.append(
                        dict(
                            generation_index=len(sampling),
                            name=f"{anchor['name']}_{variant}_s{seed}",
                            anchor_name=anchor["name"],
                            pair_id=f"{anchor['name']}_s{seed}",
                            sampling_seed=seed,
                            variant=variant,
                            request=request,
                        )
                    )
    require(len(anchors) == 48 and len(teacher) == 144 and len(sampling) == 96, "registered144/96 manifest differs")
    require(sum(t["split"] == "train" for t in teacher) == 72, "whole-episode teacher split differs")
    return teacher, sampling


def prepare():
    require(Path.cwd().resolve() == REPO, "025 preparation must run from S2A")
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "025 preparation requires explicitly empty CUDA/LD",
    )
    evidence = Evidence()
    evidence.check(PLAN, PLAN_SHA)
    evidence.check(ROUNDING, ROUNDING_SHA)
    config = evidence.json(CONFIG, CONFIG_SHA)
    root = (REPO / config["output_dir"]).resolve()
    require(
        root.is_relative_to(REPO / "outputs") and not root.exists(), "025 output must be a fresh repository directory"
    )
    admission = numerics_admission(config, evidence)
    quantization = quantization_admission(evidence)
    samples = evidence.json(REPO / config["source_samples_manifest"])
    require(samples["config"]["control_fps"] == 20, "source measured-state clock is not20Hz")
    selected = split_episodes(config, samples["cases"])
    require(len(selected) == 12, "registered12 development episodes required")
    prior = evidence.json(REPO / config["source_sampler_prepared"])
    fingerprints = evidence.json(REPO / config["source_checkpoint_fingerprints"])
    actual = next(w for w in fingerprints["checkpoints"] if w["name"] == "8400_masked")
    checkpoint = str(Path(config["checkpoint"]["run"]) / "checkpoints" / config["checkpoint"]["iteration"] / "model")
    require(
        prior["checkpoint"] == actual["checkpoint"] == checkpoint and prior["weight_files"] == actual["files"],
        "actual8400 checkpoint identity differs",
    )
    check_weights(actual["files"])
    saved_path = Path(config["checkpoint"]["run"]) / "config.yaml"
    saved = yaml.safe_load(evidence.check(saved_path).read_text())["model"]["config"]
    domain = validate_domain(config, saved["action128_spec"])
    require(saved["action_tower_preset"] == "width_only_1b", "source hidden1280 architecture differs")
    fresh = evidence.json(FRESH)
    exclusions = [source_exclusion(item["sample"], fresh) for item in selected]
    cosmos = Path(config["cosmos_repo"])
    head = subprocess.check_output(["git", "-C", str(cosmos), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(cosmos), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    require(not dirty.strip(), "external tracked source must be clean")
    require(head == admission["external_cosmos"]["head"], "external source differs from admitted024")
    sources = [REPO / p for p in LOCAL_SOURCES] + [cosmos / "cosmos_framework" / p for p in EXTERNAL_SOURCES]
    sources += [
        CONFIG,
        PLAN,
        saved_path,
        REPO / config["policy_template"],
        QUANTIZATION / "owner.py",
        QUANTIZATION / "worker.py",
    ]
    source_hashes = {str(p.resolve()): sha256(evidence.check(p)) for p in sources}
    templates, identity = {}, []
    for task in range(3):
        sample = next(s for s in samples["cases"] if s["task_index"] == task and s["phase"] == "approach")
        name = f"{sample['name']}_h40_task_only_s42"
        cache = prior["cached_cases"][name]
        request = evidence.json(cache["request_path"], cache["request_sha256"])
        response = evidence.json(cache["response_path"], cache["response_sha256"])
        evidence.check(sample["image_path"], sample["image_sha256"])
        validate_request(request, sample["task"], sample["image_path"])
        require(request["seed"] == 42, "identity requires its original seed42 request")
        content = response["outputs"][0]["content"]
        require(
            np.asarray(content["action"]).shape
            == np.asarray(content["action128"]["normalized_action"]).shape
            == (40, 128),
            "identity source must contain complete raw/normalizedH40",
        )
        metadata = content["action128"]
        require(
            metadata["seed"] == 42
            and metadata["action_domain"] == DOMAIN
            and metadata["action_dim_mask"] == ACTIVE.tolist()
            and metadata["checkpoint_path"] == checkpoint,
            "identity response checkpoint/domain/mask/seed differs",
        )
        templates[task] = request
        identity.append(dict(name=name, task_index=task, request=request, cache=cache, sample=sample))
    root.mkdir()
    directory = root / "preparation"
    directory.mkdir()
    image_dir = directory / "images"
    image_dir.mkdir()
    try:
        anchors, episodes = [], []
        for selected_episode in selected:
            sample, split = selected_episode["sample"], selected_episode["split"]
            evidence.check(sample["arrays_path"], sample["arrays_sha256"])
            with np.load(sample["arrays_path"], allow_pickle=False) as original:
                arrays = {
                    k: original[k]
                    for k in (
                        "states",
                        "state_mask",
                        "actions",
                        "action_mask",
                        "action_valid",
                        "auxiliary_0",
                        "left_wrist",
                    )
                }
            frames = uniform_anchors(len(arrays["states"]))
            names = []
            for index, frame in enumerate(frames):
                observed = validate_window(arrays, frame)
                name = f"task{sample['task_index']}_ep{sample['episode']}_a{index}_f{frame}"
                image = image_dir / f"{name}.png"
                Image.fromarray(canvas([arrays[r][frame] for r in ("auxiliary_0", "left_wrist")])).save(image)
                request = copy.deepcopy(templates[sample["task_index"]])
                request.update(vision_path=str(image), prompt=sample["task"], seed=42)
                request["extra"]["action128"]["task_title"] = sample["task"]
                validate_request(request, sample["task"], image)
                anchors.append(
                    dict(
                        name=name,
                        episode=sample["episode"],
                        source_episode_id=sample["source_episode_id"],
                        task=sample["task"],
                        task_index=sample["task_index"],
                        split=split,
                        original_split=sample["split"],
                        anchor_index=index,
                        anchor_frame=frame,
                        arrays_path=sample["arrays_path"],
                        arrays_sha256=sample["arrays_sha256"],
                        image_path=str(image),
                        image_sha256=sha256(image),
                        request=request,
                        **observed,
                    )
                )
                names.append(name)
            episodes.append(
                dict(
                    episode=sample["episode"],
                    source_episode_id=sample["source_episode_id"],
                    task_index=sample["task_index"],
                    split=split,
                    frames=frames,
                    anchors=names,
                )
            )
        teacher, sampling = build_call_manifests(anchors, config)
        snapshots, copies = directory / "source_snapshot", {}
        snapshots.mkdir()
        for index, (path, digest) in enumerate(source_hashes.items()):
            target = snapshots / f"{index:03d}_{Path(path).name}"
            shutil.copyfile(path, target)
            require(sha256(target) == digest, "source snapshot differs")
            copies[str(target)] = digest
        evidence.recheck()
        check_weights(actual["files"])
        final_head = subprocess.check_output(["git", "-C", str(cosmos), "rev-parse", "HEAD"], text=True).strip()
        final_dirty = subprocess.check_output(
            ["git", "-C", str(cosmos), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        require(final_head == head and not final_dirty.strip(), "external tracked source changed during preparation")
        result = dict(
            schema_version="s2a.frozen-readout-preparation.v1",
            config=config,
            config_ref=reference(CONFIG),
            registration=reference(PLAN),
            output_root=str(root),
            checkpoint=checkpoint,
            weight_files=actual["files"],
            domain=domain,
            numerics_admission=admission,
            quantization_admission=quantization,
            external_cosmos=dict(head=head, tracked_source_clean=True),
            episodes=episodes,
            anchors=anchors,
            zero_identity_cases=identity,
            teacher_calls=teacher,
            validation_generations=sampling,
            conservative_source_exclusion=dict(
                reference=reference(FRESH),
                records=exclusions,
                historical_initial_physical_sync_certified=False,
                pretraining_visibility="unknown",
            ),
            budget=config["budget"],
            source_sha256=source_hashes,
            snapshot_sha256=copies,
            input_sha256=evidence.hashes,
            new_model_calls=0,
            fit_count=0,
            physics_steps=0,
            fresh_performance_cases=0,
            full_weight_bytes_rehashed=False,
        )
        path = directory / "prepared.json"
        write_once(path, result)
        return path
    except BaseException as error:
        write_once(
            directory / "error.json",
            dict(type=type(error).__name__, message=str(error), input_sha256=evidence.hashes, automatic_retry=False),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(args.seed == 42, "025 requires registered seed42")
    path = prepare()
    print(json.dumps(dict(prepared=reference(path), model_calls=0, fit_count=0, physics_steps=0)))


if __name__ == "__main__":
    main()
