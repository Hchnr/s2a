"""Independent CPU verification of024's recorded training/inference tensors.

NumPy implements the recorded external rectified-flow equations and explicit
valid-mask loss branch. No producer math, Cosmos modules, torch or dataloader is
imported. Tensor recording/forward execution belongs to the separate producer.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from s2a.artifacts import REPO

CONFIG = REPO / "configs/wam_current_numerics_20261021.json"
CONFIG_SHA = "2fbdd966fae22af37f28202ca15cc76e33cdf896c8b3ad0f9df0869c06fb6edd"
PLAN = REPO / "plans/v0.0.1/024_current_training_inference_numerics_20261021.md"
PLAN_SHA = "30e039da96aecf63eb7bf5ff542855004fcc4cecd05435e1d36c1c4881564813"
CANVAS_SHA = "903a8616e26bc4ee5fb0c9575870ed0600e98f5a74229cbf5be9ff610ffe74c1"
FINGERPRINT_SHA = "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8"
ACTIVE = np.asarray([True] * 10 + [False] * 118)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Evidence:
    """Read hash-bound evidence and recheck every consumed input after auditing."""

    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).absolute()
        actual = digest(path)
        require(expected is None or actual == expected, f"source SHA differs: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == actual, f"source changed: {path}")
        self.hashes[str(path)] = actual
        return path

    def json(self, path, expected=None):
        return json.loads(self.check(path, expected).read_text())

    def reference(self, reference):
        require(set(reference) == {"path", "sha256"}, "invalid evidence reference")
        return self.json(reference["path"], reference["sha256"])

    def recheck(self):
        for path, expected in self.hashes.copy().items():
            self.check(path, expected)


def load_bundle(reference, evidence):
    metadata = evidence.reference(reference)
    path = evidence.check(metadata["arrays"]["path"], metadata["arrays"]["sha256"])
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: numeric(source[key], key).copy() for key in source.files}
    require(set(arrays) == set(metadata["array_index"]), "bundle index has omitted/extra tensor")
    for key, value in arrays.items():
        require(
            metadata["array_index"][key] == dict(shape=list(value.shape), dtype=str(value.dtype)),
            f"bundle index differs: {key}",
        )
    return metadata, arrays


def materialize(value, arrays):
    if isinstance(value, dict) and "npz_key" in value:
        require(set(value) == {"npz_key", "original_dtype"}, "malformed recorded tensor descriptor")
        return arrays[value["npz_key"]]
    if isinstance(value, dict):
        return {key: materialize(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize(item, arrays) for item in value]
    return value


def same_tree(left, right, name):
    """Compare real metadata recursively; descriptor names are not semantics."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        exact(left, right, name)
    elif isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), f"{name}: metadata fields differ")
        for key in left:
            same_tree(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), f"{name}: list length differs")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            same_tree(a, b, f"{name}.{i}")
    else:
        require(type(left) is type(right) and left == right, f"{name}: metadata value differs")


def close(actual, expected, name, *, rtol, atol):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(actual.shape == expected.shape, f"{name}: shape mismatch")
    difference = float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0))
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), f"{name}: numerical mismatch max_abs={difference}")
    return difference


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, name):
    array = np.asarray(value)
    require(array.dtype.kind in "biuf" and np.isfinite(array).all(), f"{name}: requires finite numeric data")
    return array


def exact(actual, expected, name):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(actual.shape == expected.shape and np.array_equal(actual, expected), f"{name}: elementwise exact mismatch")


def cast_model_precision(value, dtype):
    """Round float32 to recorded model precision without importing torch.

    BF16 uses round-to-nearest, ties-to-even on the retained upper16 bits.
    Values are re-expanded losslessly to float32 for ordinary NPZ storage.
    """
    value = numeric(value, "precision input").astype(np.float32)
    if dtype in ("torch.float32", "float32"):
        return value.copy()
    if dtype in ("torch.float16", "float16"):
        result = value.astype(np.float16).astype(np.float32)
    elif dtype in ("torch.bfloat16", "bfloat16"):
        bits = value.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
        result = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    else:
        raise ValueError(f"unsupported recorded model dtype: {dtype}")
    require(np.isfinite(result).all(), "precision cast overflowed")
    return result


def expanded_mask(mask, shape, name):
    mask = numeric(mask, name)
    require(np.all((mask == 0) | (mask == 1)), f"{name}: requires binary mask")
    try:
        return np.broadcast_to(mask, shape)
    except ValueError as error:
        raise ValueError(f"{name}: cannot broadcast {mask.shape} to {shape}") from error


def flow_equations(x0, epsilon, condition, valid, sigma, *, modality, model_dtype):
    """epsilon-x0 target; xt=sigma_eff*epsilon+(1-sigma_eff)*x0.

    This follows rectified_flow.get_interpolation and the caller's subsequent
    dtype cast/Action128 valid-mask zeroing. Conditioning vision targets remain
    epsilon-x0; conditioning is excluded by the loss, not by changing targets.
    """
    x0, epsilon = numeric(x0, "x0").astype(np.float32), numeric(epsilon, "epsilon").astype(np.float32)
    require(x0.shape == epsilon.shape and x0.size > 0, "flow: x0/epsilon shape mismatch")
    require(modality in ("vision", "action"), "flow: unsupported modality")
    require(x0.ndim in ((4, 5) if modality == "vision" else (2,)), "flow: wrong modality rank")
    require(x0.ndim != 5 or x0.shape[0] == 1, "flow: recorded per-item batch must be one")
    sigma = np.float32(sigma)
    require(np.isfinite(sigma) and 0 < sigma < 1, "flow: invalid fixed sigma")
    condition = numeric(condition, "condition").astype(np.float32)
    expanded_mask(condition, x0.shape, "condition")
    effective_sigma = sigma * (np.float32(1) - condition)
    target = epsilon - x0
    xt = cast_model_precision(epsilon * effective_sigma + x0 * (np.float32(1) - effective_sigma), model_dtype)
    if modality == "action":
        require(valid is not None, "flow: training action needs explicit valid mask")
        active = expanded_mask(valid, x0.shape, "valid").astype(bool)
        xt = np.where(active, xt, np.float32(0))
        target = np.where(active, target, np.float32(0))
    return dict(xt=xt, target=target, effective_sigma=effective_sigma)


def explicit_mask_loss(prediction, target, condition, valid, timesteps, *, modality, time_weight_method):
    """Rebuild the explicit loss_valid_mask branch, including its denominator.

    The current external implementation supports only uniform RF time weights
    (legacy 'reweighting' aliases uniform). The valid-mask branch always divides
    by active broadcast elements, independent of normalize_loss_by_active.
    """
    prediction, target = numeric(prediction, "prediction"), numeric(target, "target")
    require(prediction.shape == target.shape and prediction.size > 0, "loss: prediction/target shape mismatch")
    require(modality in ("vision", "action"), "loss: unsupported modality")
    require(prediction.ndim in ((4, 5) if modality == "vision" else (2,)), "loss: wrong modality rank")
    require(prediction.ndim != 5 or prediction.shape[0] == 1, "loss: recorded per-item batch must be one")
    require(valid is not None, "loss: explicit valid mask is required")
    condition = numeric(condition, "condition").astype(np.float32)
    condition_broadcast = expanded_mask(condition, prediction.shape, "condition")
    valid_broadcast = expanded_mask(valid, prediction.shape, "valid")
    active = ((1 - condition_broadcast) * valid_broadcast).astype(bool)
    denominator = max(int(np.count_nonzero(active)), 1)
    # Subtraction/square execute in float32 in the source (BF16 pred minus
    # float32 target). Accumulate independently in float64 within declared
    # float32-reduction tolerances; never use this tolerance for B/C equality.
    difference = prediction.astype(np.float32) - target.astype(np.float32)
    squared = np.square(np.where(active, difference, np.float32(0)), dtype=np.float32)
    require(time_weight_method in ("uniform", "reweighting"), "loss: unregistered time weight")
    times = numeric(timesteps, "timesteps").reshape(-1)
    temporal_count = prediction.shape[-3 if modality == "vision" else 0]
    require(len(times) in (1, temporal_count), "loss: incompatible timestep count")
    weights = np.ones(times.shape, dtype=np.float32)
    time_shape = ((1,) * (prediction.ndim - 3) + (len(times), 1, 1)) if modality == "vision" else (len(times), 1)
    weighted = squared * weights.reshape(time_shape)
    return dict(
        denominator=denominator,
        active_elements=int(np.count_nonzero(active)),
        loss=float(np.sum(weighted, dtype=np.float64) / denominator),
        per_instance_unweighted=float(np.sum(squared, dtype=np.float64) / denominator),
        time_weight=weights,
    )


def registered_call_grid(records):
    """Require the registered nine triplets/27 denoise calls without duplicates."""
    expected = {(task, sigma, branch) for task in range(3) for sigma in (0.2, 0.5, 0.8) for branch in ("A", "B", "C")}
    actual = [(record["task_index"], record["sigma"], record["branch"]) for record in records]
    require(len(actual) == 27 and len(set(actual)) == 27 and set(actual) == expected, "wrong registered 27-call grid")


def verify_preparation(path, evidence):
    prepared = evidence.json(path)
    require(prepared["schema_version"] == "s2a.current-numerics-preparation.v1", "wrong preparation schema")
    evidence.check(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    require(prepared["config_ref"] == dict(path=str(CONFIG), sha256=CONFIG_SHA), "prepared config identity differs")
    require(prepared["registration"] == dict(path=str(PLAN), sha256=PLAN_SHA), "prepared registration identity differs")
    config = evidence.json(CONFIG)
    require(prepared["config"] == config and prepared["new_model_calls"] == 0, "preparation already used model")
    for mapping in ("source_sha256", "input_sha256", "snapshot_sha256"):
        require(prepared[mapping], f"empty {mapping}")
        for source, expected in prepared[mapping].items():
            evidence.check(source, expected)
    canvas_path = REPO / "src/s2a/libero_simulation.py"
    require(prepared["source_sha256"][str(canvas_path)] == CANVAS_SHA, "unregistered canvas implementation")
    evidence.check(canvas_path, CANVAS_SHA)
    fingerprint = evidence.json(REPO / config["source_checkpoint_fingerprints"], FINGERPRINT_SHA)
    weights = next(c for c in fingerprint["checkpoints"] if c["name"] == "8400_masked")
    checkpoint = config["checkpoints"][0]
    expected_path = str(Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model")
    require(weights["checkpoint"] == prepared["checkpoint"] == expected_path, "wrong actual8400 checkpoint")
    require(
        weights["files"] == prepared["weight_files"] and len(weights["files"]) == 163, "wrong checkpoint file roster"
    )
    for filename, saved in weights["files"].items():
        stat = Path(filename).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (saved["bytes"], saved["mtime_ns"]),
            f"checkpoint stat changed: {filename}",
        )
        require(len(saved["sha256"]) == 64, "missing original checkpoint byte fingerprint")
    source = evidence.json(REPO / config["source_samples_manifest"])["cases"]
    expected_samples = [next(s for s in source if s["task_index"] == i and s["phase"] == "approach") for i in range(3)]
    require([c["sample"] for c in prepared["cases"]] == expected_samples, "wrong registered three approach samples")
    for case in prepared["cases"]:
        sample, request = case["sample"], case["request"]
        require(request == evidence.reference(case["prior_request"]), "prior online request differs")
        trace = evidence.reference(case["prior_trace"])
        require(trace["conditioning"]["caption"] == [case["online_caption"]], "prior actual online caption differs")
        caption = json.loads(case["online_caption"])
        require(caption["task_title"] == caption["command"] == sample["task"], "online task caption mismatches sample")
        require(
            caption.get("task_plan", []) == [] and "current_subtask" not in caption,
            "future/phase plan in online condition",
        )
        require(
            request["prompt"] == request["extra"]["action128"]["task_title"] == sample["task"],
            "wrong online task route",
        )
        require(request["extra"]["action128"]["action_dim_mask"] == ACTIVE.tolist(), "wrong current mask10 request")
        for key, expected in dict(
            seed=42,
            action_chunk_size=40,
            num_frames=41,
            guidance=1.0,
            shift=5.0,
            num_steps=35,
            fps=20,
            model_mode="wam",
        ).items():
            require(request[key] == expected, f"wrong registered request {key}")
        require(request["vision_path"] == sample["image_path"], "wrong request source image")
        for role in ("arrays", "image"):
            evidence.check(sample[f"{role}_path"], sample[f"{role}_sha256"])
    return prepared, config


def normalized_actions(physical, transform, valid):
    result = numeric(physical, "physical actions").astype(np.float32).copy()
    mask = numeric(transform["normalization_mask"], "normalization mask").astype(bool)
    scale, bias = (numeric(transform[name], name).astype(np.float32) for name in ("scale", "bias"))
    require(mask.shape == scale.shape == bias.shape == (128,), "wrong normalization128 dimensions")
    require(np.all(scale[mask] != 0), "zero normalization scale")
    result[:, mask] = result[:, mask] * scale[mask] + bias[mask]
    result[:, mask] = np.clip(result[:, mask], transform["clip_min"], transform["clip_max"])
    result[~valid] = 0
    return result


def verify_source(case, reference, evidence):
    metadata, arrays = load_bundle(reference, evidence)
    sample, frame = case["sample"], case["sample"]["anchor_frame"]
    with np.load(evidence.check(sample["arrays_path"], sample["arrays_sha256"]), allow_pickle=False) as original:
        for role in ("auxiliary_0", "left_wrist"):
            raw = arrays[f"raw_camera_{role}"]
            require(raw.dtype == np.uint8 and len(raw) == 41, "source camera needs actual41 uint8 frames")
            exact(raw, original[role][frame : frame + 41], f"true source {role}")
        for output, source, count, dtype in (
            ("physical_action", "actions", 40, np.float32),
            ("action_valid", "action_valid", 40, bool),
            ("action_mask", "action_mask", 40, bool),
        ):
            exact(arrays[output], original[source][frame : frame + count].astype(dtype), f"source {output}")
    valid = arrays["action_mask"].astype(bool) & arrays["action_valid"].astype(bool)[:, None]
    exact(valid, np.broadcast_to(ACTIVE, (40, 128)), "true future active10")
    exact(
        arrays["normalized_action"],
        normalized_actions(arrays["physical_action"], metadata["transform"], valid),
        "independent full normalization",
    )
    canvas = arrays["canvas"]
    require(canvas.shape == (41, 960, 640, 3) and canvas.dtype == np.uint8, "wrong actual41 canvas")
    require(not np.any(canvas[:, 478:482]), "historical4px seam differs")
    with Image.open(evidence.check(sample["image_path"], sample["image_sha256"])) as image:
        exact(canvas[0], np.asarray(image), "actual first canvas versus released anchor")
    # Future resizing reuses the pinned, reviewed training cover-crop contract.
    # NumPy does not independently reproduce VAE encoding or torch's uint8 ties.
    require(metadata["canvas_source_sha256"] == CANVAS_SHA, "actual canvas source pin differs")
    saved_model = yaml.safe_load(
        evidence.check(metadata["service_spec_source"]["path"], metadata["service_spec_source"]["sha256"]).read_text()
    )["model"]["config"]
    require(
        metadata["service_spec"] == saved_model["action128_spec"],
        "actual normalization spec differs from saved checkpoint contract",
    )
    require(
        metadata["transform"] == saved_model["action128_spec"]["transforms"][metadata["domain"]],
        "actual domain transform differs",
    )
    return metadata, arrays


def verify_loss_settings(settings, saved_model):
    rf = saved_model["rectified_flow_training_config"]
    expected = dict(
        vision_weight=float(rf["loss_scale"]),
        action_weight=float(rf["action_loss_weight"]),
        text_weight=float(saved_model["text_loss_weight"]),
        normalize_loss_by_active=rf["normalize_loss_by_active"],
        sample_level_loss_averaging=rf["sample_level_loss_averaging"],
        time_weight_vision=rf["train_time_weight"],
        time_weight_action=rf["train_time_weight"],
        predict_text_tokens=saved_model["predict_text_tokens"],
        vision_gen=saved_model["vision_gen"],
        action_gen=saved_model["action_gen"],
        sound_gen=saved_model["sound_gen"],
    )
    require(settings == expected, "actual loss configuration differs from saved checkpoint contract")


def verify_clean(case, references, source_reference, source, evidence, video_key="video"):
    clean = {}
    for branch in ("A", "B", "C"):
        metadata, arrays = load_bundle(references[branch], evidence)
        require(metadata["source_ref"] == source_reference, "clean source identity differs")
        tree = dict(
            clean=materialize(metadata["tree"], arrays), raw_batch=materialize(metadata["raw_batch_tree"], arrays)
        )
        clean[branch] = dict(metadata=metadata, arrays=arrays, tree=tree)
        expected_caption = case["sample"]["published_prompts"]["40"] if branch == "A" else case["online_caption"]
        require(metadata["caption"] == expected_caption, f"{branch}: actual caption differs")
        require(metadata["has_action"] is True, "current conditional bridge requires noisy action")
        require(
            tree["clean"]["batch_size"] == 1 and tree["clean"]["is_image_batch"] is False,
            "wrong training modality/batch",
        )
        require(
            len(tree["clean"]["x0_tokens_vision"]) == len(tree["clean"]["x0_tokens_action"]) == 1,
            "wrong latent item count",
        )
        latent = tree["clean"]["x0_tokens_vision"][0]
        require(
            latent.ndim == 5 and latent.shape[0] == 1 and latent.shape[2:] == (11, 60, 40),
            "wrong41-frame latent geometry",
        )
        require(tree["clean"]["x0_tokens_action"][0].shape == (40, 128), "wrong40-action geometry")
        if branch in ("A", "B"):
            exact(
                tree["clean"]["x0_tokens_action"][0],
                source["normalized_action"],
                f"{branch}: prepared true action labels",
            )
            # The actual raw batch tensor, not merely a reconstructed source copy.
            exact(
                tree["raw_batch"][video_key][0][0],
                source["canvas"].transpose(3, 0, 1, 2),
                f"{branch}: actual training future41 RGB",
            )
        else:
            actual_video = tree["raw_batch"][video_key][0][0]
            require(actual_video.shape == (3, 41, 960, 640), "stock online video shape differs")
            exact(actual_video[:, 0], source["canvas"][0].transpose(2, 0, 1), "stock current-frame RGB")
            require(not np.any(actual_video[:, 1:]), "future RGB entered stock online condition")
            require(
                not np.any(tree["raw_batch"]["action"][0][0]), "future action labels entered stock online condition"
            )
            initial = arrays["initial.0"]
            require(initial.ndim == 1 and len(initial) == latent.size + 40 * 128, "wrong stock initial noise shape")
            require(
                not np.any(initial[latent.size :].reshape(40, 128)[:, 10:]), "stock initial inactive slots are nonzero"
            )
    same_tree(clean["A"]["tree"]["clean"], clean["B"]["tree"]["clean"], "A/B full clean labels and metadata")
    same_tree(
        materialize(clean["A"]["metadata"]["plans"], clean["A"]["arrays"]),
        materialize(clean["B"]["metadata"]["plans"], clean["B"]["arrays"]),
        "A/B sequence plan",
    )
    same_tree(
        materialize(clean["B"]["metadata"]["plans"], clean["B"]["arrays"]),
        materialize(clean["C"]["metadata"]["plans"], clean["C"]["arrays"]),
        "B/C actual stock sequence plan",
    )
    same_tree(clean["B"]["metadata"]["text_tokens"], clean["C"]["metadata"]["text_tokens"], "B/C actual tokens")
    require(
        clean["A"]["metadata"]["domain"] == clean["B"]["metadata"]["domain"] == clean["C"]["metadata"]["domain"],
        "actual domain differs",
    )
    for key in (
        "fps_vision",
        "fps_action",
        "temporal_positions_vision",
        "spatial_seams_vision",
        "action_domain_id",
        "raw_action_dim",
    ):
        same_tree(clean["B"]["tree"]["clean"][key], clean["C"]["tree"]["clean"][key], f"B/C {key}")
    exact(
        clean["B"]["tree"]["clean"]["x0_tokens_vision"][0][:, :, :1],
        clean["C"]["tree"]["clean"]["x0_tokens_vision"][0][:, :, :1],
        "B/C actual first conditioning latent",
    )
    return clean


def forward_metadata(packed):
    """Exclude only loss supervision; all model-facing fields remain exact."""
    result = {
        key: item
        for key, item in packed.items()
        if key not in {"label_ids", "ce_loss_indexes", "ce_loss_weights", "vision", "action"}
    }
    for modality in ("vision", "action"):
        result[modality] = {key: item for key, item in packed[modality].items() if key != "loss_valid_mask"}
    return result


def verify_recorded_loss(metadata, arrays, noised, packed, config):
    losses, loss_config = materialize(metadata["losses"], arrays), metadata["loss_config"]
    require(
        set(losses)
        <= {
            "total",
            "total_loss",
            "cross_entropy_loss_text",
            "flow_matching_loss_vision",
            "flow_matching_loss_vision_per_instance",
            "flow_matching_loss_action",
            "flow_matching_loss_sound",
        },
        "unaccounted auxiliary/other loss",
    )
    computed = {}
    for modality in ("vision", "action"):
        result = explicit_mask_loss(
            arrays[f"pred_{modality}"],
            noised[f"vt_target_{modality}"][0],
            packed[modality]["condition_mask"][0],
            packed[modality]["loss_valid_mask"][0],
            arrays["timestep"],
            modality=modality,
            time_weight_method=loss_config[f"time_weight_{modality}"],
        )
        close(
            arrays[f"time_weight_{modality}"].reshape(-1),
            result["time_weight"],
            f"actual {modality} time weight",
            rtol=0,
            atol=0,
        )
        close(
            losses[f"flow_matching_loss_{modality}"],
            result["loss"],
            f"independent {modality} loss",
            rtol=config["loss_rtol"],
            atol=config["loss_atol"],
        )
        if modality == "vision":
            close(
                losses["flow_matching_loss_vision_per_instance"],
                [result["per_instance_unweighted"]],
                "vision per-instance unweighted loss",
                rtol=config["loss_rtol"],
                atol=config["loss_atol"],
            )
        computed[modality] = result
    require(
        loss_config["sample_level_loss_averaging"] is False
        and loss_config["predict_text_tokens"] is False
        and loss_config["sound_gen"] is False,
        "unregistered non-flow loss branch",
    )
    require(
        loss_config["vision_gen"] is True and loss_config["action_gen"] is True, "missing real vision/action objective"
    )
    require(
        float(np.asarray(losses["cross_entropy_loss_text"]))
        == float(np.asarray(losses["flow_matching_loss_sound"]))
        == 0,
        "unexpected nonzero omitted loss",
    )
    total = (
        computed["vision"]["loss"] * loss_config["vision_weight"]
        + computed["action"]["loss"] * loss_config["action_weight"]
    )
    close(
        losses.get("total", losses.get("total_loss")),
        total,
        "independent weighted total loss",
        rtol=config["loss_rtol"],
        atol=config["loss_atol"],
    )
    return {
        key: {name: value for name, value in item.items() if name != "time_weight"} for key, item in computed.items()
    } | dict(total=total)


def rng_content(reference, evidence):
    metadata, arrays = load_bundle(reference, evidence)
    require(
        metadata["cuda_devices"] == 1 and set(arrays) == {"numpy_keys", "torch_cpu", "torch_cuda_0"},
        "wrong single-visible-GPU RNG roster",
    )
    require(
        arrays["numpy_keys"].dtype == np.uint32
        and arrays["torch_cpu"].dtype == arrays["torch_cuda_0"].dtype == np.uint8,
        "wrong actual RNG state dtype",
    )
    return dict(
        python=metadata["python"], numpy=metadata["numpy"], cuda_devices=metadata["cuda_devices"], arrays=arrays
    )


def check_call(identity, case, clean, evidence, config, model_dtype):
    metadata, arrays = load_bundle(identity["bundle"], evidence)
    for key in ("task_index", "sigma", "branch"):
        require(metadata[key] == identity[key], f"call identity {key} differs")
    branch, sigma = identity["branch"], identity["sigma"]
    require(metadata["clean_ref"] == case["clean"][branch], "call consumed wrong clean branch")
    packed = materialize(metadata["packed_tree"], arrays)
    for modality in ("vision", "action"):
        exact(arrays[f"packed_{modality}"], packed[modality]["tokens"][0], f"{branch}: actual packed {modality} alias")
        exact(
            arrays[f"condition_{modality}"],
            packed[modality]["condition_mask"][0],
            f"{branch}: actual condition {modality} alias",
        )
    expected_condition = np.zeros((11, 1, 1), np.float32)
    expected_condition[0] = 1
    exact(arrays["condition_vision"], expected_condition, "actual first-only vision condition mask")
    exact(arrays["condition_action"], np.zeros((40, 1)), "actual no conditioned actions")
    require(
        arrays["pred_action"].shape == arrays["packed_action"].shape == (40, 128),
        "wrong actual action prediction/input geometry",
    )
    require(
        not np.any(arrays["pred_action"][:, 10:]) and not np.any(arrays["packed_action"][:, 10:]),
        "inactive model-boundary action/velocity is nonzero",
    )
    exact(arrays["timestep"], np.asarray([[sigma * 1000]], np.float32), "actual registered flow timestep")
    report = dict(task_index=identity["task_index"], sigma=sigma, branch=branch)
    noised = None
    if branch in ("A", "B"):
        noised = materialize(metadata["noised_tree"], arrays)
        exact(
            packed["action"]["loss_valid_mask"][0],
            np.broadcast_to(ACTIVE, (40, 128)),
            "actual training action valid mask",
        )
        exact(packed["vision"]["loss_valid_mask"][0], np.ones((11, 60, 40)), "actual training future vision valid mask")
        for modality in ("vision", "action"):
            flow = flow_equations(
                clean[branch]["tree"]["clean"][f"x0_tokens_{modality}"][0],
                noised[f"epsilon_{modality}"][0],
                packed[modality]["condition_mask"][0],
                packed[modality]["loss_valid_mask"][0],
                sigma,
                modality=modality,
                model_dtype=model_dtype,
            )
            close(
                noised[f"sigmas_{modality}"][0],
                flow["effective_sigma"],
                f"{branch} {modality} effective sigma",
                rtol=0,
                atol=0,
            )
            for recorded, expected in (("xt_tokens", "xt"), ("vt_target", "target")):
                close(
                    noised[f"{recorded}_{modality}"][0],
                    flow[expected],
                    f"independent {branch} {modality} {recorded}",
                    rtol=config["flow_rtol"],
                    atol=config["flow_atol"],
                )
            exact(
                packed[modality]["tokens"][0],
                noised[f"xt_tokens_{modality}"][0],
                f"{branch} noised to packed {modality}",
            )
        report["independent_losses"] = verify_recorded_loss(metadata, arrays, noised, packed, config)
    else:
        require(
            metadata["noised_tree"] is None and metadata["losses"] is None,
            "C must use supplied B xt without extra training noising/loss",
        )
    before = rng_content(metadata["rng_before"], evidence)
    after = rng_content(metadata["rng_after"], evidence)
    return dict(
        metadata=metadata, arrays=arrays, packed=packed, noised=noised, before=before, after=after, report=report
    )


def verify_result(prepared, path, evidence):
    result = evidence.json(path)
    require(result["schema_version"] == "s2a.current-numerics-result.v1", "wrong actual run schema")
    require(
        result["complete"] is True and result["passed"] is True and "error" not in result,
        "partial/rejected run cannot pass independent audit",
    )
    require(
        result["prepared_ref"]
        == dict(
            path=str(Path(prepared["output_root"]) / "preparation/prepared.json"),
            sha256=digest(Path(prepared["output_root"]) / "preparation/prepared.json"),
        ),
        "actual run preparation reference differs",
    )
    require(
        result["physical_gpu"] == 6 and result["cuda_visible_devices"] == "6" and result["ld_library_path"] == "",
        "actual model GPU/environment differs",
    )
    require(
        result["sampling_calls"] == result["physics_steps"] == result["fresh_cases"] == 0,
        "unregistered sampling/physics/fresh use",
    )
    require(result["budgets"] == prepared["budget"], "actual registered budget differs")
    require(
        result["training_prepare_iteration"] == 8400
        and result["noising_iteration_argument"] is None
        and result["all_modules_eval"] is True,
        "wrong actual training iteration/noising/eval contract",
    )
    expected_labels = [
        f"{task}:{sigma}:{branch}" for task in range(3) for sigma in (0.2, 0.5, 0.8) for branch in ("A", "B", "C")
    ]
    require(
        result["denoise_calls"] == [dict(label=label, completed=True) for label in expected_labels],
        "actual denoise boundary ledger differs from exact27 budget",
    )
    require(len(result["cases"]) == 3, "partial sample roster")
    records = [call for case in result["cases"] for call in case["calls"]]
    registered_call_grid(records)
    config = prepared["config"]["comparison"]
    reports = []
    for registered, actual in zip(prepared["cases"], result["cases"], strict=True):
        sample = registered["sample"]
        require(
            actual["sample_name"] == sample["name"] and actual["task_index"] == sample["task_index"],
            "actual sample/task routing differs",
        )
        source_meta, source = verify_source(registered, actual["source"], evidence)
        clean = verify_clean(
            registered, actual["clean"], actual["source"], source, evidence, source_meta["model_input_video_key"]
        )
        saved_model = yaml.safe_load(
            evidence.check(
                source_meta["service_spec_source"]["path"], source_meta["service_spec_source"]["sha256"]
            ).read_text()
        )["model"]["config"]
        verify_loss_settings(result["loss_config"], saved_model)
        require(
            source_meta["domain"] == clean["B"]["metadata"]["domain"], "source/prepared normalization domain differs"
        )
        require(len(actual["calls"]) == 9, "wrong per-sample denoise count")
        for offset in range(0, 9, 3):
            triplet = actual["calls"][offset : offset + 3]
            require([c["branch"] for c in triplet] == ["A", "B", "C"], "wrong denoise branch order")
            require(
                all(
                    c["task_index"] == sample["task_index"] and c["sigma"] == (0.2, 0.5, 0.8)[offset // 3]
                    for c in triplet
                ),
                "wrong actual sample/sigma order",
            )
            branches = {
                c["branch"]: check_call(c, actual, clean, evidence, config, result["model_dtype"]) for c in triplet
            }
            a, b, c = (branches[k] for k in ("A", "B", "C"))
            same_tree(a["noised"], b["noised"], "A/B actual paired noise/xt/target")
            same_tree(a["before"], b["before"], "A/B restored full RNG state")
            same_tree(a["after"], b["after"], "A/B paired full RNG after actual forward")
            same_tree(b["after"], c["before"], "B to C no hidden RNG reset")
            same_tree(c["before"], c["after"], "C conditional consumes no extra RNG")
            same_tree(
                forward_metadata(b["packed"]),
                forward_metadata(c["packed"]),
                "B/C full actual model-facing packed metadata",
            )
            exact(
                b["arrays"]["pred_action"][:, :10],
                c["arrays"]["pred_action"][:, :10],
                "B/C active conditional velocity",
            )
            flat = np.concatenate(
                (
                    b["arrays"]["packed_vision"].astype(np.float32).reshape(-1),
                    b["arrays"]["packed_action"].astype(np.float32).reshape(-1),
                )
            )
            exact(c["arrays"]["received_xt"], flat, "C receives actual B xt")
            offset_action = b["arrays"]["packed_vision"].size
            returned_action = c["arrays"]["velocity"][offset_action:].reshape(40, 128)
            exact(returned_action, b["arrays"]["pred_action"], "actual returned conditional Action128 velocity")
            returned_vision = c["arrays"]["velocity"][:offset_action].reshape(c["arrays"]["pred_vision"].shape)
            expected_vision = c["arrays"]["pred_vision"] * (1 - c["arrays"]["condition_vision"])
            exact(returned_vision, expected_vision, "actual returned conditional vision velocity mask")
            for branch in branches.values():
                require(branch["metadata"]["loss_config"] == result["loss_config"], "call loss settings changed")
                reports.append(branch["report"])
    return dict(
        complete=True,
        calls_verified=27,
        true_future_samples=3,
        independent_losses_verified=18,
        sampling_calls=0,
        physics_steps=0,
        fresh_cases=0,
        checkpoint_files_stat_rechecked=163,
        checkpoint_weight_bytes_rehashed=False,
        raw_checkpoint_task_success_proven=False,
        calls=reports,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(args.seed == 42, "024 independent verifier requires registered seed42")
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("LD_LIBRARY_PATH") == "",
        "CPU verifier requires explicit empty CUDA/LD",
    )
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "independent output must be new under repository outputs",
    )
    require(args.prepare_only is (args.result is None), "prepare-only/result mode mismatch")
    evidence = Evidence()
    receipt = dict(
        schema_version="s2a.current-numerics-independent-verification.v1",
        passed=False,
        prepare_only=args.prepare_only,
        new_model_calls=0,
    )
    error = None
    try:
        prepared, _ = verify_preparation(args.prepared, evidence)
        if args.prepare_only:
            receipt.update(preparation_verified=True, new_denoise_calls=0, actual_forward_results_verified=False)
        else:
            receipt.update(verify_result(prepared, args.result, evidence))
        evidence.recheck()
        receipt["passed"] = True
    except Exception as caught:
        error = caught
        receipt["error"] = dict(type=type(caught).__name__, message=str(caught))
    receipt["input_sha256"] = evidence.hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=receipt["passed"], output=str(output), sha256=digest(output))))
    if error is not None:
        raise SystemExit(f"independent024 rejected: {error}")


if __name__ == "__main__":
    main()
