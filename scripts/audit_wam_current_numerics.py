"""024: true-future training objectives and the current online conditional bridge.

Preparation is CPU-only. The explicit run mode performs exactly the registered
27 denoise calls, never a sampler, HTTP generation, backward or physics step.
"""

import argparse
import copy
import dataclasses
import datetime
import enum
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

CONFIG = REPO / "configs/wam_current_numerics_20261021.json"
CONFIG_SHA = "2fbdd966fae22af37f28202ca15cc76e33cdf896c8b3ad0f9df0869c06fb6edd"
PLAN = REPO / "plans/v0.0.1/024_current_training_inference_numerics_20261021.md"
PLAN_SHA = "30e039da96aecf63eb7bf5ff542855004fcc4cecd05435e1d36c1c4881564813"
ACTIVE = [True] * 10 + [False] * 118


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def check_hashes(mapping):
    for name, digest in mapping.items():
        if sha256(name) != digest:
            raise ValueError(f"frozen input changed: {name}")


def check_weights(files):
    if len(files) != 163:
        raise ValueError("024 requires the original163 actual8400 file stats")
    for name, entry in files.items():
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
            raise ValueError(f"checkpoint stat differs: {name}")


def environment(run=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ("6" if run else "") or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("024 requires explicit CUDA6 for run, empty CUDA for preparation, and empty LD")


def validate_request(request, sample):
    if (
        request["prompt"] != sample["task"]
        or request["extra"]["action128"]["task_title"] != sample["task"]
        or request["extra"]["action128"]["action_dim_mask"] != ACTIVE
        or request["vision_path"] != sample["image_path"]
        or request["seed"] != 42
        or request["action_chunk_size"] != 40
        or request["num_frames"] != 41
        or request["guidance"] != 1.0
        or request["shift"] != 5.0
        or request["num_steps"] != 35
        or request["fps"] != 20
        or request["model_mode"] != "wam"
    ):
        raise ValueError("request is not the registered correct-task H40 online condition")


def future_arrays(sample):
    """Read real future frames and labels, without a Cosmos dataloader or model."""
    with np.load(sample["arrays_path"], allow_pickle=False) as data:
        f = sample["anchor_frame"]
        result = {role: data[role][f : f + 41].copy() for role in ("auxiliary_0", "left_wrist")}
        result.update(
            physical_action=data["actions"][f : f + 40].astype(np.float32),
            action_valid=data["action_valid"][f : f + 40].astype(bool),
            action_mask=data["action_mask"][f : f + 40].astype(bool),
        )
    if (
        any(len(result[role]) != 41 for role in ("auxiliary_0", "left_wrist"))
        or result["physical_action"].shape != (40, 128)
        or result["action_mask"].shape != (40, 128)
        or result["action_valid"].shape != (40,)
        or not result["action_valid"].all()
        or not np.array_equal(result["action_mask"], np.broadcast_to(ACTIVE, (40, 128)))
        or not np.isfinite(result["physical_action"]).all()
    ):
        raise ValueError("024 needs complete true41-frame/40-label active10 futures")
    return result


def prepare(config_path=CONFIG):
    environment()
    if sha256(config_path) != CONFIG_SHA or sha256(PLAN) != PLAN_SHA:
        raise ValueError("024 registration changed")
    c = read(config_path)
    root = (REPO / c["output_dir"]).resolve()
    if not root.is_relative_to(REPO / "outputs") or root.exists():
        raise FileExistsError("024 output must be new; never overwrite or resume")
    prior_path = REPO / c["source_sampler_prepared"]
    prior = read(prior_path)
    source_path = REPO / c["source_samples_manifest"]
    samples = read(source_path)["cases"]
    selected = [next(s for s in samples if s["task_index"] == i and s["phase"] == "approach") for i in range(3)]
    weights = prior["weight_files"]
    checkpoint = c["checkpoints"][0]
    expected_checkpoint = str(Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model")
    if prior["checkpoint"] != expected_checkpoint:
        raise ValueError("011 checkpoint identity differs")
    fingerprints_path = REPO / c["source_checkpoint_fingerprints"]
    fingerprints = next(x for x in read(fingerprints_path)["checkpoints"] if x["name"] == "8400_masked")
    if fingerprints["files"] != weights or str(Path(fingerprints["checkpoint"])) != expected_checkpoint:
        raise ValueError("actual8400 byte fingerprint/stat mapping differs")
    check_weights(weights)
    # Keep just the actual read closure; unrelated old checkpoint retention is irrelevant.
    inputs = {
        str(p.resolve()): sha256(p) for p in (Path(config_path), PLAN, prior_path, source_path, fingerprints_path)
    }
    cases = []
    for sample in selected:
        for kind in ("arrays", "image"):
            p, digest = sample[f"{kind}_path"], sample[f"{kind}_sha256"]
            if sha256(p) != digest:
                raise ValueError("sample source changed")
            inputs[str(Path(p).resolve())] = digest
        future_arrays(sample)
        name = f"{sample['name']}_h40_task_only_s42"
        cached = prior["cached_cases"][name]
        request_path = Path(cached["request_path"])
        if sha256(request_path) != cached["request_sha256"]:
            raise ValueError("011 request changed")
        request = read(request_path)
        validate_request(request, sample)
        trace_path = prior_path.parent.parent / "service/traces35" / f"{name}.json"
        trace = read(trace_path)
        admission_path = prior_path.parent / "run/admission35.json"
        admission = read(admission_path)
        if (
            not admission["passed"]
            or admission["full_cases"] != 48
            or sha256(trace_path) != admission["trace_sha256"][name]
        ):
            raise ValueError("011 actual-caption trace was not admitted")
        cap = trace["conditioning"]["caption"][0]
        if (
            trace["conditioning"]["original_request_without_steps"]
            != {k: v for k, v in request.items() if k != "num_steps"}
            or json.loads(cap)["task_title"] != sample["task"]
            or json.loads(cap)["command"] != sample["task"]
            or trace["actual_generation_branch"]["guidance"] != 1.0
        ):
            raise ValueError("011 actual online condition differs")
        for p in (request_path, trace_path, admission_path):
            inputs[str(p.resolve())] = sha256(p)
        cases.append(
            dict(
                sample=sample,
                request=request,
                online_caption=cap,
                prior_request=ref(request_path),
                prior_trace=ref(trace_path),
            )
        )
    local = [
        "scripts/audit_wam_current_numerics.py",
        "tests/test_wam_current_numerics_runner.py",
        "scripts/run_wam_current_numerics.py",
        "src/s2a/worker_handshake.py",
        "scripts/verify_wam_current_numerics.py",
        "tests/test_wam_current_numerics_verification.py",
        "src/s2a/wam_runtime.py",
        "src/s2a/artifacts.py",
        "src/s2a/libero_simulation.py",
        "scripts/audit_wam_numerics.py",
        c["policy_template"],
    ]
    external = [
        "model/generator/omni_mot_model.py",
        "model/generator/mot/cosmos3_vfm_network.py",
        "model/generator/algorithm/loss/flow_matching.py",
        "model/generator/diffusion/rectified_flow.py",
        "model/generator/algorithm/loss/time_weight.py",
        "model/generator/utils/data_and_condition.py",
        "model/generator/tokenizers/wan2pt2_vae_4x16x16.py",
        "model/generator/tokenizers/interface.py",
        "model/generator/reasoner/qwen3_vl/utils.py",
        "data/generator/sequence_packing/static_layout.py",
        "data/generator/sequence_packing/sequence.py",
        "data/generator/sequence_packing/modality.py",
        "data/generator/action/unified_action128.py",
        "inference/action_server.py",
        "inference/action_serving.py",
        "inference/args.py",
        "inference/inference.py",
        "inference/common/init.py",
        "utils/misc.py",
        "utils/compile_cache.py",
    ]
    paths = [REPO / p for p in local] + [Path(c["cosmos_repo"]) / "cosmos_framework" / p for p in external]
    paths += [Path(checkpoint["run"]) / "config.yaml"]
    sources = {str(p.resolve()): sha256(p) for p in paths}
    inputs.update(sources)
    check_hashes(inputs)
    dest = root / "preparation"
    dest.mkdir(parents=True)
    snapshot = dest / "source_snapshot"
    snapshot.mkdir()
    copies = {}
    for i, (p, digest) in enumerate(sources.items()):
        target = snapshot / f"{i:03d}_{Path(p).name}"
        shutil.copyfile(p, target)
        if sha256(target) != digest:
            raise ValueError("source snapshot differs")
        copies[str(target)] = digest
    result = dict(
        schema_version="s2a.current-numerics-preparation.v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config=c,
        config_ref=ref(config_path),
        registration=ref(PLAN),
        output_root=str(root),
        cases=cases,
        checkpoint=expected_checkpoint,
        weight_files=weights,
        source_sha256=sources,
        input_sha256=inputs,
        snapshot_sha256=copies,
        budget=dict(tasks=3, sigmas=3, branches=["A", "B", "C"], denoise_calls=27, sampling_calls=0, physics_steps=0),
        new_model_calls=0,
        full_weight_bytes_rehashed=False,
    )
    write_new(dest / "prepared.json", result)
    return dest / "prepared.json"


def exact(name, left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        raise ValueError(f"exact prerequisite differs: {name}")


def numpy_tensor(value):
    # BF16 values export losslessly in FP32; preserve original dtype in the ledger.
    tensor = value.detach().cpu()
    return (tensor.float() if str(tensor.dtype) == "torch.bfloat16" else tensor).numpy().copy()


def bundle(directory, name, arrays, metadata):
    p = directory / f"{name}.npz"
    with p.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    index = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}
    j = directory / f"{name}.json"
    write_new(j, dict(metadata, arrays=ref(p), array_index=index))
    return ref(j)


class TensorDump:
    """Serialize actual dataclass tensors without changing any model object."""

    def __init__(self):
        self.arrays, self.dtypes = {}, {}

    def tensor(self, name, value):
        self.arrays[name] = numpy_tensor(value)
        self.dtypes[name] = str(value.dtype)
        return dict(npz_key=name, original_dtype=str(value.dtype))

    def tree(self, name, value):
        if hasattr(value, "detach"):
            return self.tensor(name, value)
        if dataclasses.is_dataclass(value):
            return {f.name: self.tree(f"{name}.{f.name}", getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, dict):
            return {str(k): self.tree(f"{name}.{k}", v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.tree(f"{name}.{i}", v) for i, v in enumerate(value)]
        if isinstance(value, np.ndarray):
            self.arrays[name] = value.copy()
            return dict(npz_key=name, original_dtype=str(value.dtype), original_type="numpy.ndarray")
        if isinstance(value, np.generic):
            return self.tree(name, value.item())
        if isinstance(value, enum.Enum):
            return dict(original_type=type(value).__qualname__, value=self.tree(name, value.value))
        if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
            return dict(original_type=type(value).__name__, value=str(value))
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unrecorded actual metadata type at {name}: {type(value)}")


class DenoiseBudget:
    """Count at the real denoise boundary; refuse the28th call before execution."""

    def __init__(self, original, labels):
        self.original, self.labels, self.calls = original, list(labels), []
        self.current, self.dump, self.compare, self.capture = None, None, None, None

    def arm(self, label):
        if self.current is not None or len(self.calls) >= len(self.labels) or label != self.labels[len(self.calls)]:
            raise ValueError("unregistered/repeated denoise branch or exhausted27-call budget")
        self.current = label

    def __call__(self, *args, **kwargs):
        if args or self.current is None:
            raise ValueError("denoise outside the registered A/B/C boundary")
        label, self.current = self.current, None
        packed = kwargs["data_batch_packed"]
        if np.count_nonzero(numpy_tensor(packed.action.tokens[0])[:, 10:]):
            raise ValueError("inactive packed input is nonzero before the mask wrapper")
        dump = TensorDump()
        tree = dump.tree("packed", packed)
        if self.capture is not None:
            self.capture(dump, tree)
        if self.compare is not None:
            compare_packing(self.compare, dump.arrays)
        self.calls.append(dict(label=label, completed=False))
        prediction = self.original(**kwargs)
        dump.tensor("packed_vision", packed.vision.tokens[0])
        dump.tensor("packed_action", packed.action.tokens[0])
        dump.tensor("condition_vision", packed.vision.condition_mask[0])
        dump.tensor("condition_action", packed.action.condition_mask[0])
        for kind in ("vision", "action"):
            modality = getattr(packed, kind)
            if modality.loss_valid_mask is not None:
                dump.tensor(f"valid_{kind}", modality.loss_valid_mask[0])
            dump.tensor(f"pred_{kind}", prediction[f"preds_{kind}"][0])
        if np.count_nonzero(dump.arrays["pred_action"][:, 10:]):
            raise ValueError("inactive predicted velocity is nonzero")
        self.calls[-1]["completed"] = True
        self.dump = dump, tree
        return prediction


def compare_packing(training, inference):
    """Compare model-facing metadata, excluding training-only supervision."""
    keys = [
        "packed.text_ids",
        "packed.text_indexes",
        "packed.position_ids",
        "packed.vision.condition_mask.0",
        "packed.action.condition_mask.0",
        "packed.vision.sequence_indexes",
        "packed.action.sequence_indexes",
        "packed.vision.timesteps",
        "packed.action.timesteps",
        "packed.vision.tokens.0",
        "packed.action.tokens.0",
        "packed.action.domain_id.0",
    ]
    for key in keys:
        exact(key, training[key], inference[key])


def rng_state(torch):
    """Capture all actual global RNGs; GPU visibility has already been checked."""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_rng(directory, name, state):
    n = state["numpy"]
    arrays = dict(numpy_keys=n[1].copy(), torch_cpu=numpy_tensor(state["torch_cpu"]))
    arrays.update({f"torch_cuda_{i}": numpy_tensor(x) for i, x in enumerate(state["torch_cuda"])})
    return bundle(
        directory,
        name,
        arrays,
        dict(
            python=state["python"],
            numpy=dict(algorithm=n[0], position=n[2], has_gauss=n[3], cached_gaussian=n[4]),
            cuda_devices=len(state["torch_cuda"]),
        ),
    )


def clean_exact(a, b):
    """Only text may differ between the two true-future training preparations."""
    da, db = TensorDump(), TensorDump()
    ta, tb = da.tree("clean", a), db.tree("clean", b)
    if ta != tb:
        raise ValueError("A/B clean dataclass metadata differs")
    for key in da.arrays:
        exact(key, da.arrays[key], db.arrays[key])


def require_eval(model):
    training = [name for name, module in model.named_modules() if module.training]
    if training:
        raise ValueError(f"024 requires every submodule in eval mode: {training}")


def check_flow(clean, noised, packed, comparison):
    """Check interpolation before any forward, including precision casts."""
    import torch

    for kind in ("vision", "action"):
        x0 = getattr(clean, f"x0_tokens_{kind}")[0]
        epsilon = getattr(noised, f"epsilon_{kind}")[0]
        sigma = getattr(noised, f"sigmas_{kind}")[0]
        target = epsilon - x0
        xt = (epsilon * sigma + x0 * (1 - sigma)).to(getattr(noised, f"xt_tokens_{kind}")[0].dtype)
        if kind == "action":
            valid = packed.action.loss_valid_mask[0]
            target, xt = target.masked_fill(~valid, 0), xt.masked_fill(~valid, 0)
        for name, actual, expected in (
            ("target", getattr(noised, f"vt_target_{kind}")[0], target),
            ("xt", getattr(noised, f"xt_tokens_{kind}")[0], xt),
        ):
            if not torch.allclose(actual, expected, rtol=comparison["flow_rtol"], atol=comparison["flow_atol"]):
                raise ValueError(f"{kind} flow {name} differs")
    if np.count_nonzero(numpy_tensor(noised.xt_tokens_action[0])[:, 10:]):
        raise ValueError("training inactive input is nonzero before forward")


def loss_settings(model):
    rf = model.config.rectified_flow_training_config
    if (
        model.config.predict_text_tokens
        or model.config.sound_gen
        or not model.config.vision_gen
        or not model.config.action_gen
        or rf.sample_level_loss_averaging
    ):
        raise ValueError("024 supports the frozen vision/action objective without extra scaling")
    return dict(
        vision_weight=float(rf.loss_scale),
        action_weight=float(rf.action_loss_weight),
        sample_level_loss_averaging=bool(rf.sample_level_loss_averaging),
        normalize_loss_by_active=bool(rf.normalize_loss_by_active),
        time_weight_vision=model.rectified_flow_video.train_time_weight.weight,
        time_weight_action=model.rectified_flow_action.train_time_weight.weight,
        text_weight=float(model.config.text_loss_weight),
        predict_text_tokens=bool(model.config.predict_text_tokens),
        vision_gen=bool(model.config.vision_gen),
        action_gen=bool(model.config.action_gen),
        sound_gen=bool(model.config.sound_gen),
    )


def validate_losses(dump, total, losses, settings, comparison):
    """Recompute explicit-mask reductions from recorded values in float32."""
    recomputed = {}
    for kind in ("vision", "action"):
        pred = dump.arrays[f"pred_{kind}"].astype(np.float32)
        target = dump.arrays[f"noised.vt_target_{kind}.0"].astype(np.float32)
        cond = dump.arrays[f"condition_{kind}"]
        valid = dump.arrays[f"valid_{kind}"]
        active = np.broadcast_to((1 - cond) * valid, pred.shape).astype(bool)
        den = max(int(active.sum()), 1)
        error = np.where(active, pred - target, np.float32(0)) ** 2
        tw = dump.arrays[f"time_weight_{kind}"].reshape((-1,) + (1,) * (cond.ndim - 1))
        value = float((error * tw).sum(dtype=np.float32) / den)
        actual = float(losses[f"flow_matching_loss_{kind}"])
        if not np.isfinite(actual) or not np.isclose(
            actual, value, rtol=comparison["loss_rtol"], atol=comparison["loss_atol"]
        ):
            raise ValueError(f"independent {kind} loss differs")
        recomputed[kind] = dict(loss=value, denominator=den, active_elements=int(active.sum()))
    expected = sum(recomputed[k]["loss"] * settings[f"{k}_weight"] for k in ("vision", "action"))
    if not np.isfinite(float(total)) or not np.isclose(
        float(total), expected, rtol=comparison["loss_rtol"], atol=comparison["loss_atol"]
    ):
        raise ValueError("independent total objective differs or has an unregistered term")
    return dict(modalities=recomputed, total=expected)


def save_dump(directory, name, dump, metadata):
    return bundle(directory, name, dump.arrays, dict(metadata, original_dtypes=dump.dtypes))


def training_payload(model, raw, plans, caption, video, normalized, valid, geometry):
    import torch
    from cosmos_framework.data.generator.sequence_packing.static_layout import (
        PACKED_SEQUENCE_KEY,
        build_static_packed_sequence,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_generation_caption

    payload = copy.deepcopy(raw)
    payload[model.input_video_key] = [[video]]
    payload["action"], payload["ai_caption"] = [[normalized]], [caption]
    ids, _ = tokenize_generation_caption(caption, model.vlm_tokenizer, is_video=True)
    payload["text_token_ids"] = [[torch.tensor(ids)]]
    item = dict(
        sequence_plan=plans[0],
        text_token_ids=ids,
        video=video,
        action=normalized,
        action_loss_mask=valid,
        vision_loss_mask=torch.ones((11, 60, 40), dtype=torch.bool),
        domain_id=raw["domain_id"][0],
        raw_action_dim=128,
        conditioning_fps=20.0,
        conditioning_fps_action=20.0,
        decode_metadata=raw["decode_metadata"][0],
    )
    static = build_static_packed_sequence([item], special_tokens=model.llm_special_tokens, config=geometry)
    payload[PACKED_SEQUENCE_KEY] = static
    return payload, static


def normalize_labels(torch, physical, valid, transform):
    normalized = torch.tensor(physical, dtype=torch.float32)
    mask = torch.tensor(transform["normalization_mask"], dtype=torch.bool)
    normalized[:, mask] = (
        normalized[:, mask] * torch.tensor(transform["scale"])[mask] + torch.tensor(transform["bias"])[mask]
    )
    clipped = int(((normalized[:, mask] < transform["clip_min"]) | (normalized[:, mask] > transform["clip_max"])).sum())
    normalized[:, mask] = normalized[:, mask].clamp(transform["clip_min"], transform["clip_max"])
    normalized[~valid] = 0
    return normalized, clipped


def run_case(service, case, directory, budget, comparison, settings):
    import torch
    from cosmos_framework.data.generator.sequence_packing.static_layout import (
        StaticPackingConfig,
        bind_static_packed_sequence,
    )
    from cosmos_framework.inference.args import OmniSampleOverrides
    from cosmos_framework.utils import misc

    from s2a.libero_simulation import canvas

    model, sample = service.model, case["sample"]
    directory.mkdir()
    prepared, raw = service.prepare(OmniSampleOverrides.model_validate(case["request"]), directory)
    if raw["ai_caption"] != [case["online_caption"]] or prepared.action_dim_mask != ACTIVE:
        raise ValueError("actual stock prepare differs from admitted011 caption/mask")
    require_eval(model)
    inference = model._prepare_inference_data(misc.to(copy.deepcopy(raw), device="cuda"), [42])
    plans, infer_clean, infer_text, _, initial, condition_reference, condition_mask, has_action = inference
    if not has_action or len(initial) != 1:
        raise ValueError("024 requires one WAM conditional sample")
    arrays = future_arrays(sample)
    frames = np.stack([canvas([arrays[r][j] for r in ("auxiliary_0", "left_wrist")]) for j in range(41)])
    video = torch.tensor(frames).permute(3, 0, 1, 2)
    domain = raw["normalization_domain_id"][0]
    transform = service.spec["transforms"][domain]
    valid = torch.tensor(arrays["action_mask"] & arrays["action_valid"][:, None])
    normalized, clipped = normalize_labels(torch, arrays["physical_action"], valid, transform)
    source_arrays = dict(
        raw_camera_auxiliary_0=arrays["auxiliary_0"],
        raw_camera_left_wrist=arrays["left_wrist"],
        canvas=frames,
        physical_action=arrays["physical_action"],
        normalized_action=numpy_tensor(normalized),
        action_valid=arrays["action_valid"],
        action_mask=arrays["action_mask"],
    )
    source = bundle(
        directory,
        "source",
        source_arrays,
        dict(
            sample=sample,
            domain=domain,
            transform=transform,
            clipped_label_elements=clipped,
            model_input_video_key=model.input_video_key,
            service_spec=service.spec,
            service_spec_source=ref(service.info["config_file"]),
            canvas_source_sha256=sha256(REPO / "src/s2a/libero_simulation.py"),
        ),
    )
    options = model._input_sequence_options()
    geometry = StaticPackingConfig(
        **{f.name: options[f.name] for f in dataclasses.fields(StaticPackingConfig) if f.name in options}
    )
    clean, static, clean_refs, texts = {}, {}, {}, {}
    for branch, caption in (
        ("A", sample["published_prompts"]["40"]),
        ("B", case["online_caption"]),
        ("C", case["online_caption"]),
    ):
        dump = TensorDump()
        if branch != "C":
            payload, topology = training_payload(model, raw, plans, caption, video, normalized, valid, geometry)
            raw_tree = dump.tree("raw_batch", payload)
            result = model._prepare_training_data(misc.to(payload, device="cuda"), iteration=8400)
            text, train_plans, train_clean = result[:3]
            clean[branch], static[branch], texts[branch] = train_clean, topology, text
            plan = train_plans
        else:
            raw_tree = dump.tree("raw_batch", raw)
            clean[branch], texts[branch], plan = infer_clean, infer_text, plans
            dump.tree("initial", initial)
            dump.tree("condition_reference", condition_reference)
            dump.tree("condition_mask", condition_mask)
        tree = dump.tree("clean", clean[branch])
        clean_refs[branch] = save_dump(
            directory,
            f"clean_{branch}",
            dump,
            dict(
                branch=branch,
                tree=tree,
                raw_batch_tree=raw_tree,
                plans=dump.tree("plans", plan),
                text_tokens=texts[branch],
                caption=caption,
                domain=domain,
                has_action=has_action,
                source_ref=source,
            ),
        )
    clean_exact(clean["A"], clean["B"])
    for kind in ("vision", "action"):
        for field in ("condition_mask", "loss_valid_mask"):
            exact(
                f"A/B {kind} {field}",
                numpy_tensor(getattr(getattr(static["A"], kind), field)[0]),
                numpy_tensor(getattr(getattr(static["B"], kind), field)[0]),
            )
    if texts["B"] != infer_text:
        raise ValueError("B/C actual token ids differ")
    exact("B normalized labels", numpy_tensor(clean["B"].x0_tokens_action[0]), numpy_tensor(normalized))
    for key in ("fps_vision", "fps_action", "temporal_positions_vision", "action_domain_id", "spatial_seams_vision"):
        da, db = TensorDump(), TensorDump()
        ta, tb = da.tree(key, getattr(clean["B"], key)), db.tree(key, getattr(infer_clean, key))
        if ta != tb:
            raise ValueError(f"B/C clean conditioning metadata differs: {key}")
        for name in da.arrays:
            exact(name, da.arrays[name], db.arrays[name])
    exact(
        "first conditional latent",
        numpy_tensor(clean["B"].x0_tokens_vision[0][:, :, :1]),
        numpy_tensor(infer_clean.x0_tokens_vision[0][:, :, :1]),
    )
    # The actual stock return includes flattened masks. Compare them before A's
    # first denoise rather than waiting for C to expose a preparation mismatch.
    expected_condition = torch.cat(
        (
            static["B"].vision.condition_mask[0].expand_as(clean["B"].x0_tokens_vision[0]).flatten(),
            static["B"].action.condition_mask[0].expand_as(clean["B"].x0_tokens_action[0]).flatten(),
        )
    ).float()
    exact("stock conditioning mask", numpy_tensor(expected_condition), numpy_tensor(condition_mask[0]))
    offset = infer_clean.x0_tokens_vision[0].numel()
    if np.count_nonzero(numpy_tensor(initial[0][offset:].reshape(40, 128))[:, 10:]):
        raise ValueError("stock inference after active-slot initialization has nonzero inactive slots")
    records = []
    for sigma in (0.2, 0.5, 0.8):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        paired_rng = rng_state(torch)
        paired_noised, trained_dump = None, None
        for branch in ("A", "B", "C"):
            label = f"{sample['task_index']}:{sigma}:{branch}"
            timestep = torch.tensor([[sigma * 1000]], device="cuda", dtype=torch.float32)
            require_eval(model)
            budget.capture = lambda dump, tree, sigma=sigma, branch=branch: save_dump(
                directory,
                f"sigma{sigma}_{branch}_pre_forward",
                dump,
                dict(packed_tree=tree, task_index=sample["task_index"], sigma=sigma, branch=branch),
            )
            if branch != "C":
                restore_rng(torch, paired_rng)
            before = save_rng(directory, f"sigma{sigma}_{branch}_rng_before", rng_state(torch))
            if branch != "C":
                packed = bind_static_packed_sequence(
                    copy.deepcopy(static[branch]),
                    clean[branch],
                    timestep.cpu(),
                    latent_patch_size=geometry.latent_patch_size,
                    vision_spatial_position_mode=geometry.vision_spatial_position_mode,
                )
                packed.to_cuda()
                noised = model._add_noise_to_input(clean[branch], packed, timestep / 1000)
                # Persist actual noising before checking prerequisites, including rejection cases.
                noise_dump = TensorDump()
                noised_tree = noise_dump.tree("noised", noised)
                save_dump(
                    directory, f"sigma{sigma}_{branch}_noise", noise_dump, dict(tree=noised_tree, rng_before=before)
                )
                check_flow(clean[branch], noised, packed, comparison)
                if branch == "A":
                    paired_noised = noise_dump.arrays
                else:
                    for key in paired_noised:
                        exact(f"paired A/B {key}", paired_noised[key], noise_dump.arrays[key])
                model._replace_clean_with_noised(packed, noised)
                budget.compare = None
                budget.arm(label)
                prediction = model.denoise(data_batch_packed=packed)
                dump, packed_tree = budget.dump
                noised_tree = dump.tree("noised", noised)
                total, losses = model._compute_losses(prediction, packed, noised, timestep, False)
                dump.tensor("timestep", timestep)
                dump.tensor(
                    "time_weight_vision",
                    model.rectified_flow_video.train_time_weight(timestep[0], model.tensor_kwargs_fp32),
                )
                dump.tensor(
                    "time_weight_action",
                    model.rectified_flow_action.train_time_weight(timestep[0], model.tensor_kwargs_fp32),
                )
                actual_losses = {k: numpy_tensor(v).tolist() for k, v in losses.items()}
                actual_losses["total"] = float(total)
                # Save all actual tensors even if a loss gate rejects the run.
                after = save_rng(directory, f"sigma{sigma}_{branch}_rng_after", rng_state(torch))
                call_ref = save_dump(
                    directory,
                    f"sigma{sigma}_{branch}",
                    dump,
                    dict(
                        task_index=sample["task_index"],
                        branch=branch,
                        sigma=sigma,
                        packed_tree=packed_tree,
                        noised_tree=noised_tree,
                        losses=actual_losses,
                        loss_config=settings,
                        clean_ref=clean_refs[branch],
                        rng_before=before,
                        rng_after=after,
                    ),
                )
                validate_losses(dump, total, actual_losses, settings, comparison)
                if branch == "B":
                    trained_dump = dump
                    flat = [
                        torch.cat(
                            (noised.xt_tokens_vision[0].flatten().float(), noised.xt_tokens_action[0].flatten().float())
                        )
                    ]
            else:
                budget.compare = trained_dump.arrays
                budget.arm(label)
                velocity = model._get_velocity(
                    noise_x=flat,
                    timestep=timestep.flatten(),
                    text_tokens=infer_text,
                    sequence_plans=plans,
                    gen_data_clean=infer_clean,
                    has_noisy_actions=has_action,
                )
                dump, packed_tree = budget.dump
                dump.tensor("timestep", timestep)
                dump.tensor("velocity", velocity[0])
                dump.tensor("received_xt", flat[0])
                after = save_rng(directory, f"sigma{sigma}_{branch}_rng_after", rng_state(torch))
                call_ref = save_dump(
                    directory,
                    f"sigma{sigma}_{branch}",
                    dump,
                    dict(
                        task_index=sample["task_index"],
                        branch=branch,
                        sigma=sigma,
                        packed_tree=packed_tree,
                        noised_tree=None,
                        losses=None,
                        loss_config=settings,
                        clean_ref=clean_refs[branch],
                        rng_before=before,
                        rng_after=after,
                    ),
                )
                exact(
                    "B/C active conditional velocity",
                    trained_dump.arrays["pred_action"][:, :10],
                    dump.arrays["pred_action"][:, :10],
                )
                exact(
                    "returned conditional active velocity",
                    trained_dump.arrays["pred_action"][:, :10],
                    numpy_tensor(velocity[0][offset:].reshape(40, 128))[:, :10],
                )
            records.append(dict(task_index=sample["task_index"], sigma=sigma, branch=branch, bundle=call_ref))
            print(json.dumps(dict(label=label, denoise_completed=len(budget.calls))), flush=True)
    return dict(
        sample_name=sample["name"], task_index=sample["task_index"], source=source, clean=clean_refs, calls=records
    )


def run(args):
    environment(run=True)
    if args.prepared is None or args.prepared_sha256 is None or sha256(args.prepared) != args.prepared_sha256:
        raise ValueError("024 run requires the exact reviewed preparation SHA")
    prepared = read(args.prepared)
    if sha256(args.config) != CONFIG_SHA or sha256(PLAN) != PLAN_SHA or prepared["config"] != read(args.config):
        raise ValueError("024 registered configuration changed")
    check_hashes(prepared["input_sha256"])
    check_hashes(prepared["snapshot_sha256"])
    check_weights(prepared["weight_files"])
    root = Path(prepared["output_root"])
    if args.prepared.resolve() != root / "preparation/prepared.json" or not root.is_relative_to(REPO / "outputs"):
        raise ValueError("preparation location differs")
    output = root / "run"
    output.mkdir(exist_ok=False)
    write_new(
        root / "worker_started.json",
        dict(
            schema_version="s2a.current-numerics-worker-started.v1",
            pid=os.getpid(),
            ppid=os.getppid(),
            argv=sys.argv,
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            ld_library_path=os.environ["LD_LIBRARY_PATH"],
            prepared_ref=ref(args.prepared),
            model_budget=prepared["budget"],
        ),
    )
    result = dict(
        schema_version="s2a.current-numerics-result.v1",
        complete=False,
        passed=False,
        prepared_ref=ref(args.prepared),
        cases=[],
        denoise_calls=[],
        physical_gpu=6,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        ld_library_path=os.environ["LD_LIBRARY_PATH"],
        budgets=prepared["budget"],
        sampling_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    budget = None
    try:
        sys.path.insert(0, prepared["config"]["cosmos_repo"])
        import torch

        from s2a.wam_runtime import install_active_slot_mask, load_service

        service = load_service(prepared["config"], "8400_masked", output)
        model = service.model
        require_eval(model)
        evidence = install_active_slot_mask(model, ACTIVE)
        settings = loss_settings(model)
        labels = [
            f"{task}:{sigma}:{branch}" for task in range(3) for sigma in (0.2, 0.5, 0.8) for branch in ("A", "B", "C")
        ]
        budget = DenoiseBudget(model.denoise, labels)
        model.denoise = budget
        result.update(
            torch_version=torch.__version__,
            model_dtype=str(model.tensor_kwargs["dtype"]),
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
            noising_iteration_argument=None,
            training_prepare_iteration=8400,
            loss_config=settings,
            runtime=service.info,
            all_modules_eval=True,
        )
        with torch.inference_mode():
            for case in prepared["cases"]:
                check_hashes(prepared["input_sha256"])
                check_weights(prepared["weight_files"])
                result["cases"].append(
                    run_case(
                        service,
                        case,
                        output / case["sample"]["name"],
                        budget,
                        prepared["config"]["comparison"],
                        settings,
                    )
                )
                write_new(output / f"progress{len(result['cases'])}.json", dict(result, denoise_calls=budget.calls))
        if len(budget.calls) != 27 or not all(c["completed"] for c in budget.calls):
            raise ValueError("024 did not complete the exact27 registered denoise boundaries")
        check_hashes(prepared["input_sha256"])
        check_weights(prepared["weight_files"])
        result.update(complete=True, passed=True, active_slot_mask_evidence=evidence)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result["denoise_calls"] = [] if budget is None else budget.calls
        write_new(output / "result.json", result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--seed", type=int, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    p.add_argument("--prepared", type=Path)
    p.add_argument("--prepared-sha256")
    return p


def main():
    args = parser().parse_args()
    if args.seed != 42:
        raise ValueError("024 registered seed is42")
    if args.prepare_only:
        if args.prepared is not None or args.prepared_sha256 is not None:
            raise ValueError("prepare-only cannot consume or overwrite an earlier prepared output")
        p = prepare(args.config)
        print(json.dumps(ref(p)))
    else:
        run(args)


if __name__ == "__main__":
    main()
