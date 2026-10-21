"""025 phase1: three zero-head reproductions, then144 frozen teacher forwards."""

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from audit_wam_current_numerics import (
    bundle,
    check_flow,
    check_hashes,
    check_weights,
    normalize_labels,
    numpy_tensor,
    ref,
    require_eval,
    rng_state,
    save_rng,
    training_payload,
    write_new,
)

from s2a.artifacts import sha256
from s2a.wam_readout import ACTIVE_MASK, DomainContext, ReadoutHook, ResidualReadout


def read_prepared(path, digest):
    if sha256(path) != digest:
        raise ValueError("025 preparation SHA differs")
    prepared = json.loads(Path(path).read_text())
    if prepared["schema_version"] != "s2a.frozen-readout-preparation.v1":
        raise ValueError("025 preparation schema differs")
    for name in ("input_sha256", "source_sha256", "snapshot_sha256"):
        check_hashes(prepared[name])
    check_weights(prepared["weight_files"])
    return prepared


def environment():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("025 model workers require physicalGPU6 and empty LD")


def setup_service(prepared, output):
    from s2a.wam_runtime import install_active_slot_mask, load_service

    c = prepared["config"]
    service_config = dict(c, checkpoints=[c["checkpoint"]])
    service = load_service(service_config, "8400_masked", output)
    if service.info["checkpoint_path"] != prepared["checkpoint"] or service.info["ema_weights"]:
        raise ValueError("025 loaded weights differ")
    install_active_slot_mask(service.model, ACTIVE_MASK)
    require_eval(service.model)
    for parameter in service.model.parameters():
        parameter.requires_grad_(False)
    return service


def parameter_versions(model):
    return {name: (tuple(p.shape), p.data_ptr(), p._version, p.requires_grad) for name, p in model.named_parameters()}


def actual_context(service, request, batch):
    spec = service.spec
    domain = request.action_domain
    normalization = int(batch["normalization_domain_id"][0])
    projection = int(batch["domain_id"][0])
    if spec["registry"]["domains"][normalization] != domain:
        raise ValueError("025 actual normalization domain does not match full request domain")
    if spec["registry"]["domain_projection_ids"][normalization] != projection:
        raise ValueError("025 actual projection routing differs from the saved domain contract")
    return DomainContext(domain, normalization, projection, tuple(request.action_dim_mask))


class BoundaryCounter:
    """Only an armed generation/teacher scope may call the real masked denoiser."""

    def __init__(self, model, maximum):
        self.original = model.denoise
        self.maximum, self.current, self.calls = maximum, None, []
        model.denoise = self.forward

    @contextmanager
    def scope(self, label, count):
        if self.current is not None:
            raise ValueError("025 nested forward scope")
        self.current = dict(label=label, remaining=count)
        try:
            yield
            if self.current["remaining"] != 0:
                raise ValueError("025 actual NFE does not match its registered scope")
        finally:
            self.current = None

    def forward(self, *args, **kwargs):
        if args or self.current is None or self.current["remaining"] <= 0 or len(self.calls) >= self.maximum:
            raise ValueError("025 unregistered forward or exhausted budget")
        packed = kwargs["data_batch_packed"]
        if np.any(numpy_tensor(packed.action.tokens[0])[:, 10:]):
            raise ValueError("025 inactive action input was not zero before the model boundary")
        self.current["remaining"] -= 1
        record = dict(index=len(self.calls), label=self.current["label"], completed=False)
        self.calls.append(record)
        result = self.original(**kwargs)
        if np.any(numpy_tensor(result["preds_action"][0])[:, 10:]):
            raise ValueError("025 inactive action velocity is not zero")
        record["completed"] = True
        return result


def generate(service, request, directory, counter, hook, label):
    from cosmos_framework.inference.args import OmniSampleOverrides

    directory.mkdir()
    write_new(directory / "request.json", request)
    sample, batch = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(request)), directory)
    context = actual_context(service, sample, batch)
    if not context.is_libero:
        raise ValueError("025 registered requests must all be LIBERO")
    caption = json.loads(batch["ai_caption"][0])
    if caption["task_title"] != request["prompt"] or caption["command"] != request["prompt"]:
        raise ValueError("025 actual online text differs from current task-only input")
    with hook.request(context), counter.scope(label, 35):
        result = service.execute(sample, batch)
    response = result.model_dump(mode="json")
    write_new(directory / "response.json", response)
    write_new(
        directory / "conditioning.json",
        dict(
            caption=batch["ai_caption"][0],
            caption_sha256=hashlib.sha256(batch["ai_caption"][0].encode()).hexdigest(),
            context=dataclasses.asdict(context),
            model_runtime=service.info,
            inference_only_current_image=True,
        ),
    )
    return response


def teacher_anchor(service, prepared, anchor, entries, directory, counter, hook, capture):
    import torch
    from cosmos_framework.data.generator.sequence_packing.static_layout import (
        StaticPackingConfig,
        bind_static_packed_sequence,
    )
    from cosmos_framework.inference.args import OmniSampleOverrides
    from cosmos_framework.utils import misc

    from s2a.libero_simulation import canvas

    directory.mkdir()
    model = service.model
    sample, raw = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(anchor["request"])), directory)
    context = actual_context(service, sample, raw)
    caption = json.loads(raw["ai_caption"][0])
    if not context.is_libero or caption["task_title"] != anchor["task"] or caption["command"] != anchor["task"]:
        raise ValueError("025 teacher actual domain/task condition differs")
    with np.load(anchor["arrays_path"], allow_pickle=False) as source:
        frame = anchor["anchor_frame"]
        frames = np.stack(
            [canvas([source[role][j] for role in ("auxiliary_0", "left_wrist")]) for j in range(frame, frame + 41)]
        )
        physical = source["actions"][frame : frame + 40].copy()
        valid_np = source["action_mask"][frame : frame + 40] & source["action_valid"][frame : frame + 40, None]
    if physical.shape != (40, 128) or not np.array_equal(valid_np, np.broadcast_to(ACTIVE_MASK, (40, 128))):
        raise ValueError("025 teacher window is not fully valid active10")
    valid = torch.tensor(valid_np, dtype=torch.bool)
    transform = service.spec["transforms"][context.normalization_id]
    normalized, clipped = normalize_labels(torch, physical, valid, transform)
    inference = model._prepare_inference_data(misc.to(copy.deepcopy(raw), device="cuda"), [42])
    plans, _, text = inference[:3]
    options = model._input_sequence_options()
    geometry = StaticPackingConfig(
        **{
            field.name: options[field.name]
            for field in dataclasses.fields(StaticPackingConfig)
            if field.name in options
        }
    )
    video = torch.tensor(frames).permute(3, 0, 1, 2)
    payload, topology = training_payload(model, raw, plans, raw["ai_caption"][0], video, normalized, valid, geometry)
    train_text, _, clean = model._prepare_training_data(misc.to(payload, device="cuda"), iteration=8400)[:3]
    if train_text != text or not np.array_equal(numpy_tensor(clean.x0_tokens_action[0]), numpy_tensor(normalized)):
        raise ValueError("025 teacher preparation changed text or true action labels")
    write_new(
        directory / "conditioning.json",
        dict(
            caption=raw["ai_caption"][0],
            context=dataclasses.asdict(context),
            anchor_name=anchor["name"],
            arrays_ref=dict(path=anchor["arrays_path"], sha256=anchor["arrays_sha256"]),
            source_frame=frame,
            clipped_label_elements=clipped,
            true_future_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
        ),
    )
    records = []
    for entry in entries:
        sigma = entry["sigma"]
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        before = save_rng(directory, f"sigma{sigma}_rng_before", rng_state(torch))
        timestep = torch.tensor([[sigma * 1000]], device="cuda", dtype=torch.float32)
        packed = bind_static_packed_sequence(
            copy.deepcopy(topology),
            clean,
            timestep.cpu(),
            latent_patch_size=geometry.latent_patch_size,
            vision_spatial_position_mode=geometry.vision_spatial_position_mode,
        )
        packed.to_cuda()
        noised = model._add_noise_to_input(clean, packed, timestep / 1000)
        check_flow(clean, noised, packed, dict(flow_rtol=1e-6, flow_atol=1e-7))
        model._replace_clean_with_noised(packed, noised)
        capture.clear()
        with hook.request(context), counter.scope(f"teacher:{entry['call_index']}", 1):
            prediction = model.denoise(data_batch_packed=packed)
        if len(capture) != 1:
            raise ValueError("025 teacher did not observe exactly one original output-head invocation")
        observed = capture[0]
        base = numpy_tensor(prediction["preds_action"][0])
        if not np.array_equal(base[:, :10], observed["head_base"][:, :10]):
            raise ValueError("025 captured head is not the actual active velocity output")
        hidden_dtype = observed.pop("hidden_dtype")
        head_dtype = observed.pop("head_dtype")
        arrays = dict(
            **observed,
            base=base,
            target=numpy_tensor(noised.vt_target_action[0]),
            xt=numpy_tensor(noised.xt_tokens_action[0]),
            valid=valid_np.copy(),
            epsilon=numpy_tensor(noised.epsilon_action[0]),
            x0_normalized=numpy_tensor(clean.x0_tokens_action[0]),
            sigma_action=numpy_tensor(noised.sigmas_action[0]),
            condition_action=numpy_tensor(packed.action.condition_mask[0]),
        )
        after = save_rng(directory, f"sigma{sigma}_rng_after", rng_state(torch))
        reference = bundle(
            directory,
            f"sigma{sigma}",
            arrays,
            dict(
                **entry,
                original_dtypes=dict(
                    base=head_dtype, hidden=hidden_dtype, model_velocity=str(prediction["preds_action"][0].dtype)
                ),
                conditioning_ref=ref(directory / "conditioning.json"),
                rng_before=before,
                rng_after=after,
            ),
        )
        records.append(dict(entry, bundle=reference))
        print(
            json.dumps(dict(teacher_completed=len(counter.calls) - 105, anchor=anchor["name"], sigma=sigma)), flush=True
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    environment()
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    output = Path(prepared["output_root"]) / "features"
    output.mkdir(exist_ok=False)
    write_new(output / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), cuda="6", ld=""))
    result = dict(
        schema_version="s2a.frozen-readout-features.v1",
        complete=False,
        passed=False,
        prepared_ref=ref(args.prepared),
        zero_identity=[],
        teacher_calls=[],
        denoise_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    counter = hook = capture_handle = None
    try:
        import torch

        service_dir = output / "service"
        service_dir.mkdir()
        service = setup_service(prepared, service_dir)
        model = service.model
        before_parameters = parameter_versions(model)
        hook = ReadoutHook(model.net.llm2action, ResidualReadout.zero())
        counter = BoundaryCounter(model, 249)
        capture = []

        def observe_head(module, values, kwargs, base):
            if counter.current is not None and counter.current["label"].startswith("teacher:"):
                hidden, domains = values
                if kwargs or hidden.shape != (40, 1280) or base.shape != (40, 128):
                    raise ValueError("025 actual output-head feature geometry differs")
                capture.append(
                    dict(
                        hidden=numpy_tensor(hidden),
                        hidden_dtype=str(hidden.dtype),
                        head_dtype=str(base.dtype),
                        head_base=numpy_tensor(base),
                        projection_ids=numpy_tensor(domains),
                    )
                )

        capture_handle = model.net.llm2action.register_forward_hook(observe_head, with_kwargs=True)
        with torch.inference_mode():
            for item in prepared["zero_identity_cases"]:
                directory = output / f"zero_identity_task{item['task_index']}"
                response = generate(service, item["request"], directory, counter, hook, f"zero:{item['task_index']}")
                original = json.loads(Path(item["cache"]["response_path"]).read_text())
                actual, expected = (x["outputs"][0]["content"] for x in (response, original))
                for a, b in (
                    (actual["action"], expected["action"]),
                    (actual["action128"]["normalized_action"], expected["action128"]["normalized_action"]),
                ):
                    if not np.array_equal(np.asarray(a), np.asarray(b)):
                        raise ValueError("025 zero readout did not reproduce complete raw/normalized128 output")
                result["zero_identity"].append(ref(directory / "response.json"))
            for anchor in prepared["anchors"]:
                entries = [x for x in prepared["teacher_calls"] if x["anchor_name"] == anchor["name"]]
                if len(entries) != 3:
                    raise ValueError("025 registered teacher anchor has wrong sigma roster")
                result["teacher_calls"].extend(
                    teacher_anchor(service, prepared, anchor, entries, output / anchor["name"], counter, hook, capture)
                )
        if len(counter.calls) != 249 or len(result["teacher_calls"]) != 144:
            raise ValueError("025 phase1 did not finish its complete registered forward budget")
        if parameter_versions(model) != before_parameters:
            raise ValueError("025 original model parameter identity/version/gradient flags changed")
        for name in ("input_sha256", "source_sha256", "snapshot_sha256"):
            check_hashes(prepared[name])
        check_weights(prepared["weight_files"])
        result.update(complete=True, passed=True, all_original_parameters_frozen=True, runtime=service.info)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if capture_handle is not None:
            capture_handle.remove()
        if hook is not None:
            hook.remove()
        result["denoise_calls"] = 0 if counter is None else len(counter.calls)
        result["denoise_ledger"] = [] if counter is None else counter.calls
        write_new(output / "result.json", result)


if __name__ == "__main__":
    main()
