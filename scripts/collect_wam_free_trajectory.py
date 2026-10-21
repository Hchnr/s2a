"""029 reproduce48 raw trajectories, then collect144 matched-time teacher calls."""

import argparse
import copy
import dataclasses
import datetime
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from audit_wam_current_numerics import (
    bundle,
    check_flow,
    normalize_labels,
    numpy_tensor,
    ref,
    rng_state,
    save_rng,
    training_payload,
    write_new,
)
from prepare_wam_free_trajectory import read_prepared
from run_wam_free_trajectory import precision_admission
from run_wam_readout_features import BoundaryCounter, actual_context, parameter_versions, setup_service
from run_wam_readout_sampling import InitialCapture
from sample_wam_scaled_readout import RequestBridge, check_zero

from s2a.artifacts import REPO, sha256
from s2a.wam_flow_readout import FlowReadoutHook, FlowResidualReadout
from s2a.wam_readout import ACTIVE_MASK, ReadoutHook, ResidualReadout
from s2a.wam_trajectory_capture import (
    TrajectoryCaptureBridge,
    capture_packed,
    exact_packed_times,
    packing_config,
    require,
    tensor_array,
)


def read_ref(reference):
    require(sha256(reference["path"]) == reference["sha256"], "029 evidence SHA differs")
    return json.loads(Path(reference["path"]).read_text())


def point_arrays(reference):
    metadata = read_ref(reference)
    archive = metadata["arrays"]
    require(sha256(archive["path"]) == archive["sha256"], "029 free-point arrays changed")
    with np.load(archive["path"], allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in ("time_vision", "time_action")}
    return metadata, values


def free_generate(service, entry, directory, counter, request_bridge, initial):
    from cosmos_framework.inference.args import OmniSampleOverrides

    directory.mkdir()
    request = entry["request"]
    write_new(directory / "request.json", request)
    initial.arm(entry, directory)
    sample, batch = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(request)), directory)
    context = actual_context(service, sample, batch)
    caption = json.loads(batch["ai_caption"][0])
    require(
        context.is_libero and caption["task_title"] == caption["command"] == request["prompt"],
        "029 free input changed the actual task/full domain",
    )
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
    request_bridge.entry = entry
    with request_bridge.request(context), counter.scope(f"free:{entry['generation_index']}", 35):
        result = service.execute(sample, batch)
    request_bridge.entry = None
    response = result.model_dump(mode="json")
    write_new(directory / "response.json", response)
    initial_ref = initial.finish()
    trace = request_bridge.bridge.last
    require(trace["complete"] is True and len(trace["steps"]) == 35, "029 free solver trace incomplete")
    write_new(directory / "actual_sigma_trace.json", trace)
    return response, dict(
        response=ref(directory / "response.json"),
        conditioning=ref(directory / "conditioning.json"),
        initial=initial_ref,
        actual_sigma_trace=ref(directory / "actual_sigma_trace.json"),
        points=request_bridge.bridge.last_points,
    )


def teacher_anchor(
    service, prepared, anchor, entries, output, counter, hook, capture, free_points, free_complete, event_index
):
    import torch
    from cosmos_framework.data.generator.sequence_packing.static_layout import (
        StaticPackingConfig,
        bind_static_packed_sequence,
    )
    from cosmos_framework.inference.args import OmniSampleOverrides
    from cosmos_framework.utils import misc

    from s2a.libero_simulation import canvas

    output.mkdir()
    require(
        sha256(free_complete["path"]) == free_complete["sha256"], "029 free completion changed before future access"
    )
    require(
        len(counter.calls) >= 1680 and all(x["completed"] for x in counter.calls),
        "029 future opened before free completion",
    )
    event = dict(
        event_index=event_index,
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        free_complete_ref=free_complete,
        anchor_name=anchor["name"],
        arrays_ref=dict(path=anchor["arrays_path"], sha256=anchor["arrays_sha256"]),
        denoise_calls_before=len(counter.calls),
        first_teacher_call_index=entries[0]["call_index"],
        future_source_opened_after_free_complete=True,
    )
    write_new(output / "source_open.json", event)
    require(anchor["split"] == "validation", "029 may not open a training episode")
    require(sha256(anchor["arrays_path"]) == anchor["arrays_sha256"], "029 future source SHA differs")
    with np.load(anchor["arrays_path"], allow_pickle=False) as source:
        frame = anchor["anchor_frame"]
        cameras = {key: source[key][frame : frame + 41].copy() for key in ("auxiliary_0", "left_wrist")}
        physical = source["actions"][frame : frame + 40].copy()
        mask = source["action_mask"][frame : frame + 40].copy()
        action_valid = source["action_valid"][frame : frame + 40].copy()
    valid_np = mask & action_valid[:, None]
    require(
        physical.shape == (40, 128) and np.array_equal(valid_np, np.broadcast_to(ACTIVE_MASK, (40, 128))),
        "029 teacher requires a complete active10 future",
    )
    frames = np.stack([canvas([cameras[key][j] for key in ("auxiliary_0", "left_wrist")]) for j in range(41)])
    model = service.model
    sample, raw = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(anchor["request"])), output)
    context = actual_context(service, sample, raw)
    caption = json.loads(raw["ai_caption"][0])
    require(
        context.is_libero and caption["task_title"] == caption["command"] == anchor["task"],
        "029 teacher task/full domain differs",
    )
    valid = torch.tensor(valid_np, dtype=torch.bool)
    transform = service.spec["transforms"][context.normalization_id]
    normalized, clipped = normalize_labels(torch, physical, valid, transform)
    source_ref = bundle(
        output,
        "source",
        dict(
            physical_action=physical,
            normalized_action=numpy_tensor(normalized),
            valid=valid_np,
            action_mask=mask,
            action_valid=action_valid,
            raw_camera_auxiliary_0=cameras["auxiliary_0"],
            raw_camera_left_wrist=cameras["left_wrist"],
            canvas=frames,
        ),
        dict(
            anchor=anchor,
            source_open_ref=ref(output / "source_open.json"),
            free_complete_ref=free_complete,
            transform=transform,
            context=dataclasses.asdict(context),
            true_future_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
            canvas_source_sha256=sha256(REPO / "src/s2a/libero_simulation.py"),
            clipped_label_elements=clipped,
            model_input_video_key=model.input_video_key,
        ),
    )
    plans, _, text = model._prepare_inference_data(misc.to(copy.deepcopy(raw), device="cuda"), [42])[:3]
    options = model._input_sequence_options()
    geometry = packing_config(StaticPackingConfig, options)
    video = torch.tensor(frames).permute(3, 0, 1, 2)
    payload, topology = training_payload(model, raw, plans, raw["ai_caption"][0], video, normalized, valid, geometry)
    train_text, _, clean = model._prepare_training_data(misc.to(payload, device="cuda"), iteration=8400)[:3]
    require(train_text == text, "029 teacher preparation changed actual text")
    require(
        np.array_equal(numpy_tensor(clean.x0_tokens_action[0]), numpy_tensor(normalized)),
        "029 x0 normalization differs",
    )
    write_new(
        output / "conditioning.json",
        dict(
            caption=raw["ai_caption"][0],
            context=dataclasses.asdict(context),
            source_ref=source_ref,
            inference_only_current_image=False,
        ),
    )
    result = []
    for entry in entries:
        name = f"call{entry['call_index']:03d}"
        point_ref = free_points[(entry["free_generation_index"], entry["step_index"])]
        point, times = point_arrays(point_ref)
        require(
            point["sigma_actual"] == entry["sigma_actual"] and point["network_timestep"] == entry["network_timestep"],
            "029 paired free sigma/network time differs",
        )
        seed = entry["sampling_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        before = save_rng(output, f"{name}_rng_before", rng_state(torch))
        network_time = torch.tensor([[entry["network_timestep"]]], dtype=torch.int64, device="cuda")
        sigma = torch.tensor([[entry["sigma_actual"]]], dtype=torch.float32, device="cuda")
        require(float(sigma.item()) == entry["sigma_actual"], "029 actual sigma was not exactly FP32")
        packed = bind_static_packed_sequence(
            copy.deepcopy(topology),
            clean,
            network_time.cpu(),
            latent_patch_size=geometry.latent_patch_size,
            vision_spatial_position_mode=geometry.vision_spatial_position_mode,
        )
        packed.to_cuda()
        noised = model._add_noise_to_input(clean, packed, sigma)
        check_flow(clean, noised, packed, dict(flow_rtol=1e-6, flow_atol=1e-7))
        model._replace_clean_with_noised(packed, noised)
        exact_packed_times(packed, times, point["original_dtypes"])
        arrays, dtypes = capture_packed(packed)
        capture.clear()
        with hook.request(context), counter.scope(f"teacher:{entry['call_index']}", 1):
            prediction = model.denoise(data_batch_packed=packed)
        require(len(capture) == 1, "029 teacher must invoke its original head exactly once")
        head_arrays, head_dtypes = capture[0]
        arrays.update(head_arrays)
        dtypes.update(head_dtypes)
        tensors = dict(
            pred_action=prediction["preds_action"][0],
            target=noised.vt_target_action[0],
            epsilon=noised.epsilon_action[0],
            x0_normalized=clean.x0_tokens_action[0],
            sigma_action=noised.sigmas_action[0],
            x0_vision=clean.x0_tokens_vision[0],
            epsilon_vision=noised.epsilon_vision[0],
            target_vision=noised.vt_target_vision[0],
            sigma_vision=noised.sigmas_vision[0],
            network_timestep=network_time,
        )
        tensors["velocity_action"] = prediction["preds_action"][0] * (1.0 - packed.action.condition_mask[0]).to(
            prediction["preds_action"][0]
        )
        tensors["velocity_vision"] = prediction["preds_vision"][0] * (1.0 - packed.vision.condition_mask[0]).to(
            prediction["preds_vision"][0]
        )
        for key, value in tensors.items():
            arrays[key], dtypes[key] = tensor_array(value), str(value.dtype)
        arrays["valid"] = valid_np.copy()
        dtypes["valid"] = "torch.bool"
        require(
            np.array_equal(arrays["pred_action"][:, :10], arrays["base"][:, :10]), "029 teacher captured head differs"
        )
        after = save_rng(output, f"{name}_rng_after", rng_state(torch))
        recorded = bundle(
            output,
            name,
            arrays,
            dict(
                **entry,
                original_dtypes=dtypes,
                context=dataclasses.asdict(context),
                conditioning_ref=ref(output / "conditioning.json"),
                source_ref=source_ref,
                free_point_ref=point_ref,
                free_complete_ref=free_complete,
                rng_before=before,
                rng_after=after,
                packed_network_times_exact=True,
                head_calls=1,
                denoise_calls=1,
                supervision="epsilon-x0; distinct from free point pull velocity",
                velocity_source="actual denoise predictions with the frozen inference condition-mask operation",
            ),
        )
        result.append(dict(entry, bundle=recorded))
        print(json.dumps(dict(teacher_completed=len(counter.calls) - 1680, name=entry["name"])), flush=True)
    return result, ref(output / "source_open.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "precision", "precision-verification"):
        parser.add_argument(f"--{key}", type=Path, required=True)
        parser.add_argument(f"--{key}-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "6" and os.environ.get("LD_LIBRARY_PATH") == "",
        "029 worker requires physicalGPU6 and emptyLD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    admission = (
        prepared,
        args.prepared,
        args.precision,
        args.precision_sha256,
        args.precision_verification,
        args.precision_verification_sha256,
    )
    precision_admission(*admission)
    old = read_ref(prepared["source_prepared_ref"])
    require(
        len(prepared["free_generations"]) == 48 and len(prepared["teacher_calls"]) == 144, "029 finite roster differs"
    )
    for anchor in prepared["anchors"]:
        require(
            anchor["split"] == "validation" and sha256(anchor["image_path"]) == anchor["image_sha256"],
            "029 current image/source differs",
        )
    output = Path(prepared["output_root"]) / "collection"
    output.mkdir()
    write_new(output / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), cuda="6", ld=""))
    result = dict(
        schema_version="s2a.free-trajectory-collection.v1",
        complete=False,
        passed=False,
        prepared_ref=ref(args.prepared),
        precision_ref=ref(args.precision),
        precision_verification_ref=ref(args.precision_verification),
        free_generations=[],
        teacher_calls=[],
        teacher_source_opens=[],
        wam_model_initializations=0,
        fit_groups=0,
        physics_steps=0,
        fresh_cases=0,
        train_episode_reads=0,
    )
    bridge = hook = initial = counter = teacher_observer = None
    try:
        import torch

        service_dir = output / "service"
        service_dir.mkdir()
        service = setup_service(old, service_dir)
        result["wam_model_initializations"] = 1
        model = service.model
        require(all(not p.requires_grad for p in model.parameters()), "029 original parameter gradients are enabled")
        before = parameter_versions(model)
        counter = BoundaryCounter(model, 1824)
        hook = FlowReadoutHook(model.net.llm2action, FlowResidualReadout.zero())
        sampler_path = str(
            Path(prepared["config"]["cosmos_repo"]) / "cosmos_framework/model/generator/diffusion/samplers/unipc.py"
        )
        convert_path = str(Path(sampler_path).with_name("fm_solvers_unipc.py"))
        generator_path = str(
            Path(prepared["config"]["cosmos_repo"]) / "cosmos_framework/model/generator/omni_mot_model.py"
        )
        sources = {path: prepared["source_sha256"][path] for path in (sampler_path, convert_path, generator_path)}
        active = {}

        def rng_capture(phase, record):
            return save_rng(active["directory"], f"step{record['step_index']}_rng_{phase}", rng_state(torch))

        def sink(record, arrays, metadata):
            require(arrays["packed_vision"].shape == (1, 48, 11, 60, 40), "029 actual packed vision geometry differs")
            require(
                arrays["hidden"].shape == (40, 1280) and arrays["base"].shape == (40, 128),
                "029 actual head geometry differs",
            )
            return bundle(
                active["directory"],
                f"step{record['step_index']}",
                arrays,
                dict(**metadata, **active["entry"], conditioning_ref=ref(active["directory"] / "conditioning.json")),
            )

        bridge = TrajectoryCaptureBridge(
            model,
            hook,
            sink=sink,
            rng_capture=rng_capture,
            forward_file=sampler_path,
            forward_sha256=sources[sampler_path],
            sources=sources,
            convert_file=convert_path,
            convert_sha256=sources[convert_path],
        )
        initial = InitialCapture(model)
        request_bridge = RequestBridge(bridge)
        with torch.inference_mode():
            for entry in prepared["free_generations"]:
                directory = output / entry["name"]
                active.update(directory=directory, entry=entry)
                response, refs = free_generate(service, entry, directory, counter, request_bridge, initial)
                check_zero(response, read_ref(entry["cache"]["response"]))
                result["free_generations"].append(dict(entry, **refs))
                print(
                    json.dumps(dict(free_completed=len(result["free_generations"]), denoise_calls=len(counter.calls))),
                    flush=True,
                )
            require(
                len(counter.calls) == 1680 and parameter_versions(model) == before,
                "029 free boundary or parameters changed",
            )
            require(bridge.total_head_calls == 1680, "029 free head budget differs")
            free_complete_path = output / "free_complete.json"
            write_new(
                free_complete_path,
                dict(
                    complete=True,
                    passed=True,
                    event_index=0,
                    prepared_ref=ref(args.prepared),
                    free_generations=result["free_generations"],
                    denoise_calls=1680,
                    denoise_ledger=list(counter.calls),
                    all_original_parameters_frozen=True,
                    original_parameters_require_grad_false=True,
                    teacher_calls=0,
                    future_source_opens=0,
                ),
            )
            result["free_complete_ref"] = ref(free_complete_path)
            free_points = {
                (entry["generation_index"], read_ref(point)["step_index"]): point
                for entry in result["free_generations"]
                for point in entry["points"]
            }
            initial.remove()
            initial = None
            bridge.restore()
            bridge = None
            hook.remove()
            hook = ReadoutHook(model.net.llm2action, ResidualReadout.zero())
            capture, teacher_heads = [], []

            def observe_teacher(module, values, kwargs, base):
                require(
                    counter.current is not None and counter.current["label"].startswith("teacher:"),
                    "029 teacher head outside scope",
                )
                require(not kwargs and len(values) == 2, "029 teacher head arguments differ")
                require(
                    values[0].shape == (40, 1280) and base.shape == (40, 128),
                    "029 teacher actual head geometry differs",
                )
                values = dict(hidden=values[0], projection_ids=values[1], base=base)
                capture.append(
                    ({k: tensor_array(v) for k, v in values.items()}, {k: str(v.dtype) for k, v in values.items()})
                )
                teacher_heads.append(1)

            teacher_observer = model.net.llm2action.register_forward_hook(observe_teacher, with_kwargs=True)
            teacher_dir = output / "teacher"
            teacher_dir.mkdir()
            for event_index, anchor in enumerate(prepared["anchors"], start=1):
                entries = [e for e in prepared["teacher_calls"] if e["anchor_name"] == anchor["name"]]
                require(len(entries) == 6, "029 teacher anchor requires both seeds and three exact times")
                recorded, source_open = teacher_anchor(
                    service,
                    prepared,
                    anchor,
                    entries,
                    teacher_dir / anchor["name"],
                    counter,
                    hook,
                    capture,
                    free_points,
                    result["free_complete_ref"],
                    event_index,
                )
                result["teacher_calls"].extend(recorded)
                result["teacher_source_opens"].append(source_open)
        require(
            len(counter.calls) == 1824 and len(result["teacher_calls"]) == len(teacher_heads) == 144,
            "029 full diagnostic budget differs",
        )
        require(parameter_versions(model) == before, "029 original model parameters changed")
        read_prepared(args.prepared, args.prepared_sha256)
        precision_admission(*admission)
        result.update(
            complete=True,
            passed=True,
            all_original_parameters_frozen=True,
            original_parameters_require_grad_false=True,
            head_call_counts=dict(free=1680, teacher=144),
            runtime=service.info,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if teacher_observer is not None:
            teacher_observer.remove()
        if initial is not None:
            initial.remove()
        if bridge is not None:
            bridge.restore()
        if hook is not None:
            hook.remove()
        result["denoise_calls"] = 0 if counter is None else len(counter.calls)
        result["denoise_ledger"] = [] if counter is None else counter.calls
        write_new(output / "worker_result.json", result)


if __name__ == "__main__":
    main()
