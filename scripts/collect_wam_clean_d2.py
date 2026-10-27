"""056 single fixed052-policy D2 capture, with original training initial pairs.

The model sampling/capture body is derived from the established038 collector.
Admission and experiment identity are independent; no old D1 authorization is used.
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from collect_wam_clean_training import normalization_spec, qualified_grid, same_initial, validate_training_caption
from qualify_wam_clean_bridge import (
    CONVERSION_SOURCE,
    SAMPLER_SOURCE,
    ConversionWitness,
    imported_cosmos_sources,
    response_arrays,
    runtime_environment,
)

from s2a.artifacts import REPO
from s2a.clean_training_d2_execution import admit
from s2a.clean_training_runtime import (
    GPU,
    VISION_SHAPE,
    checked,
    publish,
    read,
    reference,
    require,
)
from s2a.sampler_trace import check_sources, parameters_identity
from s2a.wam_clean_capture import NoiseRoster
from s2a.wam_clean_compact import CompactCaptureSink, CompactRequest
from s2a.wam_readout import ACTIVE_MASK


def run(prepared_ref, registration_ref, *, gpu_uuid, dry=False):
    required = [Path(__file__), REPO / "src/s2a/clean_training_d2_execution.py", REPO / "src/s2a/wam_clean_compact.py"]
    prepared, _, sources, head, params_ref, d0 = admit(prepared_ref, registration_ref, required_sources=required)
    require(gpu_uuid == GPU, "056 allowed UUID required")
    if dry:
        return dict(
            complete=True,
            passed=True,
            requests=300,
            model_initializations=0,
            future_label_reads=0,
            prepared_ref=prepared_ref,
            registration_ref=registration_ref,
        )
    qualified = read(prepared["prerequisite_refs"]["bridge_prepared"])
    runtime_environment(qualified, gpu_uuid)
    grid = qualified_grid(prepared)
    round_name = "D2"
    directory = Path(prepared["root"]) / "capture" / round_name
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.clean-d2-capture.v1",
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        round_name=round_name,
        readout_ref=params_ref,
        current_manifest_ref=prepared["current_manifest_ref"],
        requests=[],
        actual_requests=0,
        actual_denoise_calls=0,
        actual_head_calls=0,
        model_initialization_attempts=0,
        model_initializations=0,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
        training_seed_plan_ref=prepared["training_seed_plan_ref"],
        source_sha256=sources,
        gpu_uuid=gpu_uuid,
    )
    bridge = counter = initial = model = None
    try:
        import torch
        from audit_wam_current_numerics import require_eval
        from run_wam_readout_features import BoundaryCounter, actual_context
        from run_wam_readout_sampling import InitialCapture

        from s2a.wam_runtime import install_active_slot_mask, load_service

        require(
            gpu_uuid == GPU and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "exactly the registered UUID must be visible",
        )
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        require(actual_uuid.removeprefix("GPU-") == GPU.removeprefix("GPU-"), "actual GPU UUID differs")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        service_dir = directory / "service"
        service_dir.mkdir()
        checkpoint = prepared["model_config"]["checkpoint"]
        result["model_initialization_attempts"] = 1
        service = load_service(dict(checkpoints=[checkpoint]), "8400_masked", service_dir)
        result["model_initializations"] = 1
        model = service.model
        require(
            Path(service.info["checkpoint_path"]).resolve()
            == (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
            and service.info["ema_weights"] is False,
            "actual checkpoint/EMA differs",
        )
        require_eval(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        result["parameters_before"] = parameters_identity(model)
        result["mask_evidence"] = install_active_slot_mask(model, ACTIVE_MASK)
        result["cosmos_imported_sources_after_load"] = imported_cosmos_sources(qualified)
        normalization = normalization_spec(service.spec)
        if d0 is None:
            result["normalization_ref"] = publish(directory / "normalization.json", normalization)
        else:
            require(normalization == read(d0["normalization_ref"]), "D0/D1 actual normalization differs")
            result["normalization_ref"] = d0["normalization_ref"]
        cosmos = Path(qualified["config"]["cosmos_source_root"])
        sampler, conversion = (str((cosmos / p).resolve()) for p in (SAMPLER_SOURCE, CONVERSION_SOURCE))
        active_sink = None

        def sink(point):
            require(active_sink is not None, "point outside registered request")
            active_sink(point)

        bridge = ConversionWitness(
            model,
            noise_roster=NoiseRoster(**prepared["noise_roster"]),
            readout=head,
            feature_terms=(),
            sink=sink,
            denoise_delegate=model.denoise,
            forward_file=sampler,
            forward_sha256=sources[sampler],
            sources=sources,
            conversion_source_ref=dict(path=conversion, sha256=sources[conversion]),
        )
        counter, initial = BoundaryCounter(model, 10500), InitialCapture(model)
        anchors = {a["name"]: a for a in prepared["training_anchors"]}
        from cosmos_framework.inference.args import OmniSampleOverrides

        with torch.inference_mode():
            for entry in prepared["training_requests"]:
                request, anchor = entry["request"], anchors[entry["anchor_name"]]
                current = directory / f"{entry['index']:03d}_{entry['name']}"
                current.mkdir()
                record = {key: entry[key] for key in ("index", "name", "anchor_name", "sampling_seed")}
                record["complete"] = False
                result["requests"].append(record)
                record["request_ref"] = publish(current / "request.json", request)
                active_sink = CompactCaptureSink(
                    current / "compact",
                    CompactRequest(
                        entry["name"],
                        entry["sampling_seed"],
                        "training",
                        round_name,
                        entry["anchor_name"],
                        None if params_ref is None else params_ref["sha256"],
                        anchor["image_ref"]["sha256"],
                    ),
                    expected_vision_shape=VISION_SHAPE,
                    expected_grid=grid,
                )
                initial.arm(
                    dict(
                        generation_index=entry["index"],
                        name=entry["name"],
                        pair_id=entry["name"],
                        variant="probe052",
                        sampling_seed=entry["sampling_seed"],
                    ),
                    current,
                )
                checked(anchor["image_ref"])
                sample, batch = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(request)), current)
                checked(anchor["image_ref"])
                context = actual_context(service, sample, batch)
                caption = json.loads(batch["ai_caption"][0])
                validate_training_caption(caption, request["prompt"], entry["sampling_seed"])
                require(
                    context.is_libero,
                    "actual conditioning contains noncurrent or privileged input",
                )
                record["conditioning_ref"] = publish(
                    current / "conditioning.json",
                    dict(
                        caption=batch["ai_caption"][0],
                        caption_sha256=hashlib.sha256(batch["ai_caption"][0].encode()).hexdigest(),
                        context=dataclasses.asdict(context),
                        inference_only_current_image=True,
                    ),
                )
                with (
                    bridge.request(context, name=entry["name"], seed=entry["sampling_seed"], purpose="training"),
                    counter.scope(entry["name"], 35),
                ):
                    response = service.execute(sample, batch).model_dump(mode="json")
                response_arrays(response)
                record["response_ref"] = publish(current / "response.json", response)
                record["initial_ref"] = initial.finish()
                if d0 is not None:
                    same_initial(d0["requests"][entry["index"]]["initial_ref"], record["initial_ref"])
                record["trace_ref"] = publish(current / "actual_sigma_trace.json", bridge.last)
                record["compact_ledger_ref"] = active_sink.finish()
                record["complete"] = True
                result["actual_requests"] += 1
                bridge.last_points = ()
                active_sink = None
                print(
                    json.dumps(dict(round=round_name, requests=result["actual_requests"], denoise=len(counter.calls))),
                    flush=True,
                )
        require(
            len(counter.calls) == 10500
            and all(c["completed"] for c in counter.calls)
            and bridge.total_head_calls == 10500,
            "full300/10500 budget is incomplete",
        )
        result["parameters_after"] = parameters_identity(model)
        require(result["parameters_before"] == result["parameters_after"], "base parameters changed")
        check_sources(sources)
        admit(prepared_ref, registration_ref, required_sources=required)
        if params_ref is not None:
            checked(params_ref)
        result["cosmos_imported_sources_final"] = imported_cosmos_sources(qualified)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result["boundary_calls"] = [] if counter is None else counter.calls
        result["actual_denoise_calls"] = len(result["boundary_calls"])
        result["actual_head_calls"] = 0 if bridge is None else bridge.total_head_calls
        cleanup = []
        for label, remove in (
            ("initial", None if initial is None else initial.remove),
            ("counter", None if counter is None else lambda: setattr(model, "denoise", counter.original)),
            ("bridge", None if bridge is None else bridge.restore),
        ):
            if remove is not None:
                try:
                    remove()
                except BaseException as error:
                    cleanup.append(dict(stage=label, type=type(error).__name__, message=str(error)))
        if cleanup:
            result.update(complete=False, passed=False, cleanup_errors=cleanup)
        publish(directory / "result.json", result)
    require(result["passed"], "collection cleanup failed")
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--gpu-uuid", choices=(GPU,), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    result = run(
        dict(path=str(args.prepared.resolve()), sha256=args.prepared_sha256),
        dict(path=str(args.execution_registration.resolve()), sha256=args.execution_registration_sha256),
        gpu_uuid=args.gpu_uuid,
        dry=args.dry,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
