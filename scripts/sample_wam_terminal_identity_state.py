"""056 raw/terminal058 sampling on synchronized current endpoints.

The sampler execution body is derived from038 with explicit new admission,
parameter identity and variant names. Scheduler, mask, capture and zero checks
retain their established implementations. No validation future archive is read.
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from collect_wam_clean_training import normalization_spec, qualified_grid, same_initial
from qualify_wam_clean_bridge import (
    CONVERSION_SOURCE,
    SAMPLER_SOURCE,
    imported_cosmos_sources,
    response_arrays,
    runtime_environment,
)
from qualify_wam_clean_bridge import (
    ConversionWitness as OriginalConversionWitness,
)
from sample_wam_clean_training_state import historical_source_summaries, require_zero_response, require_zero_sources

from s2a.artifacts import REPO
from s2a.clean_training_runtime import VISION_SHAPE, checked, publish, read, reference, require
from s2a.sampler_trace import check_sources, parameters_identity
from s2a.terminal_capture_identity import TerminalIdentityBridgeMixin
from s2a.terminal_identity_state_execution import GPU, admit, admit_sampling, select_readout
from s2a.wam_clean_capture import NoiseRoster
from s2a.wam_clean_compact import CompactCaptureSink, CompactRequest
from s2a.wam_clean_readout import CleanResidualReadout


class ConversionWitness(TerminalIdentityBridgeMixin, OriginalConversionWitness):
    pass


SCHEMA = "s2a.terminal-identity-state-sampling.v1"


def run(prepared_ref, registration_ref, *, candidate_proof_ref, gpu_uuid):
    required = [
        Path(__file__),
        REPO / "src/s2a/clean_training_runtime.py",
        REPO / "src/s2a/terminal_identity_state_execution.py",
        REPO / "src/s2a/wam_clean_compact.py",
        REPO / "scripts/collect_wam_clean_training.py",
        REPO / "scripts/run_wam_clean_training.py",
    ]
    prepared, _, sources = admit(prepared_ref, registration_ref, required_sources=required)
    head, params_ref, qualification_ref, qualification, entries = admit_sampling(
        prepared, prepared_ref, registration_ref, candidate_proof_ref
    )
    qualified = read(prepared["prerequisite_refs"]["bridge_prepared"])
    runtime_environment(qualified, gpu_uuid)
    require(gpu_uuid == GPU, "056 aggregate sampler must use its single registered allowed UUID")
    grid = qualified_grid(prepared)
    directory = Path(prepared["root"]) / "state_sampling"
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version=SCHEMA,
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        candidate_proof_ref=candidate_proof_ref,
        readout_ref=params_ref,
        generations=[],
        model_initialization_attempts=0,
        model_initializations=0,
        actual_requests=0,
        actual_denoise_calls=0,
        actual_head_calls=0,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
        source_sha256=sources,
        gpu_uuid=gpu_uuid,
        historical_model_capture_archives_decoded=0,
    )
    bridge = counter = initial = model = None
    try:
        gold = {}
        for entry in entries[:3]:
            original = qualification["generations"][entry["qualified_generation_index"]]
            gold[entry["name"]] = historical_source_summaries(original)
            result["historical_model_capture_archives_decoded"] += 35
        import torch
        from audit_wam_current_numerics import require_eval
        from run_wam_readout_features import BoundaryCounter, actual_context
        from run_wam_readout_sampling import InitialCapture

        from s2a.wam_runtime import install_active_slot_mask, load_service

        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "state sampler requires exactly one allowed GPU",
        )
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        require(actual_uuid.removeprefix("GPU-") == GPU.removeprefix("GPU-"), "actual state GPU UUID differs")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        service_dir = directory / "service"
        service_dir.mkdir()
        checkpoint = prepared["model_config"]["checkpoint"]
        result["model_initialization_attempts"] = 1
        service = load_service(dict(checkpoints=[checkpoint]), "8400_masked", service_dir)
        model = service.model
        result["model_initializations"] = 1
        require(
            Path(service.info["checkpoint_path"]).resolve()
            == (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
            and service.info["ema_weights"] is False,
            "state checkpoint/EMA differs",
        )
        require_eval(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        result["parameters_before"] = parameters_identity(model)
        from s2a.wam_readout import ACTIVE_MASK

        result["mask_evidence"] = install_active_slot_mask(model, ACTIVE_MASK)
        result["cosmos_imported_sources_after_load"] = imported_cosmos_sources(qualified)
        result["normalization_ref"] = publish(directory / "normalization.json", normalization_spec(service.spec))
        zero = CleanResidualReadout(np.zeros((1291, 10)), np.zeros(1290), 1.0, 0.1)
        result["zero_head_ref"] = publish(directory / "zero_head.json", zero.to_parameters())
        cosmos = Path(qualified["config"]["cosmos_source_root"])
        sampler, conversion = (str((cosmos / name).resolve()) for name in (SAMPLER_SOURCE, CONVERSION_SOURCE))
        active_sink, active_gold = None, None

        def sink(point):
            require(active_sink is not None, "state point is outside registered generation")
            point_ref = active_sink(point)
            if active_gold is not None:
                require_zero_sources(read(point_ref), active_gold[point.metadata["step_index"]])

        bridge = ConversionWitness(
            model,
            noise_roster=NoiseRoster(**prepared["noise_roster"]),
            readout=None,
            feature_terms=(),
            sink=sink,
            denoise_delegate=model.denoise,
            forward_file=sampler,
            forward_sha256=sources[sampler],
            sources=sources,
            conversion_source_ref=dict(path=conversion, sha256=sources[conversion]),
        )
        counter, initial = BoundaryCounter(model, 3465), InitialCapture(model)
        from cosmos_framework.inference.args import OmniSampleOverrides

        raw_by_pair = {}
        with torch.inference_mode():
            for entry in entries:
                current = directory / f"{entry['index']:03d}_{entry['name']}"
                current.mkdir()
                record = {key: value for key, value in entry.items() if key not in ("request", "image_ref")}
                record["complete"] = False
                result["generations"].append(record)
                request = entry["request"]
                record["request_ref"] = publish(current / "request.json", request)
                bridge.readout = select_readout(entry["variant"], zero=zero, candidate=head)
                active_gold = gold.get(entry["name"])
                active_sink = CompactCaptureSink(
                    current / "compact",
                    CompactRequest(
                        entry["name"],
                        entry["sampling_seed"],
                        "evaluation",
                        "state",
                        entry["zero_case_name"] if entry["variant"] == "zero" else entry["anchor_name"],
                        params_ref["sha256"] if entry["variant"] == "terminal058" else None,
                        entry["image_ref"]["sha256"],
                    ),
                    expected_vision_shape=VISION_SHAPE,
                    expected_grid=grid,
                )
                initial.arm(
                    dict(
                        generation_index=entry["index"],
                        name=entry["name"],
                        pair_id=entry["pair_id"],
                        variant=entry["variant"],
                        sampling_seed=entry["sampling_seed"],
                    ),
                    current,
                )
                checked(entry["image_ref"])
                sample, batch = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(request)), current)
                checked(entry["image_ref"])
                context = actual_context(service, sample, batch)
                caption = json.loads(batch["ai_caption"][0])
                require(
                    context.is_libero
                    and caption["task_title"] == caption["command"] == request["prompt"]
                    and caption["task_plan"] == [],
                    "state sampler received noncurrent or privileged conditioning",
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
                    bridge.request(context, name=entry["name"], seed=entry["sampling_seed"], purpose="evaluation"),
                    counter.scope(entry["name"], 35),
                ):
                    response = service.execute(sample, batch).model_dump(mode="json")
                response_arrays(response)
                record["response_ref"] = publish(current / "response.json", response)
                record["initial_ref"] = initial.finish()
                record["trace_ref"] = publish(current / "actual_sigma_trace.json", bridge.last)
                record["compact_ledger_ref"] = active_sink.finish()
                if entry["variant"] == "zero":
                    original = qualification["generations"][entry["qualified_generation_index"]]
                    require_zero_response(response, entry, original)
                    same_initial(original["initial_ref"], record["initial_ref"])
                    record.update(
                        qualified_generation_ref=qualification_ref,
                        historical_output_exact=True,
                        qualified_output_exact=True,
                        all35_source_hashes_exact=True,
                        zero_source_hashes_ref=publish(
                            current / "historical_source_hashes.json",
                            dict(
                                qualified_generation_ref=qualification_ref,
                                qualified_generation_index=entry["qualified_generation_index"],
                                point_refs=original["points"],
                                source_arrays=active_gold,
                            ),
                        ),
                    )
                elif entry["variant"] == "raw":
                    raw_by_pair[entry["pair_id"]] = record
                else:
                    raw = raw_by_pair[entry["pair_id"]]
                    require(raw["index"] == entry["paired_raw_generation_index"], "raw/W2 pair row order differs")
                    same_initial(raw["initial_ref"], record["initial_ref"])
                    record["paired_initial_exact"] = True
                record["complete"] = True
                result["actual_requests"] += 1
                bridge.last_points = ()
                active_sink = active_gold = None
                print(
                    json.dumps(dict(generation=entry["index"], variant=entry["variant"], denoise=len(counter.calls))),
                    flush=True,
                )
        require(
            len(counter.calls) == 3465
            and all(c["completed"] for c in counter.calls)
            and bridge.total_head_calls == 3465
            and result["actual_requests"] == 99,
            "state99/3465 budget is incomplete",
        )
        result["parameters_after"] = parameters_identity(model)
        require(result["parameters_before"] == result["parameters_after"], "state base-model parameters changed")
        check_sources(sources)
        checked(params_ref)
        admit(prepared_ref, registration_ref, required_sources=required)
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
    require(result["passed"], "state sampling cleanup failed")
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration", "candidate-proof"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--gpu-uuid", choices=(GPU,), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    def ref(name):
        return dict(path=str(getattr(args, name).resolve()), sha256=getattr(args, name + "_sha256"))

    if args.dry:
        prepared, _, _ = admit(ref("prepared"), ref("execution_registration"), required_sources=[Path(__file__)])
        admit_sampling(
            prepared,
            ref("prepared"),
            ref("execution_registration"),
            ref("candidate_proof"),
        )
        print(json.dumps({"passed": True, "dry": True, "model_initializations": 0, "requests": 0}), flush=True)
        return
    print(
        json.dumps(
            run(
                ref("prepared"),
                ref("execution_registration"),
                candidate_proof_ref=ref("candidate_proof"),
                gpu_uuid=args.gpu_uuid,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
