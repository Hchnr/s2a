"""038 unique W2: three exact zero regressions, then96 paired state requests.

This model worker reads current PNGs, saved model captures and metadata only.
No validation future labels, simulator, fit or fresh performance are consumed.
A final execution registration and normally exited W2 fit owner are mandatory.
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
    ConversionWitness,
    imported_cosmos_sources,
    outputs_exact,
    response_arrays,
    runtime_environment,
)
from run_wam_clean_training import prior

from s2a.artifacts import REPO
from s2a.clean_training_runtime import (
    GPU,
    VISION_SHAPE,
    admit_execution,
    checked,
    publish,
    read,
    reference,
    require,
    trained_head,
    validate_current_request,
)
from s2a.sampler_trace import check_sources, parameters_identity
from s2a.wam_clean_capture import NoiseRoster
from s2a.wam_clean_compact import CompactCaptureSink, CompactRequest
from s2a.wam_clean_readout import CleanResidualReadout

SCHEMA = "s2a.clean-training-state-sampling.v1"
VAL_EPISODES = (1254, 1126, 868, 1139, 1209, 1176)


def sampling_entries(prepared, qualification):
    """Build only the preregistered99 inputs, without opening any future archive."""
    cases = prepared["zero_identity_cases"]
    require(len(cases) == 3 and len({c["name"] for c in cases}) == 3, "requires three original zero cases")
    raw = [(i, g) for i, g in enumerate(qualification["generations"]) if g["variant"] == "raw"]
    require(len(raw) == 3 and [i for i, _ in raw] == [0, 1, 2], "037 must supply its three original raw generations")
    result = []
    for case, (qualified_index, original) in zip(cases, raw, strict=True):
        request = case["request"]
        require(
            original["case_id"] == case["name"]
            and original["sampling_seed"] == request["seed"] == 42
            and original["complete"] is original["historical_output_exact"] is True,
            "zero case differs from its original037 raw input",
        )
        validate_current_request(
            request,
            name=request["name"],
            seed=42,
            image_ref=case["image_ref"],
            task=request["prompt"],
            template=case["request"],
        )
        result.append(
            dict(
                index=len(result),
                name="zero_" + case["name"],
                variant="zero",
                validation_index=None,
                anchor_name=None,
                zero_case_name=case["name"],
                pair_id=case["name"],
                sampling_seed=42,
                image_ref=case["image_ref"],
                request=copy.deepcopy(request),
                qualified_generation_index=qualified_index,
                expected_response_ref=case["expected_response_ref"],
                qualified_response_ref=original["response_ref"],
            )
        )
    anchors = prepared["validation_anchors"]
    require(
        len(anchors) == 24
        and len({a["name"] for a in anchors}) == 24
        and {a["episode"] for a in anchors} == set(VAL_EPISODES),
        "all six validation EPs/four anchors are required",
    )
    by_name = {a["name"]: a for a in anchors}
    entries = prepared["validation_generations"]
    require(
        len(entries) == 96
        and [e["index"] for e in entries] == list(range(96))
        and [e["variant"] for e in entries] == ["raw", "w2"] * 48,
        "complete96 paired generation roster differs",
    )
    for index, entry in enumerate(entries):
        anchor = by_name[entry["anchor_name"]]
        seed = entry["sampling_seed"]
        require(
            seed in (42, 43) and anchor["future_labels_parsed"] is False,
            "validation purpose or unopened-label contract differs",
        )
        expected_anchor, expected_seed = anchors[index // 4], (42, 43)[(index % 4) // 2]
        require(anchor == expected_anchor and seed == expected_seed, "validation anchor/seed row order differs")
        validate_current_request(
            entry["request"],
            name=entry["name"],
            seed=seed,
            image_ref=anchor["image_ref"],
            task=anchor["task"],
            template=anchor["request_template"],
        )
        result.append(
            dict(
                index=3 + index,
                validation_index=index,
                name=entry["name"],
                variant=entry["variant"],
                anchor_name=entry["anchor_name"],
                zero_case_name=None,
                pair_id=f"{entry['anchor_name']}_s{seed}",
                sampling_seed=seed,
                image_ref=anchor["image_ref"],
                request=copy.deepcopy(entry["request"]),
                paired_raw_generation_index=(3 + index - 1 if entry["variant"] == "w2" else None),
            )
        )
    require(len(result) == len({e["name"] for e in result}) == 99, "state generation identity collision")
    return result


def admit_sampling(prepared, prepared_ref, registration_ref, w2_fit_ref, w2_verification_ref):
    owner = prior(prepared, prepared_ref, registration_ref, "W2")
    require(
        owner["producer_ref"] == w2_fit_ref and owner["verification_ref"] == w2_verification_ref,
        "W2 must be the exact normally exited/verified fixed owner result",
    )
    head, params = trained_head(
        w2_fit_ref, w2_verification_ref, stage="W2", prepared_ref=prepared_ref, registration_ref=registration_ref
    )
    fit, verified = read(w2_fit_ref), read(w2_verification_ref)
    require(
        fit["rounds"] == ["D0", "D1"]
        and fit["actual_rows"] == verified["actual_rows"] == 840000
        and len(fit["collection_refs"]) == len(fit["capture_audit_refs"]) == 2
        and verified["independent_reconstructions"] == 1
        and verified["full_training_row_reconstruction"] is True,
        "state sampling requires complete840000-row W2 and independent QR",
    )
    primary, independent = fit["fit_metadata"]["replay_sha256"], verified["qr_metadata"]["replay_sha256"]
    require(
        primary["features"] == independent["features"]
        and primary["weights"] == independent["weights"]
        and primary["target_clean_residual"] == independent["targets"],
        "W2 all-row replay identities differ",
    )
    qualification_ref = prepared["prerequisite_refs"]["bridge_qualification"]
    qualification = read(qualification_ref)
    require(
        qualification["complete"] is qualification["passed"] is qualification["bridge_qualified"] is True,
        "original037 actual model bridge was not qualified",
    )
    return head, params, qualification_ref, qualification, sampling_entries(prepared, qualification)


def historical_source_summaries(generation):
    """Hash actual037 model snapshots only; never read a validation GT archive."""
    require(
        generation["variant"] == "raw" and len(generation["points"]) == 35,
        "zero gold requires all35 original raw points",
    )
    result = []
    for step, point_ref in enumerate(generation["points"]):
        point = read(point_ref)
        require(
            point["step_index"] == step and point["request_name"] == generation["name"],
            "037 source point identity differs",
        )
        path = checked(point["arrays"])
        require(
            path.parent == checked(point_ref).parent and path.stem == checked(point_ref).stem,
            "037 model snapshot escaped its declared point path",
        )
        with np.load(path, allow_pickle=False) as archive:
            require(
                set(archive.files) == set(point["array_index"]) == set(dict(point["original_dtypes"]))
                and len(archive.files) == 18,
                "037 full source array roster differs",
            )
            summaries = {}
            for name in archive.files:
                value = archive[name]
                require(
                    np.isfinite(value).all()
                    and point["array_index"][name] == dict(shape=list(value.shape), dtype=str(value.dtype)),
                    "037 source array layout/value differs",
                )
                summaries[name] = dict(
                    shape=list(value.shape),
                    storage_dtype=value.dtype.str,
                    snapshot_bytes=value.nbytes,
                    snapshot_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                    original_dtype=dict(point["original_dtypes"])[name],
                )
        result.append(summaries)
    return result


def require_zero_response(response, entry, qualified):
    require(
        outputs_exact(response, read(entry["expected_response_ref"]))
        and outputs_exact(response, read(qualified["response_ref"])),
        "zero head changed historical full40x128 outputs",
    )


def require_zero_sources(point_record, expected):
    require(point_record["source_arrays"] == expected, "zero head changed an actual037 source array byte/dtype/shape")


def run(prepared_ref, registration_ref, *, w2_fit_ref, w2_verification_ref, gpu_uuid):
    required = [
        Path(__file__),
        REPO / "src/s2a/clean_training_runtime.py",
        REPO / "src/s2a/wam_clean_compact.py",
        REPO / "scripts/collect_wam_clean_training.py",
        REPO / "scripts/run_wam_clean_training.py",
    ]
    prepared, _, sources = admit_execution(prepared_ref, registration_ref, required_sources=required)
    head, params_ref, qualification_ref, qualification, entries = admit_sampling(
        prepared, prepared_ref, registration_ref, w2_fit_ref, w2_verification_ref
    )
    qualified = read(prepared["prerequisite_refs"]["bridge_prepared"])
    runtime_environment(qualified, gpu_uuid)
    require(gpu_uuid == GPU, "state sampler must use the single038 registered UUID")
    grid = qualified_grid(prepared)
    directory = Path(prepared["root"]) / "state_sampling"
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version=SCHEMA,
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        w2_fit_ref=w2_fit_ref,
        w2_verification_ref=w2_verification_ref,
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
                bridge.readout = dict(raw=None, zero=zero, w2=head)[entry["variant"]]
                active_gold = gold.get(entry["name"])
                active_sink = CompactCaptureSink(
                    current / "compact",
                    CompactRequest(
                        entry["name"],
                        entry["sampling_seed"],
                        "evaluation",
                        "state",
                        entry["zero_case_name"] if entry["variant"] == "zero" else entry["anchor_name"],
                        params_ref["sha256"] if entry["variant"] == "w2" else None,
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
        admit_execution(prepared_ref, registration_ref, required_sources=required)
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
    for name in ("prepared", "execution-registration", "w2-fit", "w2-verification"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--gpu-uuid", choices=(GPU,), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()

    def ref(name):
        return dict(path=str(getattr(args, name).resolve()), sha256=getattr(args, name + "_sha256"))

    print(
        json.dumps(
            run(
                ref("prepared"),
                ref("execution_registration"),
                w2_fit_ref=ref("w2_fit"),
                w2_verification_ref=ref("w2_verification"),
                gpu_uuid=args.gpu_uuid,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
