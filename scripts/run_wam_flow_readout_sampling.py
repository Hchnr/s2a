"""026 conditional3 zero reproductions and96 paired current-image samples."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from run_wam_flow_readout import check_hashes, read, read_prepared, reference, require, sampling_admission
from run_wam_readout_features import BoundaryCounter, generate, parameter_versions, setup_service
from run_wam_readout_sampling import InitialCapture

from s2a.artifacts import sha256
from s2a.wam_flow_readout import FlowReadoutHook, FlowResidualReadout, SolverSigmaBridge
from s2a.worker_handshake import write_once


class RequestBridge:
    """Adapt the frozen generation helper's request boundary to the flow bridge."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.entry = None

    def request(self, context):
        require(self.entry is not None, "026 generation bridge is unarmed")
        return self.bridge.request(context, name=self.entry["name"], seed=self.entry["sampling_seed"])


def admit_fit(prepared_path, fit_path, fit_sha, audit_path, audit_sha):
    fit, audit = read(fit_path, fit_sha), read(audit_path, audit_sha)
    require(
        audit.get("phase") == "fit" and audit.get("passed") is True, "026 actual independent fit verification missing"
    )
    check_hashes(audit["input_sha256"])
    sampling_admission(fit, audit, prepared_path, fit_path)
    ref = fit["models"]["adapt"]
    require(sha256(ref["path"]) == ref["sha256"], "026 adapted model SHA differs")
    with np.load(ref["path"], allow_pickle=False) as archive:
        require(
            archive.files == ["weights"] and archive["weights"].shape == (1281, 10),
            "026 adapted model geometry differs",
        )
        adapted = FlowResidualReadout(archive["weights"])
    return fit, adapted


def generate_one(service, entry, directory, counter, request_bridge, initial, label):
    request_bridge.entry = entry
    initial.arm(entry, directory)
    response = generate(service, entry["request"], directory, counter, request_bridge, label)
    initial_ref = initial.finish()
    trace = request_bridge.bridge.last
    require(trace["complete"] is True and len(trace["steps"]) == 35, "026 actual solver trace incomplete")
    write_once(directory / "actual_sigma_trace.json", trace)
    request_bridge.entry = None
    return response, dict(
        response=reference(directory / "response.json"),
        conditioning=reference(directory / "conditioning.json"),
        initial=initial_ref,
        actual_sigma_trace=reference(directory / "actual_sigma_trace.json"),
    )


def check_zero(response, cached):
    actual, expected = (data["outputs"][0]["content"] for data in (response, cached))
    for a, b in (
        (actual["action"], expected["action"]),
        (actual["action128"]["normalized_action"], expected["action128"]["normalized_action"]),
    ):
        require(
            np.asarray(a).shape == np.asarray(b).shape == (40, 128) and np.array_equal(a, b),
            "026 zero flow bridge did not reproduce complete raw/normalized output",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--fit-sha256", required=True)
    parser.add_argument("--fit-verification", type=Path, required=True)
    parser.add_argument("--fit-verification-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "6" and os.environ.get("LD_LIBRARY_PATH") == "",
        "026 model worker requires only physicalGPU6 and emptyLD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    fit, adapted = admit_fit(
        args.prepared, args.fit, args.fit_sha256, args.fit_verification, args.fit_verification_sha256
    )
    old = read(prepared["source_prepared_ref"]["path"], prepared["source_prepared_ref"]["sha256"])
    entries = prepared["validation_generations"]
    require(
        len(entries) == 96 and [e["variant"] for e in entries] == ["raw", "adapt"] * 48, "026 paired96 roster differs"
    )
    # Current PNGs are authorized only after both teacher gates, and are checked
    # before model loading. No source episode state/pixel archive is opened.
    for anchor in prepared["anchors"]:
        if anchor["split"] == "validation":
            require(sha256(anchor["image_path"]) == anchor["image_sha256"], "026 current validation PNG changed")
    for case in prepared["zero_identity_cases"]:
        require(sha256(case["sample"]["image_path"]) == case["sample"]["image_sha256"], "026 zero current PNG changed")
    output = Path(prepared["output_root"]) / "sampling"
    output.mkdir()
    write_once(output / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), cuda="6", ld=""))
    result = dict(
        schema_version="s2a.flow-readout-sampling.v1",
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        fit_ref=reference(args.fit),
        fit_verification_ref=reference(args.fit_verification),
        adapted_model_ref=fit["models"]["adapt"],
        zero_identity=[],
        generations=[],
        physics_steps=0,
        fresh_cases=0,
        new_teacher_calls=0,
    )
    counter = hook = bridge = initial = head_observer = None
    try:
        import torch

        service_dir = output / "service"
        service_dir.mkdir()
        service = setup_service(old, service_dir)
        model = service.model
        before_parameters = parameter_versions(model)
        zero = FlowResidualReadout.zero()
        hook = FlowReadoutHook(model.net.llm2action, zero)
        sampler_path = str(
            Path(prepared["config"]["cosmos_repo"]) / "cosmos_framework/model/generator/diffusion/samplers/unipc.py"
        )
        generator_path = str(
            Path(prepared["config"]["cosmos_repo"]) / "cosmos_framework/model/generator/omni_mot_model.py"
        )
        sources = {path: prepared["source_sha256"][path] for path in (sampler_path, generator_path)}
        bridge = SolverSigmaBridge(
            model, hook, forward_file=sampler_path, forward_sha256=sources[sampler_path], sources=sources
        )
        counter = BoundaryCounter(model, 3465)
        initial = InitialCapture(model)
        request_bridge = RequestBridge(bridge)
        counts = dict(zero=0, raw=0, adapt=0)
        variant = None

        def observe_head(module, values, kwargs, base):
            require(
                variant in counts and counter.current is not None and not kwargs,
                "026 head outside registered generation",
            )
            require(
                tuple(values[0].shape) == (40, 1280) and tuple(base.shape) == (40, 128),
                "026 actual head geometry differs",
            )
            counts[variant] += 1

        head_observer = model.net.llm2action.register_forward_hook(observe_head, with_kwargs=True)
        with torch.inference_mode():
            variant = "zero"
            for case in prepared["zero_identity_cases"]:
                entry = dict(
                    name=case["name"],
                    generation_index=-case["task_index"] - 1,
                    pair_id=f"zero{case['task_index']}",
                    variant="zero",
                    sampling_seed=42,
                    request=case["request"],
                )
                directory = output / f"zero_identity_task{case['task_index']}"
                response, refs = generate_one(
                    service, entry, directory, counter, request_bridge, initial, f"zero:{case['task_index']}"
                )
                cached = read(case["cache"]["response_path"], case["cache"]["response_sha256"])
                check_zero(response, cached)
                result["zero_identity"].append(dict(case, **refs))
                print(json.dumps(dict(zero_completed=len(result["zero_identity"]), name=case["name"])), flush=True)
            for entry in entries:
                variant = entry["variant"]
                hook.readout = adapted if variant == "adapt" else zero
                _, refs = generate_one(
                    service,
                    entry,
                    output / entry["name"],
                    counter,
                    request_bridge,
                    initial,
                    f"sampling:{entry['generation_index']}",
                )
                result["generations"].append(dict(entry, **refs))
                print(
                    json.dumps(
                        dict(
                            generations_completed=len(result["generations"]),
                            name=entry["name"],
                            denoise_calls=len(counter.calls),
                        )
                    ),
                    flush=True,
                )
        require(
            len(counter.calls) == 3465 and counts == dict(zero=105, raw=1680, adapt=1680),
            "026 complete99/3465 boundary budget differs",
        )
        require(parameter_versions(model) == before_parameters, "026 original model parameters changed")
        read_prepared(args.prepared, args.prepared_sha256)
        check_hashes(read(args.fit_verification, args.fit_verification_sha256)["input_sha256"])
        result.update(
            complete=True,
            passed=True,
            all_original_parameters_frozen=True,
            head_call_counts=counts,
            runtime=service.info,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if head_observer is not None:
            head_observer.remove()
        if initial is not None:
            initial.remove()
        if bridge is not None:
            bridge.restore()
        if hook is not None:
            hook.remove()
        result["denoise_calls"] = 0 if counter is None else len(counter.calls)
        result["denoise_ledger"] = [] if counter is None else counter.calls
        write_once(output / "result.json", result)


if __name__ == "__main__":
    main()
