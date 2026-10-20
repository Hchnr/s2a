"""Private011 service: unchanged WAM/mask with observed UniPC input and schedule."""

import argparse
import json
import os
import types
from pathlib import Path

import numpy as np

from s2a.sampler_trace import SamplerTrace, check_sources, file_sha, json_sha, write_new_json
from s2a.wam_runtime import install_active_slot_mask, load_service


def full128(raw):
    if raw.get("status") != "success":
        raise ValueError("unsuccessful model response")
    content = raw["outputs"][0]["content"]
    arrays = tuple(np.asarray(v) for v in (content["action"], content["action128"]["normalized_action"]))
    if any(x.shape != (40, 128) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("invalid full40x128 response")
    return arrays


def require_exact35(raw, cached):
    actual, expected = full128(raw), full128(cached)
    if not all(np.array_equal(x, y) for x, y in zip(actual, expected, strict=True)):
        raise ValueError("traced35 full128 raw/normalized output is not exactly cached35")


def check_weight_stats(weights):
    for path, item in weights.items():
        stat = Path(path).stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"weight stat differs from existing byte fingerprint: {path}")


def save_acceptance_input(prepared, batch, service_root):
    # create_app passes the same service_root for every prepare. Stock prepare
    # stores its separate generate_UUID directory in the prepared sample args.
    directory = Path(prepared.sample_args["output_dir"]).resolve()
    service_root = Path(service_root).resolve()
    if directory == service_root or not directory.is_relative_to(service_root):
        raise ValueError("prepared output must be a separate request directory within this service")
    write_new_json(
        directory / "acceptance_input.json",
        dict(caption=batch["ai_caption"], exact_published_caption=False, active_slot_mask=True, seed=prepared.seed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("011 service requires physical GPU6 and empty LD_LIBRARY_PATH")
    if file_sha(args.prepared) != args.prepared_sha256:
        raise ValueError("prepared manifest SHA differs")
    protocol = json.loads(args.prepared.read_text())
    check_sources(protocol["source_sha256"])
    check_weight_stats(protocol["weight_files"])
    root = Path(protocol["root"])
    output = root / "service"
    output.mkdir(exist_ok=False)
    c = json.loads(Path(protocol["configs"]["35"]["path"]).read_text())
    service = load_service(c, "8400_masked", output)
    evidence = install_active_slot_mask(service.model, [True] * 10 + [False] * 118)
    sampler_source = protocol["unipc_source"]
    trace = SamplerTrace(
        service.model,
        forward_file=sampler_source,
        forward_sha256=protocol["source_sha256"][sampler_source],
        sources=protocol["source_sha256"],
    )
    if service.model.training or service.model.net.training:
        raise ValueError("trace requires an eval model")
    runtime = dict(
        active_slot_mask=True,
        exact_published_caption=False,
        sampler_trace=True,
        no_velocity_postprocess=True,
        no_network_forward_hooks_added=True,
        source_sha256=protocol["source_sha256"],
        prepared_sha256=args.prepared_sha256,
        physical_gpu=6,
        new_physics_steps=0,
    )
    service.info["acceptance_intervention"] = runtime
    service.info["port"] = 9566
    service.info["sampler_trace_protocol"] = runtime
    write_new_json(output / "intervention.json", runtime)
    records = protocol["cached_cases"]
    names = protocol["request_order"]
    counters = {35: 0, 70: 0, 140: 0}
    pending = {}
    poisoned = False
    original_prepare, original_execute = service.prepare, service.execute
    for stage in (35, 70, 140):
        (output / f"traces{stage}").mkdir()

    def prepare(self, request, output_dir):
        nonlocal poisoned
        if poisoned or pending:
            raise ValueError("011 service is poisoned or has a pending request")
        try:
            stage = request.num_steps
            if stage not in counters or counters[stage] >= 48 or request.name != names[counters[stage]]:
                raise ValueError("unexpected stage/name/order or request budget exhausted")
            if stage != 35:
                prior = 35 if stage == 70 else 70
                gate = root / "orchestration" / "run" / f"admission{prior}.json"
                if counters[prior] != 48 or not json.loads(gate.read_text())["passed"]:
                    raise ValueError("preceding full48 admission is missing")
            record = records[request.name]
            expected = json.loads(Path(record["request_path"]).read_text())
            expected["num_steps"] = stage
            incoming = request.model_dump(mode="json", exclude_unset=True)
            if incoming != expected:
                raise ValueError("request differs from the registered cached input")
            if file_sha(record["request_path"]) != record["request_sha256"]:
                raise ValueError("cached request changed")
            prepared, batch = original_prepare(request, output_dir)
            if (
                prepared.action_dim_mask != [True] * 10 + [False] * 118
                or prepared.seed != expected["seed"]
                or prepared.guidance != 1.0
                or prepared.shift != 5.0
            ):
                raise ValueError("prepared inference identity differs")
            # The pre-existing prepare renders task-only captions exactly as before.
            conditioning = dict(
                original_request_without_steps={k: v for k, v in expected.items() if k != "num_steps"},
                caption=batch["ai_caption"],
                image_sha256=file_sha(expected["vision_path"]),
            )
            if conditioning["image_sha256"] != record["image_sha256"]:
                raise ValueError("source PNG changed")
            pending.update(name=request.name, stage=stage, prepared_id=id(prepared), conditioning=conditioning)
            save_acceptance_input(prepared, batch, output_dir)
            return prepared, batch
        except BaseException:
            poisoned = True
            raise

    def execute(self, request, batch):
        nonlocal poisoned
        name, stage = pending["name"], pending["stage"]
        target = output / f"traces{stage}" / f"{name}.json"
        observed = None
        try:
            if pending["prepared_id"] != id(request):
                raise ValueError("prepared request identity differs")
            check_weight_stats(protocol["weight_files"])
            if file_sha(args.prepared) != args.prepared_sha256:
                raise ValueError("prepared protocol changed during serving")

            def admit(identity):
                if stage == 35:
                    return
                baseline = output / "traces35" / f"{name}.json"
                gate = json.loads((root / "orchestration/run/admission35.json").read_text())
                if file_sha(baseline) != gate["trace_sha256"][name]:
                    raise ValueError("admitted35 trace changed")
                old = json.loads(baseline.read_text())
                if not old["complete"] or old["pair_identity"] != identity:
                    raise ValueError("actual initial sampler state/conditioning is not paired with traced35")

            trace.begin(name=name, num_steps=stage, conditioning=pending["conditioning"], admit=admit)
            result = original_execute(request, batch)
            observed = trace.finish()
            raw = result.model_dump(mode="json")
            actual = full128(raw)
            if stage == 35:
                record = records[name]
                if file_sha(record["response_path"]) != record["response_sha256"]:
                    raise ValueError("cached35 response changed")
                require_exact35(raw, json.loads(Path(record["response_path"]).read_text()))
            identity = raw["outputs"][0]["content"]["action128"]
            if identity["seed"] != request.seed or identity["checkpoint_path"] != protocol["checkpoint"]:
                raise ValueError("response checkpoint/seed differs")
            check_weight_stats(protocol["weight_files"])
            observed.update(
                cached35_full128_exact=True if stage == 35 else None,
                actual_state_and_condition_pair_with35=True if stage != 35 else None,
                response_arrays_sha256=[json_sha(x.tolist()) for x in actual],
                prepared_manifest_sha256=args.prepared_sha256,
            )
            if file_sha(args.prepared) != args.prepared_sha256:
                raise ValueError("prepared protocol changed during request")
            write_new_json(target, observed)
            counters[stage] += 1
            pending.clear()
            return result
        except BaseException as exc:
            poisoned = True
            partial = trace.abort()
            write_new_json(
                target.with_suffix(".error.json"),
                dict(error=repr(exc), partial_trace=partial or observed, pending=pending),
            )
            raise

    service.prepare = types.MethodType(prepare, service)
    service.execute = types.MethodType(execute, service)
    import uvicorn
    from cosmos_framework.inference.action_serving import create_app

    try:
        uvicorn.run(create_app(service), host="127.0.0.1", port=9566, workers=1)
    finally:
        trace.restore()
        write_new_json(output / "mask_evidence.json", evidence)
        write_new_json(output / "closed.json", dict(counters=counters, poisoned=poisoned))


if __name__ == "__main__":
    main()
