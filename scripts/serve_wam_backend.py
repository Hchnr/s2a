"""013 private flash3_video service with observed inputs paired to011 SDPA35."""

import argparse
import json
import os
import types
from pathlib import Path

from serve_wam_sampler_trace import full128, save_acceptance_input

from s2a.artifacts import code_fingerprint
from s2a.sampler_trace import SamplerTrace, check_sources, file_sha, json_sha, write_new_json
from s2a.wam_backend_runtime import load_service as load_backend_service
from s2a.wam_runtime import install_active_slot_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("013 model requires physical GPU7 and empty LD_LIBRARY_PATH")
    if file_sha(args.prepared) != args.prepared_sha256:
        raise ValueError("prepared protocol differs")
    p = json.loads(args.prepared.read_text())
    check_sources(p["source_sha256"])
    check_sources(p["input_sha256"])
    from run_wam_sampler_steps import check_weights

    check_weights(p["weight_files"])
    root = Path(p["root"])
    output = root / "service"
    output.mkdir(exist_ok=False)
    c = json.loads(Path(p["config_path"]).read_text())
    service = load_backend_service(c, "8400_masked", output)
    if service.model.training or service.model.net.training:
        raise ValueError("requires eval model")
    evidence = install_active_slot_mask(service.model, [True] * 10 + [False] * 118)
    trace = SamplerTrace(
        service.model,
        forward_file=p["unipc_source"],
        forward_sha256=p["source_sha256"][p["unipc_source"]],
        sources=p["source_sha256"],
    )
    runtime = dict(
        active_slot_mask=True,
        exact_published_caption=False,
        warm_start_contract_run=None,
        sampler_trace=True,
        attention_backend="flash3_video",
        fixed_training_padding_restored=False,
        source_sha256=p["source_sha256"],
        code=code_fingerprint(),
        prepared_sha256=args.prepared_sha256,
        physical_gpu=7,
    )
    service.info["acceptance_intervention"] = runtime
    service.info["port"] = 9587
    write_new_json(output / "intervention.json", runtime)
    traces = output / "traces35"
    traces.mkdir()
    original_prepare, original_execute = service.prepare, service.execute
    state = dict(completed=0, pending=None, poisoned=False)

    def prepare(self, request, output_dir):
        try:
            n = state["completed"]
            if state["poisoned"] or state["pending"] is not None or n >= 48:
                raise ValueError("service is poisoned, occupied or its48 requests are complete")
            name = p["request_order"][n]
            record = p["cached_cases"][name]
            if file_sha(record["request_path"]) != record["request_sha256"]:
                raise ValueError("cached request changed")
            expected = json.loads(Path(record["request_path"]).read_text())
            if request.model_dump(mode="json", exclude_unset=True) != expected:
                raise ValueError("request differs from paired SDPA35 input/order")
            prepared, batch = original_prepare(request, output_dir)
            if (
                prepared.action_dim_mask != [True] * 10 + [False] * 118
                or prepared.seed != expected["seed"]
                or prepared.guidance != 1
                or prepared.shift != 5
            ):
                raise ValueError("prepared identity differs")
            conditioning = dict(
                original_request_without_steps={k: v for k, v in expected.items() if k != "num_steps"},
                caption=batch["ai_caption"],
                image_sha256=file_sha(expected["vision_path"]),
            )
            if conditioning["image_sha256"] != record["image_sha256"]:
                raise ValueError("paired image changed")
            save_acceptance_input(prepared, batch, output_dir)
            state["pending"] = dict(name=name, prepared_id=id(prepared), conditioning=conditioning)
            return prepared, batch
        except BaseException:
            state["poisoned"] = True
            raise

    def execute(self, request, batch):
        pending = state["pending"]
        if pending is None:
            raise ValueError("missing prepared request")
        name = pending["name"]
        target = traces / f"{name}.json"
        observed = None
        try:
            if pending["prepared_id"] != id(request) or file_sha(args.prepared) != args.prepared_sha256:
                raise ValueError("prepared identity or protocol changed")
            check_weights(p["weight_files"])

            def admit(identity):
                old = p["baseline_traces"][name]
                if file_sha(old["path"]) != old["sha256"]:
                    raise ValueError("admitted SDPA35 trace changed")
                baseline = json.loads(Path(old["path"]).read_text())
                if not baseline["complete"] or baseline["pair_identity"] != identity:
                    raise ValueError("actual initial state/conditioning differs from SDPA35")

            trace.begin(name=name, num_steps=35, conditioning=pending["conditioning"], admit=admit)
            result = original_execute(request, batch)
            observed = trace.finish()
            raw = result.model_dump(mode="json")
            arrays = full128(raw)
            identity = raw["outputs"][0]["content"]["action128"]
            if identity["seed"] != request.seed or identity["checkpoint_path"] != p["checkpoint"]:
                raise ValueError("response checkpoint/seed differs")
            check_weights(p["weight_files"])
            if file_sha(args.prepared) != args.prepared_sha256:
                raise ValueError("prepared protocol changed during request")
            observed.update(
                attention_backend="flash3_video",
                actual_state_and_condition_pair_with_sdpa35=True,
                baseline_sdpa35_trace=p["baseline_traces"][name],
                response_arrays_sha256=[json_sha(a.tolist()) for a in arrays],
                prepared_sha256=args.prepared_sha256,
            )
            write_new_json(target, observed)
            state["completed"] += 1
            state["pending"] = None
            return result
        except BaseException as exc:
            state["poisoned"] = True
            partial = trace.abort()
            write_new_json(target.with_suffix(".error.json"), dict(error=repr(exc), partial=partial or observed))
            raise

    service.prepare, service.execute = types.MethodType(prepare, service), types.MethodType(execute, service)
    import uvicorn
    from cosmos_framework.inference.action_serving import create_app

    try:
        uvicorn.run(create_app(service), host="127.0.0.1", port=9587, workers=1)
    finally:
        trace.restore()
        write_new_json(output / "mask_evidence.json", evidence)
        write_new_json(output / "closed.json", state)


if __name__ == "__main__":
    main()
