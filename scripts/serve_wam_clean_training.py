"""038 serial private raw/W2 service; exactly12 cases and900 H40 requests.

The fixed CPU executor keeps its original /info and /generate contract. Case
binding is separate from online inputs: it selects only raw or the one frozen
W2 head, seed and task. The service never opens simulator or future archives.
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from collect_wam_clean_training import normalization_spec, qualified_grid, same_initial
from qualify_wam_clean_bridge import (
    CONVERSION_SOURCE,
    SAMPLER_SOURCE,
    ConversionWitness,
    imported_cosmos_sources,
    response_arrays,
    runtime_environment,
)
from run_wam_clean_bridge import check_owned_command

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
from s2a.wam_readout import ACTIVE_MASK

PORT = 31339
SERVICE_SCHEMA = "s2a.clean-training-development-service.v1"


def case_order():
    return [
        dict(
            case_index=i,
            case_id=f"task{task}_init0_s{seed}_{variant}",
            task_index=task,
            record_name=f"task{task}_init0",
            sampling_seed=seed,
            variant=variant,
            expected_requests=75,
        )
        for i, (task, seed, variant) in enumerate(
            (task, seed, variant) for task in range(3) for seed in (42, 43) for variant in ("raw", "w2")
        )
    ]


def case_directory(root, case):
    return (
        Path(root)
        / "development/cases"
        / case["case_id"]
        / "simulation/8400_masked_position_hold"
        / (f"{case['record_name']}_s{case['sampling_seed']}")
    )


class DevelopmentPolicy:
    """One irreversible ordered ledger; no model or future-data dependency."""

    def __init__(self, prepared):
        self.root = Path(prepared["root"]).resolve()
        self.order = case_order()
        self.templates = {e["request"]["prompt"]: e["request"] for e in prepared["zero_identity_cases"]}
        self.tasks = {}
        for anchor in prepared["validation_anchors"]:
            task = anchor["task_index"]
            require(task not in self.tasks or self.tasks[task] == anchor["task"], "task metadata is inconsistent")
            self.tasks[task] = anchor["task"]
        require(
            set(self.tasks) == {0, 1, 2} and set(self.templates) == set(self.tasks.values()),
            "three fixed tasks required",
        )
        self.case_index = -1
        self.case_requests = 0
        self.requests = []
        self.pending = None
        self.poisoned = False
        self.stopped = False
        self.finish_requested = False
        self.first_raw = {}

    @property
    def current(self):
        return None if self.case_index < 0 else self.order[self.case_index]

    def bind(self, body):
        require(
            not self.poisoned and not self.stopped and self.pending is None, "service is failed, closed or occupied"
        )
        require(
            self.case_index < 11 and (self.case_index < 0 or self.case_requests == 75), "preceding case is incomplete"
        )
        expected = self.order[self.case_index + 1]
        require(body == expected, "case order/identity differs from the complete fixed12 matrix")
        self.case_index += 1
        self.case_requests = 0
        return self.current

    def reserve(self, request):
        require(
            not self.poisoned and not self.stopped and self.pending is None, "service is failed, closed or occupied"
        )
        case = self.current
        require(
            case is not None and self.case_requests < 75 and len(self.requests) < 900, "case/request budget exhausted"
        )
        directory = case_directory(self.root, case)
        image = directory / f"input_{self.case_requests:04d}.png"
        require(
            image.resolve().is_relative_to(self.root) and request.get("vision_path") == str(image),
            "current image is outside the bound case directory",
        )
        image_ref = reference(image)
        task = self.tasks[case["task_index"]]
        validate_current_request(
            request,
            name=f"s2a_{directory.name}_{self.case_requests}",
            seed=case["sampling_seed"],
            image_ref=image_ref,
            task=task,
            template=self.templates[task],
        )
        key = (case["task_index"], case["sampling_seed"])
        if case["variant"] == "w2" and self.case_requests == 0:
            require(
                key in self.first_raw and image_ref["sha256"] == self.first_raw[key]["image_ref"]["sha256"],
                "raw/W2 first current PNG bytes differ",
            )
        self.pending = dict(
            case_index=self.case_index,
            case_id=case["case_id"],
            request_index=self.case_requests,
            global_request_index=len(self.requests),
            variant=case["variant"],
            sampling_seed=case["sampling_seed"],
            image_ref=image_ref,
        )
        return dict(self.pending)

    def complete(self, record):
        require(self.pending is not None and not self.poisoned and not self.stopped, "no admitted pending request")
        require(all(record.get(k) == v for k, v in self.pending.items()), "completed request identity changed")
        require(record.get("complete") is True, "partial model request cannot be counted")
        case = self.current
        if self.case_requests == 0 and case["variant"] == "raw":
            self.first_raw[(case["task_index"], case["sampling_seed"])] = dict(record)
        self.requests.append(record)
        self.case_requests += 1
        self.pending = None

    def finish(self, body):
        require(body == dict(completed_cases=12, completed_requests=900), "finish body differs from fixed budget")
        require(
            not self.poisoned
            and self.pending is None
            and self.case_index == 11
            and self.case_requests == 75
            and len(self.requests) == 900,
            "complete12/900 ledger required",
        )
        self.finish_requested = self.stopped = True

    def abort(self, body):
        require(
            isinstance(body, dict)
            and set(body) == {"case_id", "reason"}
            and body["case_id"] == (None if self.current is None else self.current["case_id"])
            and isinstance(body["reason"], str)
            and 0 < len(body["reason"]) <= 1024,
            "invalid owned abort",
        )
        self.poisoned = self.stopped = True
        return dict(body)


def admit_state(prepared, prepared_ref, registration_ref, owner_ref, verification_ref):
    root = Path(prepared["root"])
    require(
        Path(owner_ref["path"]) == root / "orchestration/STATE/result.json"
        and Path(verification_ref["path"]) == root / "review/STATE_verified.json",
        "state evidence escaped fixed stage",
    )
    owner, audit = read(owner_ref), read(verification_ref)
    require(
        owner.get("schema_version") == "s2a.clean-training-state-owner.v1"
        and owner.get("complete") is True
        and owner.get("passed") is True
        and owner.get("prepared_ref") == prepared_ref
        and owner.get("execution_registration_ref") == registration_ref
        and owner.get("verification_ref") == verification_ref
        and len(owner["owned_commands"]) == 2,
        "normally completed owned99 state stage required",
    )
    for command in owner["owned_commands"]:
        check_owned_command(command)
    require(
        audit.get("schema_version") == "s2a.clean-training-state-verification.v1"
        and audit.get("complete") is True
        and audit.get("passed") is True
        and audit.get("prepared_ref") == prepared_ref
        and audit.get("execution_registration_ref") == registration_ref
        and audit.get("sampling_ref") == owner.get("producer_ref")
        and audit.get("numerical_contract_passed") is True
        and audit.get("eligible_for_development") is True
        and type(audit.get("state_gate_passed")) is bool
        and audit.get("actual_requests") == 99
        and audit.get("actual_denoise_calls") == 3465,
        "complete numerical state qualification required; pointwise state gate is diagnostic",
    )
    require(owner["gpu_after_ref"] == audit["gpu_after_ref"], "state model resource release binding differs")
    gpu = read(owner["gpu_after_ref"])
    require(
        gpu.get("released") is True
        and gpu.get("memory_mib") == 0
        and gpu.get("compute_apps") == []
        and gpu.get("uuid") == GPU,
        "state model GPU release missing",
    )
    head, params = trained_head(
        audit["w2_fit_ref"],
        audit["w2_verification_ref"],
        stage="W2",
        prepared_ref=prepared_ref,
        registration_ref=registration_ref,
    )
    require(params == audit["readout_ref"], "state diagnostic used a different final W2")
    return audit, head, params


def process_identity():
    stat = Path("/proc/self/stat").read_text()
    return dict(
        pid=os.getpid(),
        ppid=os.getppid(),
        start_ticks=int(stat[stat.rfind(")") + 2 :].split()[19]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ld_library_path=os.environ.get("LD_LIBRARY_PATH"),
    )


def handler_class(policy, generate, info, status, abort_record):
    """The HTTP layer cannot choose heads or change the fixed request budget."""

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/info":
                self.send_json(info())
            elif self.path == "/status":
                self.send_json(status())
            elif self.path == "/healthz":
                self.send_json(dict(status="ready" if not policy.poisoned else "failed"))
            else:
                self.send_json(dict(error="unknown route"), 404)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                require(0 < size <= 1024 * 1024, "bounded nonempty JSON body required")
                body = json.loads(self.rfile.read(size))
                if self.path == "/bind_case":
                    value = dict(bound=policy.bind(body), **status())
                elif self.path == "/generate":
                    value = generate(body)
                elif self.path == "/finish":
                    policy.finish(body)
                    value = dict(finished=True, **status())
                elif self.path == "/abort":
                    abort_record(policy.abort(body))
                    value = dict(aborted=True, **status())
                else:
                    raise ValueError("unknown mutation route")
                self.send_json(value)
            except BaseException as error:
                policy.poisoned = policy.stopped = True
                abort_record(dict(type=type(error).__name__, message=str(error)))
                self.send_json(dict(status="error", error=str(error)), 422 if isinstance(error, ValueError) else 500)

    return Handler


def run(prepared_ref, registration_ref, *, state_owner_ref, state_verification_ref, host, port, gpu_uuid):
    required = [
        Path(__file__),
        REPO / "src/s2a/clean_training_runtime.py",
        REPO / "scripts/collect_wam_clean_training.py",
    ]
    prepared, registration, sources = admit_execution(prepared_ref, registration_ref, required_sources=required)
    require(
        host == "127.0.0.1"
        and port == PORT == registration["development"]["port"]
        and registration["development"]["case_order"] == case_order(),
        "registered loopback12-case service required",
    )
    audit, head, params_ref = admit_state(
        prepared, prepared_ref, registration_ref, state_owner_ref, state_verification_ref
    )
    qualified = read(prepared["prerequisite_refs"]["bridge_prepared"])
    runtime_environment(qualified, gpu_uuid)
    grid = qualified_grid(prepared)
    directory = Path(prepared["root"]) / "development/service"
    directory.mkdir(parents=True, exist_ok=False)
    policy = DevelopmentPolicy(prepared)
    result = dict(
        schema_version=SERVICE_SCHEMA,
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        state_owner_ref=state_owner_ref,
        state_verification_ref=state_verification_ref,
        w2_fit_ref=audit["w2_fit_ref"],
        w2_verification_ref=audit["w2_verification_ref"],
        readout_ref=params_ref,
        requests=policy.requests,
        source_sha256=sources,
        gpu_uuid=gpu_uuid,
        host=host,
        port=port,
        actual_requests=0,
        actual_denoise_calls=0,
        actual_head_calls=0,
        model_initialization_attempts=0,
        model_initializations=0,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    bridge = initial = counter = model = httpd = None
    try:
        import torch
        from audit_wam_current_numerics import require_eval
        from run_wam_readout_features import BoundaryCounter, actual_context
        from run_wam_readout_sampling import InitialCapture

        from s2a.wam_runtime import install_active_slot_mask, load_service

        require(
            gpu_uuid == GPU and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "exactly one registered GPU UUID required",
        )
        require(
            str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") == GPU.removeprefix("GPU-"),
            "actual GPU UUID differs",
        )
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model_dir = directory / "model"
        model_dir.mkdir()
        result["model_initialization_attempts"] = 1
        checkpoint = prepared["model_config"]["checkpoint"]
        service = load_service(dict(checkpoints=[checkpoint]), "8400_masked", model_dir)
        result["model_initializations"] = 1
        model = service.model
        expected_checkpoint = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
        require(
            Path(service.info["checkpoint_path"]).resolve() == expected_checkpoint
            and service.info["ema_weights"] is False,
            "loaded checkpoint/EMA differs",
        )
        require_eval(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        result["parameters_before"] = parameters_identity(model)
        result["mask_evidence"] = install_active_slot_mask(model, ACTIVE_MASK)
        result["cosmos_imported_sources_after_load"] = imported_cosmos_sources(qualified)
        result["normalization_ref"] = publish(directory / "normalization.json", normalization_spec(service.spec))
        cosmos = Path(qualified["config"]["cosmos_source_root"])
        sampler, conversion = (str((cosmos / value).resolve()) for value in (SAMPLER_SOURCE, CONVERSION_SOURCE))
        sink = None

        def consume(point):
            require(sink is not None, "capture outside an admitted online request")
            sink(point)

        bridge = ConversionWitness(
            model,
            noise_roster=NoiseRoster(**prepared["noise_roster"]),
            readout=None,
            feature_terms=(),
            sink=consume,
            denoise_delegate=model.denoise,
            forward_file=sampler,
            forward_sha256=sources[sampler],
            sources=sources,
            conversion_source_ref=dict(path=conversion, sha256=sources[conversion]),
        )
        counter, initial = BoundaryCounter(model, 31500), InitialCapture(model)
        from cosmos_framework.inference.args import OmniSampleOverrides

        identity = process_identity()
        info_base = dict(
            service.info,
            port=port,
            config_file=str(Path(checkpoint["run"]) / "config.yaml"),
            acceptance_intervention=dict(
                active_slot_mask=True,
                exact_published_caption=False,
                warm_start_contract_run=None,
                implementation="serve_wam_clean_training.py",
                service_source_ref=reference(__file__),
                code=dict(
                    files_sha256={
                        str(Path(path).relative_to(REPO)): digest
                        for path, digest in sources.items()
                        if Path(path).is_relative_to(REPO)
                    },
                    scope="frozen source files; presence is not an old-server execution claim",
                ),
            ),
        )

        def status():
            case = policy.current
            return dict(
                ready=not policy.poisoned and not policy.stopped,
                current_case_id=None if case is None else case["case_id"],
                case_index=policy.case_index,
                variant=None if case is None else case["variant"],
                case_requests=policy.case_requests,
                actual_requests=len(policy.requests),
                actual_denoise_calls=len(counter.calls),
                actual_head_calls=bridge.total_head_calls,
                denoise=len(counter.calls),
                head_calls=bridge.total_head_calls,
                process_identity=identity,
                model_initializations=1,
                model_initialization_attempts=1,
                prepared_ref=prepared_ref,
                execution_registration_ref=registration_ref,
                state_verification_ref=state_verification_ref,
            )

        def info():
            return dict(
                info_base,
                **status(),
                clean_training=dict(
                    prepared_ref=prepared_ref,
                    execution_registration_ref=registration_ref,
                    state_owner_ref=state_owner_ref,
                    state_verification_ref=state_verification_ref,
                    w2_fit_ref=audit["w2_fit_ref"],
                    w2_verification_ref=audit["w2_verification_ref"],
                    readout_ref=params_ref,
                    variant=None if policy.current is None else policy.current["variant"],
                    gpu_uuid=GPU,
                    service_source_ref=reference(__file__),
                    legacy_executor_gpu_label_only=4,
                ),
            )

        def abort_record(value):
            result.setdefault("abort_records", []).append(value)

        def generate(request):
            nonlocal sink
            record = policy.reserve(request)
            record["name"] = request["name"]
            record["anchor_name"] = policy.current["case_id"]
            current = (
                directory
                / "requests"
                / f"{record['global_request_index']:03d}_{record['case_id']}_q{record['request_index']:02d}"
            )
            current.mkdir(parents=True, exist_ok=False)
            publish(current / "started.json", dict(record, complete=False))
            record["request_ref"] = publish(current / "request.json", request)
            case = policy.current
            deployed_head = None if case["variant"] == "raw" else head
            readout_ref = None if deployed_head is None else params_ref
            bridge.readout = deployed_head
            record["readout_ref"] = readout_ref
            sink = CompactCaptureSink(
                current / "compact",
                CompactRequest(
                    request["name"],
                    case["sampling_seed"],
                    "evaluation",
                    "development",
                    case["case_id"],
                    None if readout_ref is None else readout_ref["sha256"],
                    record["image_ref"]["sha256"],
                ),
                expected_vision_shape=VISION_SHAPE,
                expected_grid=grid,
            )
            initial.arm(
                dict(
                    generation_index=record["global_request_index"],
                    name=request["name"],
                    pair_id=case["case_id"],
                    variant=case["variant"],
                    sampling_seed=case["sampling_seed"],
                ),
                current,
            )
            checked(record["image_ref"])
            sample, batch = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(request)), current)
            checked(record["image_ref"])
            context = actual_context(service, sample, batch)
            caption = json.loads(batch["ai_caption"][0])
            require(
                context.is_libero
                and caption["task_title"] == caption["command"] == request["prompt"]
                and caption["task_plan"] == [],
                "privileged or inconsistent actual conditioning",
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
                bridge.request(context, name=request["name"], seed=case["sampling_seed"], purpose="evaluation"),
                counter.scope(request["name"], 35),
            ):
                response = service.execute(sample, batch).model_dump(mode="json")
            response_arrays(response)
            require("clean_training" not in response, "stock response unexpectedly contains our provenance namespace")
            response["clean_training"] = {
                key: record[key]
                for key in (
                    "case_index",
                    "case_id",
                    "request_index",
                    "global_request_index",
                    "variant",
                    "sampling_seed",
                    "readout_ref",
                )
            }
            record["response_ref"] = publish(current / "response.json", response)
            record["initial_ref"] = initial.finish()
            if record["request_index"] == 0 and case["variant"] == "w2":
                raw_first = policy.first_raw[(case["task_index"], case["sampling_seed"])]
                same_initial(raw_first["initial_ref"], record["initial_ref"])
                record["first_raw_initial_ref"] = raw_first["initial_ref"]
            record["trace_ref"] = publish(
                current / "actual_sigma_trace.json",
                dict(
                    bridge.last,
                    source_frozen=True,
                    mask_policy="native_and_registered_active_slot_wrapper",
                ),
            )
            record["compact_ledger_ref"] = sink.finish()
            record["complete"] = True
            policy.complete(record)
            publish(current / "completed.json", record)
            checked(params_ref)
            bridge.last_points = ()
            sink = None
            print(
                json.dumps(dict(case=case["case_id"], requests=len(policy.requests), denoise=len(counter.calls))),
                flush=True,
            )
            return response

        handler = handler_class(policy, generate, info, status, abort_record)
        httpd = HTTPServer((host, port), handler)
        httpd.timeout = 1
        result["process_identity"] = identity
        result["initial_info_ref"] = publish(directory / "ready.json", info())
        print(json.dumps(dict(server_ready=True, host=host, port=port, process_identity=identity)), flush=True)
        with torch.inference_mode():
            while not policy.stopped:
                httpd.handle_request()
        require(
            policy.finish_requested
            and not policy.poisoned
            and len(policy.requests) == 900
            and len(counter.calls) == bridge.total_head_calls == 31500
            and all(c["completed"] for c in counter.calls),
            "service ended before complete fixed12/900 budget",
        )
        result["parameters_after"] = parameters_identity(model)
        require(result["parameters_before"] == result["parameters_after"], "base model parameters changed")
        result["cosmos_imported_sources_final"] = imported_cosmos_sources(qualified)
        check_sources(sources)
        admit_execution(prepared_ref, registration_ref, required_sources=required)
        checked(params_ref)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if httpd is not None:
            httpd.server_close()
        result.update(
            actual_requests=len(policy.requests),
            actual_denoise_calls=0 if counter is None else len(counter.calls),
            actual_head_calls=0 if bridge is None else bridge.total_head_calls,
            boundary_calls=[] if counter is None else counter.calls,
            poisoned=policy.poisoned,
            finish_requested=policy.finish_requested,
            pending=policy.pending,
        )
        errors = []
        for label, remove in (
            ("initial", None if initial is None else initial.remove),
            ("counter", None if counter is None else lambda: setattr(model, "denoise", counter.original)),
            ("bridge", None if bridge is None else bridge.restore),
        ):
            if remove is not None:
                try:
                    remove()
                except BaseException as error:
                    errors.append(dict(stage=label, type=type(error).__name__, message=str(error)))
        if errors:
            result.update(complete=False, passed=False, cleanup_errors=errors)
        publish(directory / "result.json", result)
    require(result["passed"], "private development service cleanup failed")
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration", "state-owner", "state-verification"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--host", choices=("127.0.0.1",), required=True)
    parser.add_argument("--port", type=int, choices=(PORT,), required=True)
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
                state_owner_ref=ref("state_owner"),
                state_verification_ref=ref("state_verification"),
                host=args.host,
                port=args.port,
                gpu_uuid=args.gpu_uuid,
            )
        )
    )


if __name__ == "__main__":
    main()
