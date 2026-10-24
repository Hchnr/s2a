"""035 real frozen-Cosmos clean-bridge qualification: exactly7 current requests.

This worker accepts the owner's ``s2a.clean-bridge-preparation.v1`` manifest.
It does not prepare manifests, select cases, launch another interpreter, fit a
head, read native futures, or run physics. Run it with the manifest's borrowed
Python3.13.14, frozen Cosmos source on PYTHONPATH, and one allowed GPU UUID.

Required owner fields are root/config/source_sha256/snapshot_sha256/
input_sha256/weight_files/requests3. Config explicitly fixes model_python,
cosmos_source_root, gpu_uuid, allowed_gpu_uuids, checkpoint, python_version,
noise_roster, feature_terms=[], nonzero_clean_residual_active10,
nonzero_request_name, mask_policy and the7/245 budget. Each requests3 entry
contains name/request/image_ref/expected_response_ref/historical_entry; the
historical entry supplies its original request or request_ref. Only vision_path
may change when locating the byte-identical historical current image.

The masked stock baseline uses the frozen native Action128 mask plus the
explicit frozen S2A active-slot wrapper. Its3 outputs must match historical
gold. A zero clean head must reproduce all3 outputs and all35 complete latent
trajectories exactly. One constant small nonzero clean head then demonstrates
its returned velocity reaching the original scheduler conversion. This is
interface/numerical qualification, not task success or trained-model evidence.
"""

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import os
import platform
import random
import sys
import types
from pathlib import Path

import numpy as np

from s2a.sampler_trace import check_sources, file_sha, observation_guard, parameters_identity
from s2a.wam_clean_capture import CleanCaptureBridge, NoiseRoster
from s2a.wam_clean_readout import CleanResidualReadout
from s2a.wam_readout import ACTIVE_MASK, HIDDEN_SIZE, LIBERO_DOMAIN
from s2a.wam_trajectory_capture import same_sample_view, tensor_array

ALLOWED_GPU_UUIDS = (
    "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f",
    "GPU-7a163b30-e2cd-f747-6075-3265058ad0b8",
)
BUDGET = dict(requests=7, denoise_calls=245, model_initializations=1, fit_groups=0, physics_steps=0, fresh_cases=0)
NATIVE_MASK_SOURCES = (
    "cosmos_framework/model/generator/omni_mot_model.py",
    "cosmos_framework/model/generator/utils/data_and_condition.py",
    "cosmos_framework/inference/action_serving.py",
)
SAMPLER_SOURCE = "cosmos_framework/model/generator/diffusion/samplers/unipc.py"
CONVERSION_SOURCE = "cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def checked_json(ref):
    require(set(ref) == {"path", "sha256"}, "input reference must contain exactly path and sha256")
    return importlib.import_module("run_wam_clean_bridge").read(ref)


def resolve(path):
    return importlib.import_module("run_wam_clean_bridge").resolve(path)


def response_arrays(response):
    content = response["outputs"][0]["content"]
    arrays = dict(
        action=np.asarray(content["action"]), normalized_action=np.asarray(content["action128"]["normalized_action"])
    )
    require(
        all(
            value.shape == (40, 128) and np.isfinite(value).all() and not np.any(value[:, 10:])
            for value in arrays.values()
        ),
        "response must contain finite masked full40x128 raw and normalized outputs",
    )
    return arrays


def outputs_exact(actual, expected):
    actual, expected = response_arrays(actual), response_arrays(expected)
    return all(np.array_equal(actual[key], expected[key]) for key in actual)


def require_exact_outputs(actual, expected):
    require(outputs_exact(actual, expected), "raw/normalized historical or zero output differs")


def deployment_gold(raw, residual, sigma, dtype):
    """Independent NumPy FP64 math and direct target-dtype rounding, not Torch.

    BF16 snapshots are represented losslessly as FP32. This reproduces direct
    FP64-to-BF16 ties-to-even with the BF16 minimum-subnormal spacing, avoiding
    an intervening FP32 rounding. CUDA qualification must still execute it.
    """
    raw = np.asarray(raw)
    residual = np.asarray(residual, dtype=np.float64)
    require(raw.shape == residual.shape == (40, 10), "qualification velocity/residual must be active40x10")
    require(np.isfinite(sigma) and sigma > 0, "actual sigma must be finite and positive")
    value = raw.astype(np.float64) - residual / sigma
    require(np.isfinite(value).all(), "ideal adapted velocity is nonfinite")
    if dtype == "torch.bfloat16":
        _, exponent = np.frexp(value)
        spacing = np.ldexp(np.ones_like(value), np.maximum(exponent - 8, -133))
        value = np.rint(value / spacing) * spacing
        result = value.astype(np.float32)
    elif dtype == "torch.float32":
        result = value.astype(np.float32)
    else:
        raise ValueError("035 qualifies only original BF16 or FP32 velocity")
    require(np.isfinite(result).all(), "rounded adapted velocity is nonfinite")
    return result


def audit_online_point(point, expected_residual):
    """Independently audit one saved actual point before accepting its request."""
    arrays, meta = point.arrays, point.metadata
    offset = meta["action_offset"]
    raw, deployed, sample = (arrays[key] for key in ("raw_velocity_flat", "deployed_velocity_flat", "sample_fp32_flat"))
    require(
        sample.dtype == np.float32 and sample.ndim == raw.ndim == deployed.ndim == 1, "actual sample is not flat FP32"
    )
    require(
        sample.shape == raw.shape == deployed.shape == (offset + 5120,), "actual flat sample/velocity layout differs"
    )
    require(
        np.array_equal(sample[offset:].reshape(40, 128), arrays["sample_full128_fp32"]), "action state slice differs"
    )
    require(
        np.array_equal(raw[offset:].reshape(40, 128), arrays["raw_velocity_full128"]),
        "raw action velocity slice differs",
    )
    require(
        np.array_equal(deployed[offset:].reshape(40, 128), arrays["deployed_velocity_full128"]),
        "deployed action slice differs",
    )
    require(
        raw[:offset].tobytes() == deployed[:offset].tobytes()
        and raw[offset:].reshape(40, 128)[:, 10:].tobytes() == deployed[offset:].reshape(40, 128)[:, 10:].tobytes(),
        "adaptation changed vision or inactive velocity bits",
    )
    require(not np.any(sample[offset:].reshape(40, 128)[:, 10:]), "sample inactive118 are not zero")
    sigma = meta["sigma_actual"]
    expected_c0 = sample[offset:].reshape(40, 128).astype(np.float64) - sigma * raw[offset:].reshape(40, 128).astype(
        np.float64
    )
    require(
        np.array_equal(expected_c0, arrays["c0_full128"]),
        "C0 did not use the actual FP32 sampler state and raw velocity",
    )
    residual = np.broadcast_to(np.asarray(expected_residual, dtype=np.float64), (40, 10))
    require(np.array_equal(residual, arrays["clean_residual_active10"]), "fixed clean residual differs")
    require(
        np.array_equal(-residual / sigma, arrays["velocity_delta_active10"]), "clean-to-velocity sign or sigma differs"
    )
    dtype = dict(meta["original_dtypes"])["raw_velocity_flat"]
    gold = deployment_gold(raw[offset:].reshape(40, 128)[:, :10], residual, sigma, dtype)
    require(
        np.array_equal(gold, deployed[offset:].reshape(40, 128)[:, :10]),
        "actual deployed velocity differs from NumPy gold",
    )
    return bool(raw.tobytes() != deployed.tobytes())


def validate_prepared(prepared):
    """Validate concrete current-only requests without loading any model."""
    require(prepared["schema_version"] == "s2a.clean-bridge-preparation.v1", "035 preparation schema differs")
    config = prepared["config"]
    require(config["python_version"] == [3, 13, 14], "035 requires the registered Python3.13.14 runtime")
    require(config["gpu_uuid"] in ALLOWED_GPU_UUIDS, "GPU UUID is outside current visible2/3")
    require(tuple(config["allowed_gpu_uuids"]) == ALLOWED_GPU_UUIDS, "allowed GPU UUID roster differs")
    require(config["feature_terms"] == [], "035 qualification must not introduce an unregistered time basis")
    require(config["mask_policy"] == "native_and_registered_active_slot_wrapper", "mask instrumentation policy differs")
    require(config["budget"] == BUDGET, "035 exact7/245 zero-fit/no-physics budget differs")
    roster = NoiseRoster(**config["noise_roster"])
    residual = np.asarray(config["nonzero_clean_residual_active10"], dtype=np.float64)
    require(
        residual.shape == (10,) and np.isfinite(residual).all() and 0 < np.max(np.abs(residual)) <= 1e-4,
        "nonzero head must be an explicitly fixed small active10 clean bias",
    )
    cosmos = Path(config["cosmos_source_root"]).resolve()
    sources = prepared["source_sha256"]
    required = [cosmos / name for name in (*NATIVE_MASK_SOURCES, SAMPLER_SOURCE, CONVERSION_SOURCE)]
    local = Path(__file__).resolve().parents[1]
    required.extend(
        local / name
        for name in (
            "scripts/qualify_wam_clean_bridge.py",
            "scripts/run_wam_readout_features.py",
            "scripts/run_wam_readout_sampling.py",
            "scripts/audit_wam_current_numerics.py",
            "src/s2a/wam_clean_capture.py",
            "src/s2a/wam_clean_readout.py",
            "src/s2a/wam_runtime.py",
        )
    )
    require(
        all(sources.get(str(path)) == file_sha(path) for path in required),
        "native masks/bridge/helper source closure is not frozen",
    )
    check_sources(sources)
    checkpoint = config["checkpoint"]
    require(
        checkpoint["name"] == "8400_masked" and checkpoint["iteration"] == "iter_000008400",
        "qualification must keep original8400 non-EMA",
    )
    require(
        len(prepared["requests3"]) == 3 and len({entry["name"] for entry in prepared["requests3"]}) == 3,
        "requires exactly3 named developer requests",
    )
    require(
        config["nonzero_request_name"] in {entry["name"] for entry in prepared["requests3"]},
        "nonzero case is outside the3 registered current inputs",
    )
    for entry in prepared["requests3"]:
        request = entry["request"]
        image = entry["image_ref"]
        require(file_sha(resolve(image["path"])) == image["sha256"], "current image SHA changed")
        historical_entry = entry["historical_entry"]
        historical = historical_entry.get("request")
        if historical is None:
            historical = checked_json(historical_entry["request_ref"])
        migrated = dict(historical, vision_path=str(resolve(image["path"])))
        require(request == migrated, "current request changed fields other than historical current-image path")
        allowed = {
            "action_chunk_size",
            "domain_name",
            "extra",
            "fps",
            "guidance",
            "image_size",
            "model_mode",
            "name",
            "num_frames",
            "num_steps",
            "prompt",
            "seed",
            "shift",
            "vision_path",
        }
        require(
            set(request) <= allowed and set(request["extra"]) == {"action128"},
            "request includes unregistered online conditioning",
        )
        action = request["extra"]["action128"]
        require(
            set(action)
            == {
                "action_dim_mask",
                "action_domain",
                "action_fps",
                "camera_layout",
                "return_video",
                "selected_roles",
                "task_title",
            },
            "action request includes unregistered or future inputs",
        )
        require(
            request["model_mode"] == "wam"
            and request["action_chunk_size"] == 40
            and request["num_frames"] == 41
            and request["num_steps"] == 35
            and request["guidance"] == 1.0
            and request["shift"] == 5.0
            and request["fps"] == action["action_fps"] == 20
            and action["action_domain"] == LIBERO_DOMAIN
            and tuple(action["action_dim_mask"]) == ACTIVE_MASK
            and action["return_video"] is False
            and action["task_title"] == request["prompt"],
            "current request differs from fixed full-domain H40/20Hz/guidance1 masked WAM",
        )
        roster.admit(request["seed"], "evaluation")
        response_arrays(checked_json(entry["expected_response_ref"]))
    return roster


class ConversionWitness(CleanCaptureBridge):
    """Observe the actual scheduler receipt of the bridge's returned object."""

    def __init__(self, *args, conversion_source_ref, **kwargs):
        self.conversion_source_ref = conversion_source_ref
        self.conversion_override = self.pending_conversion = None
        super().__init__(*args, **kwargs)

    @contextlib.contextmanager
    def request(self, *args, **kwargs):
        try:
            with super().request(*args, **kwargs) as current:
                yield current
                require(
                    all(step.get("conversion_calls") == 1 for step in current["steps"]),
                    "actual scheduler conversions are incomplete",
                )
                require(self.pending_conversion is None, "final returned velocity never reached its scheduler")
        finally:
            if self.conversion_override is not None:
                scheduler, _ = self.conversion_override
                del scheduler.convert_model_output
            self.conversion_override = self.pending_conversion = None

    def _observe(self, timestep, caller):
        require(self.pending_conversion is None, "prior velocity was not consumed before the next neural call")
        sigma, record = super()._observe(timestep, caller)
        if self.conversion_override is None:
            scheduler = self.current["scheduler"]
            require("convert_model_output" not in scheduler.__dict__, "scheduler conversion already overridden")
            original = scheduler.convert_model_output
            body = inspect.unwrap(original.__func__)
            source = self.conversion_source_ref
            require(
                str(Path(body.__code__.co_filename).resolve()) == source["path"]
                and self.sources.get(source["path"]) == source["sha256"] == file_sha(source["path"])
                and scheduler.predict_x0 is True
                and scheduler.config.prediction_type == "flow_prediction"
                and scheduler.config.thresholding is False,
                "actual scheduler conversion is not the frozen unchanged flow-to-x0 implementation",
            )
            self.conversion_override = (scheduler, original)
            owner = self

            def convert(instance, model_output, *args, **kwargs):
                with observation_guard(owner.model):
                    pending = owner.pending_conversion
                    require(pending is not None, "foreign or repeated actual scheduler conversion")
                    expected_velocity, expected_sample, actual_record = pending
                    sample = kwargs.get("sample")
                    require(
                        instance is owner.current["scheduler"]
                        and instance.step_index == actual_record["step_index"]
                        and model_output is expected_velocity
                        and same_sample_view(sample, expected_sample),
                        "scheduler did not receive the exact returned velocity and original FP32 sample view",
                    )
                    saved = owner.current["points"][-1]
                    require(
                        tensor_array(model_output).tobytes() == saved.arrays["deployed_velocity_flat"].tobytes()
                        and tensor_array(sample).reshape(-1).tobytes() == saved.arrays["sample_fp32_flat"].tobytes(),
                        "actual scheduler input bytes differ from the saved deployed velocity or FP32 sample",
                    )
                    actual_record.update(
                        conversion_calls=1,
                        conversion_velocity_is_returned_object=True,
                        conversion_velocity_is_deployed_storage_view=True,
                        conversion_sample_is_original_storage_view=True,
                        conversion_velocity_dtype=str(model_output.dtype),
                    )
                    owner.pending_conversion = None
                result = original(model_output, *args, **kwargs)
                actual_record["conversion_returned_original_object"] = True
                return result

            scheduler.convert_model_output = types.MethodType(convert, scheduler)
        record["conversion_calls"] = 0
        return sigma, record

    def _finish_velocity(self, raw):
        result = super()._finish_velocity(raw)
        self.pending_conversion = (result[0], self._step["sample"], self._step["record"])
        return result


def _fixed_head(active_bias):
    coefficients = np.zeros((HIDDEN_SIZE + 11, 10), dtype=np.float64)
    coefficients[-1] = np.asarray(active_bias, dtype=np.float64)
    return CleanResidualReadout(coefficients, np.zeros(HIDDEN_SIZE + 10), 1.0, 0.1)


def _exact_points(left_refs, right_refs):
    require(len(left_refs) == len(right_refs) == 35, "raw/zero must include all35 actual points")
    for left_ref, right_ref in zip(left_refs, right_refs, strict=True):
        left, right = checked_json(left_ref), checked_json(right_ref)
        for key in ("step_index", "sigma_actual", "network_timestep", "action_offset", "original_dtypes"):
            require(left[key] == right[key], "raw/zero actual node metadata differs")
        left_arrays, right_arrays = left["arrays"], right["arrays"]
        require(
            file_sha(left_arrays["path"]) == left_arrays["sha256"]
            and file_sha(right_arrays["path"]) == right_arrays["sha256"],
            "raw/zero point archive changed",
        )
        with (
            np.load(left_arrays["path"], allow_pickle=False) as a,
            np.load(right_arrays["path"], allow_pickle=False) as b,
        ):
            require(set(a.files) == set(b.files), "raw/zero point array roster differs")
            require(
                all(
                    a[key].dtype == b[key].dtype
                    and a[key].shape == b[key].shape
                    and a[key].tobytes() == b[key].tobytes()
                    for key in a.files
                ),
                "zero head changed an actual sampler trajectory/hidden/packed input/velocity bit",
            )


def runtime_environment(prepared, gpu_uuid):
    config = prepared["config"]
    require(gpu_uuid == config["gpu_uuid"] and gpu_uuid in ALLOWED_GPU_UUIDS, "requested GPU UUID is not admitted")
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == gpu_uuid and os.environ.get("LD_LIBRARY_PATH") == "",
        "model worker requires its single explicit allowed UUID and empty LD",
    )
    require(
        list(sys.version_info[:3]) == config["python_version"], "worker is not running the registered Python3.13.14"
    )
    require(
        Path(sys.executable).resolve() == Path(config["model_python"]).resolve(),
        "worker interpreter differs from the borrowed registered executable",
    )
    require(
        not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules),
        "Cosmos was imported before frozen-root admission",
    )
    sys.path.insert(0, str(Path(config["cosmos_source_root"]).resolve()))


def imported_cosmos_sources(prepared):
    root = Path(prepared["config"]["cosmos_source_root"]).resolve()
    records = {}
    for name, module in sorted(sys.modules.items()):
        if name == "cosmos_framework" or name.startswith("cosmos_framework."):
            filename = getattr(module, "__file__", None)
            if filename is None:
                records[name] = dict(file=None, namespace_package=True)
                continue
            path = Path(filename).resolve()
            require(path.is_relative_to(root), f"Cosmos module escaped frozen caller source root: {name}")
            digest = file_sha(path)
            require(
                prepared["source_sha256"].get(str(path)) == digest,
                f"imported Cosmos module is outside source closure: {name}",
            )
            records[name] = dict(file=str(path), sha256=digest)
    return records


def run(prepared, prepared_ref, gpu_uuid):
    """Run once in the owner's already-authorized isolated model process."""
    from audit_wam_current_numerics import bundle, require_eval
    from run_wam_readout_features import BoundaryCounter, actual_context
    from run_wam_readout_sampling import InitialCapture

    from s2a.wam_runtime import install_active_slot_mask, load_service

    validate_prepared(prepared)
    runtime_environment(prepared, gpu_uuid)
    config = prepared["config"]
    output = Path(prepared["root"]) / "qualification"
    output.mkdir()
    result = dict(
        schema_version="s2a.clean-bridge-qualification.v1",
        seed=42,
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        gpu_uuid=gpu_uuid,
        python_version=platform.python_version(),
        python_executable=sys.executable,
        cosmos_source_root=config["cosmos_source_root"],
        mask_policy=config["mask_policy"],
        budget=BUDGET,
        generations=[],
        heads={},
        boundary_calls=[],
        wam_model_initializations=0,
        new_candidate_fits=0,
        new_episode_reads=0,
        new_future_reads=0,
        physics_steps=0,
        fresh_cases=0,
        mutation_steps=0,
        bridge_qualified=False,
        wam_model_initialization_attempts=0,
    )
    bridge = counter = initial = model = None
    try:
        import torch

        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "qualification requires exactly one visible allowed CUDA GPU",
        )
        props = torch.cuda.get_device_properties(0)
        actual_uuid = str(props.uuid)
        require(
            actual_uuid.removeprefix("GPU-") == gpu_uuid.removeprefix("GPU-"),
            "driver UUID does not match the admitted visible device",
        )
        result["runtime"] = dict(
            torch_version=torch.__version__, cuda_version=torch.version.cuda, gpu_name=props.name, gpu_uuid=actual_uuid
        )
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        service_dir = output / "service"
        service_dir.mkdir()
        checkpoint = config["checkpoint"]
        result["wam_model_initialization_attempts"] = 1
        service = load_service(dict(checkpoints=[checkpoint]), "8400_masked", service_dir)
        model = service.model
        result["wam_model_initializations"] = 1
        result["cosmos_imported_sources_after_load"] = imported_cosmos_sources(prepared)
        expected_checkpoint = Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model"
        require(
            Path(service.info["checkpoint_path"]).resolve() == expected_checkpoint.resolve()
            and service.info["ema_weights"] is False,
            "actual loaded8400 checkpoint/EMA differs",
        )
        require_eval(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        result["parameters_before"] = parameters_identity(model)
        mask_evidence = install_active_slot_mask(model, ACTIVE_MASK)
        zero, nonzero = (_fixed_head(value) for value in (np.zeros(10), config["nonzero_clean_residual_active10"]))
        for name, head in (("zero", zero), ("nonzero", nonzero)):
            target = output / f"{name}_head.json"
            write_new(target, head.to_parameters())
            result["heads"][f"{name}_ref"] = reference(target)
        sources = prepared["source_sha256"]
        cosmos = Path(config["cosmos_source_root"])
        sampler_path, conversion_path = (str((cosmos / name).resolve()) for name in (SAMPLER_SOURCE, CONVERSION_SOURCE))
        active_points, active_directory, expected_residual = [], None, None

        def sink(point):
            require(
                active_directory is not None and expected_residual is not None,
                "capture sink is outside a registered generation",
            )
            changed = audit_online_point(point, expected_residual)
            result["mutation_steps"] += int(changed)
            active_points.append(
                bundle(
                    active_directory,
                    f"point_{point.metadata['step_index']:02d}",
                    dict(point.arrays),
                    dict(point.metadata),
                )
            )

        bridge = ConversionWitness(
            model,
            noise_roster=NoiseRoster(**config["noise_roster"]),
            readout=None,
            feature_terms=(),
            sink=sink,
            denoise_delegate=model.denoise,
            forward_file=sampler_path,
            forward_sha256=sources[sampler_path],
            sources=sources,
            conversion_source_ref=dict(path=conversion_path, sha256=sources[conversion_path]),
        )
        counter = BoundaryCounter(model, 245)
        initial = InitialCapture(model)
        from cosmos_framework.inference.args import OmniSampleOverrides

        cases = prepared["requests3"]
        nonzero_case = next(case for case in cases if case["name"] == config["nonzero_request_name"])
        entries = [(variant, case) for variant in ("raw", "zero") for case in cases] + [("nonzero", nonzero_case)]
        raw_by_name = {}
        with torch.inference_mode():
            for index, (variant, case) in enumerate(entries):
                request = case["request"]
                name = f"{index:02d}_{variant}_{case['name']}"
                active_directory = output / name
                active_directory.mkdir()
                active_points = []
                expected_residual = np.asarray(
                    config["nonzero_clean_residual_active10"] if variant == "nonzero" else np.zeros(10),
                    dtype=np.float64,
                )
                bridge.readout = dict(raw=None, zero=zero, nonzero=nonzero)[variant]
                entry = dict(
                    generation_index=index,
                    name=name,
                    pair_id=case["name"],
                    variant=variant,
                    sampling_seed=request["seed"],
                )
                generation = dict(
                    index=index,
                    name=name,
                    variant=variant,
                    case_id=case["name"],
                    sampling_seed=request["seed"],
                    complete=False,
                    points=active_points,
                )
                result["generations"].append(generation)
                write_new(active_directory / "request.json", request)
                generation["request_ref"] = reference(active_directory / "request.json")
                initial.arm(entry, active_directory)
                sample, batch = service.prepare(
                    OmniSampleOverrides.model_validate(copy.deepcopy(request)), active_directory
                )
                context = actual_context(service, sample, batch)
                caption = json.loads(batch["ai_caption"][0])
                require(
                    context.is_libero and caption["task_title"] == caption["command"] == request["prompt"],
                    "actual model conditioning is not current-image/task-only",
                )
                write_new(
                    active_directory / "conditioning.json",
                    dict(
                        caption=batch["ai_caption"][0],
                        caption_sha256=hashlib.sha256(batch["ai_caption"][0].encode()).hexdigest(),
                        context=dataclasses.asdict(context),
                        inference_only_current_image=True,
                    ),
                )
                with (
                    bridge.request(context, name=name, seed=request["seed"], purpose="evaluation"),
                    counter.scope(name, 35),
                ):
                    response = service.execute(sample, batch).model_dump(mode="json")
                write_new(active_directory / "response.json", response)
                generation.update(
                    request_ref=reference(active_directory / "request.json"),
                    response_ref=reference(active_directory / "response.json"),
                    conditioning_ref=reference(active_directory / "conditioning.json"),
                    initial_ref=initial.finish(),
                    points=active_points,
                )
                write_new(active_directory / "actual_sigma_trace.json", bridge.last)
                generation["trace_ref"] = reference(active_directory / "actual_sigma_trace.json")
                require(len(active_points) == 35 and bridge.last["complete"], "actual generation/capture is incomplete")
                if variant == "raw":
                    generation["historical_output_exact"] = outputs_exact(
                        response, checked_json(case["expected_response_ref"])
                    )
                    raw_by_name[case["name"]] = generation
                elif variant == "zero":
                    raw = raw_by_name[case["name"]]
                    require_exact_outputs(response, checked_json(raw["response_ref"]))
                    _exact_points(raw["points"], active_points)
                    generation["zero_pair_exact"] = True
                generation["complete"] = True
                print(
                    json.dumps(dict(generation=index + 1, variant=variant, denoise_calls=len(counter.calls))),
                    flush=True,
                )
        require(
            len(counter.calls) == 245 and all(call["completed"] for call in counter.calls),
            "actual7/245 ledger is incomplete",
        )
        require(result["mutation_steps"] > 0, "the fixed nonzero head did not change any actual returned velocity bits")
        result["parameters_after"] = parameters_identity(model)
        require(result["parameters_after"] == result["parameters_before"], "original model parameters changed")
        check_sources(sources)
        result["cosmos_imported_sources_final"] = imported_cosmos_sources(prepared)
        result.update(
            complete=True,
            passed=True,
            bridge_qualified=all(
                generation.get("historical_output_exact") is True
                for generation in result["generations"]
                if generation["variant"] == "raw"
            ),
            actual_requests=7,
            actual_denoise_calls=245,
            actual_head_calls=bridge.total_head_calls,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result["boundary_calls"] = [] if counter is None else counter.calls
        if model is not None:
            result["mask_evidence"] = locals().get("mask_evidence", [])
        cleanup_errors = []
        for label, remove in (
            ("initial", None if initial is None else initial.remove),
            ("counter", None if counter is None else lambda: setattr(model, "denoise", counter.original)),
            ("bridge", None if bridge is None else bridge.restore),
        ):
            if remove is not None:
                try:
                    remove()
                except BaseException as error:
                    cleanup_errors.append(dict(stage=label, type=type(error).__name__, message=str(error)))
        if cleanup_errors:
            result.update(complete=False, passed=False, bridge_qualified=False, cleanup_errors=cleanup_errors)
        write_new(output / "result.json", result)
    require(result["passed"], "035 qualification cleanup failed")
    return output / "result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    parser.add_argument("--gpu-uuid", choices=ALLOWED_GPU_UUIDS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    owner = importlib.import_module("run_wam_clean_bridge")
    prepared = owner.read_prepared(args.prepared, args.prepared_sha256)
    validate_prepared(prepared)
    if args.check_only:
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("LD_LIBRARY_PATH") == "",
            "CPU admission check requires empty CUDA/LD",
        )
        print(json.dumps(dict(passed=True, model_initializations=0, denoise_calls=0, physics_steps=0)), flush=True)
        return
    path = run(prepared, reference(args.prepared), args.gpu_uuid or prepared["config"]["gpu_uuid"])
    print(path, flush=True)


if __name__ == "__main__":
    main()
