"""Generate one WAM reference from an independently replayed step-48 image."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_reset_live_second_window as live
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import reset_multitask_live_requests as tasks
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_h40_reobserve_model_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_h40_reobserve_model_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_h40_reobserve_model_20261112"
QA = ROOT / "outputs/qa/reset_bowl_h40_reobserve_model_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_h40_reobserve_model_dry_20261112/result.json"
CAPTURE = {
    "path": str(ROOT / "outputs/reset_bowl_h40_reobserve_capture_20261112/result.json"),
    "sha256": "456d3124b040537ab7572c0fa0d990c29fe77737499ea31f0eb47c26d334c161",
}
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"
TASK = "put the bowl on the plate"
IMAGE_SHA = "241a42e2930fc5865864668f8cc40d6222c3359e59e83008c7bc96bc662d592a"
STATE_SHA = "fe7dab2dbc8fc55aed2f4bf9d4d9c18604a400989782bbca091b79494f485937"
MODEL_SHA = "48138bb8c95ce8cc1ca0b70f925e57c8f16bfb21be0a243495b391a06cf5b8ec"
SESSION = "reset_bowl_h40_step48_offline_s42_20261112"
FIXED = dict(
    schema="s2a.reset-bowl-h40-reobserve-model.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    task=TASK,
    seed=42,
    capture_step=48,
    image_sha256=IMAGE_SHA,
    state_sha256=STATE_SHA,
    gpu_uuid=GPU,
    model_sha256=MODEL_SHA,
    max_model_initializations=1,
    max_model_requests=1,
    denoise_steps=35,
    physics_steps=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)


def validate_capture(result):
    """Accept only the independently reproduced, complete current observation."""
    capture = result["capture"]
    endpoint = capture["current_endpoint"]
    expected = {"endpoint_step": 48, "sim_time": endpoint["sim_time"], "sim_state_sha256": STATE_SHA}
    binding = {"sim_time": endpoint["sim_time"], "sim_state_sha256": STATE_SHA}
    counts = result["counts"]
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False
        and counts["control_steps"] == 48
        and counts["native_steps"] == 1200
        and counts["generated_reference_steps"] == 40
        and counts["stop"] is None
        and all(value == 0 for value in result["independent_max_absolute_difference"].values())
        and capture["frame_index"] == 48
        and endpoint == expected
        and capture["render_before"] == capture["render_after"] == binding
        and capture["future_rgb_frames"] == 0
        and capture["canvas_png"]["sha256"] == IMAGE_SHA
        and capture["camera_roles"] == ["auxiliary_0", "left_wrist"]
        and capture["canvas_shape"] == [960, 640, 3],
        "A complete independently replayed step-48 image and state are required",
    )
    return {"step": 48, "sim_time_s": endpoint["sim_time"], "state_sha256": STATE_SHA}


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_reset_live_second_window.py",
        ROOT / "scripts/wam/tracking/preview/reset_live_request.py",
        ROOT / "scripts/wam/tracking/preview/reset_multitask_live_requests.py",
        ROOT / "scripts/wam/tracking/preview/live_cv2_admission.py",
        ROOT / "scripts/wam/tracking/preview/gpu_resume_repro_20261111.py",
        ROOT / "scripts/wam/tracking/preview/ordered_feedback_wam_inference_20261111.py",
        ROOT / "scripts/wam/tracking/preview/run_causal_stream_live_wam.py",
        ROOT / "scripts/wam/readout/run_wam_readout_features.py",
        ROOT / "src/s2a/backends/wam/runtime.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/causal_stream_protocol.py",
        ROOT / "src/s2a/infrastructure/lifecycle/owned_socket_pair.py",
        ROOT / "tests/research/wam/test_reset_bowl_h40_reobserve_model.py",
    )
    return [prefix.reference(path) for path in paths]


def load_inputs():
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact image, model, device and budgets are required",
    )
    capture_result = prefix.read(CAPTURE)
    initial = validate_capture(capture_result)
    capture = capture_result["capture"]
    prefix.checked(capture["canvas_png"])
    prepared, bridge, _completed, template, _response = engine.previous_inputs()
    model_config = prepared["model_config"]
    cosmos = Path(model_config["cosmos_source_root"]).resolve()
    prefix.require(
        cosmos == Path(bridge["config"]["cosmos_source_root"]).resolve()
        and cosmos.is_relative_to(ROOT / "outputs")
        and Path(model_config["model_python"]).is_file()
        and model_config["checkpoint"]["name"] == "8400_masked"
        and model_config["checkpoint"]["iteration"] == "iter_000008400",
        "The registered model Python, source and checkpoint are required",
    )
    sources = {path: digest for path, digest in bridge["source_sha256"].items() if Path(path).is_relative_to(cosmos)}
    weights = {path: detail["sha256"] for path, detail in prepared["weight_files"].items()}
    fingerprint = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prefix.require(len(sources) >= 100 and len(weights) == 163 and fingerprint == MODEL_SHA, "Model manifests differ")
    return dict(
        config=config,
        capture=capture,
        initial=initial,
        prepared=prepared,
        bridge=bridge,
        template=template,
        model_source_hashes=sources,
        weight_hashes=weights,
    )


def input_receipt(inputs):
    return dict(
        config=prefix.reference(CONFIG),
        plan=prefix.reference(PLAN),
        capture_result=CAPTURE,
        image=inputs["capture"]["canvas_png"],
        initial=inputs["initial"],
        model_sha256=MODEL_SHA,
        original_model_config=prefix.reference(repro.CONFIG),
    )


def model_registration(inputs):
    return dict(
        session=SESSION,
        initial=inputs["initial"],
        image_root=str((OUTPUT / "producer/simulation/runtime").relative_to(ROOT)),
        task=TASK,
        seed=42,
        model_sha256=MODEL_SHA,
        max_requests=1,
        denoise_steps=35,
    )


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                model_initializations=0,
                model_requests=0,
                physics_steps=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["model_initializations"] == dry["model_requests"] == dry["physics_steps"] == 0,
        "A source-matched 32-worker regression and zero-model dry are required",
    )
    return before


def gpu_observation():
    prefix.require(engine.GPU == GPU, "The observation must query the registered GPU UUID")
    return engine.gpu_observation()


def preflight():
    prefix.environment()
    inputs = load_inputs()
    before = check_qa_and_dry(inputs)
    sources = source_refs()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = dict(complete=False, config=prefix.reference(CONFIG), pid=os.getpid())
    publish(OUTPUT / "preflight_started.json", started)
    try:
        gpu = gpu_observation()
        prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "The registered GPU must be idle")
        weights = inputs["weight_hashes"]
        before_stats = model_owner.file_stats(weights)
        repro.verify_manifest(inputs["model_source_hashes"], workers=32)
        repro.verify_manifest(weights, workers=32)
        after_stats = model_owner.file_stats(weights)
        prefix.require(before_stats == after_stats, "Checkpoint files changed during verification")
        probe_dir = OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
        probe_ref = publish(probe_dir / "result.json", probe)
        prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Preflight sources changed")
        result = dict(
            complete=True,
            passed=True,
            inputs=input_receipt(inputs),
            source_refs=sources,
            qa_ref=prefix.reference(QA),
            dry_ref=prefix.reference(DRY),
            gpu_uuid=GPU,
            gpu_before=gpu,
            cv2_probe_ref=probe_ref,
            source_files_verified=len(inputs["model_source_hashes"]),
            weight_files_verified=len(weights),
            weight_stats=after_stats,
            model_python_sha256=repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
            model_initializations=0,
            model_requests=0,
            physics_steps=0,
            automatic_retry=False,
        )
        print(publish(OUTPUT / "preflight.json", result), flush=True)
    except BaseException as error:
        publish(
            OUTPUT / "preflight_failed.json",
            {**started, "error": {"type": type(error).__name__, "message": str(error)}},
        )
        raise


def check_preflight(inputs):
    value = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    prefix.require(
        value["complete"] is value["passed"] is True
        and value["inputs"] == input_receipt(inputs)
        and value["source_refs"] == source_refs()
        and value["qa_ref"] == prefix.reference(QA)
        and value["dry_ref"] == prefix.reference(DRY)
        and value["gpu_uuid"] == GPU
        and value["source_files_verified"] == len(inputs["model_source_hashes"])
        and value["weight_files_verified"] == len(inputs["weight_hashes"]) == 163
        and value["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"])
        and value["weight_stats"] == model_owner.file_stats(inputs["weight_hashes"]),
        "A matching complete model preflight is required",
    )
    return value


def registered_bundle(inputs):
    return dict(
        input_receipt=input_receipt(inputs),
        prepared=inputs["prepared"],
        bridge=inputs["bridge"],
        template=inputs["template"],
        weight_hashes=inputs["weight_hashes"],
        model_registration=model_registration(inputs),
    )


def model_worker_inputs():
    prefix.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and all(os.environ.get(key) == "1" for key in prefix.THREAD_VARS)
        and Path.cwd().resolve() == ROOT,
        "The model worker requires the registered device and environment",
    )
    registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
    bundle = prefix.read(registration["model_bundle_ref"])
    preflight_result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    expected_weights = {path: detail["sha256"] for path, detail in bundle["prepared"]["weight_files"].items()}
    prefix.require(
        registration["inputs"] == bundle["input_receipt"] == preflight_result["inputs"]
        and registration["source_refs"] == preflight_result["source_refs"] == source_refs()
        and preflight_result["complete"] is preflight_result["passed"] is True
        and preflight_result["gpu_uuid"] == GPU
        and preflight_result["qa_ref"] == prefix.reference(QA)
        and preflight_result["dry_ref"] == prefix.reference(DRY)
        and preflight_result["weight_files_verified"] == len(expected_weights) == 163
        and bundle["weight_hashes"] == expected_weights
        and preflight_result["weight_stats"] == model_owner.file_stats(expected_weights)
        and preflight_result["model_python_sha256"] == repro.sha256(bundle["prepared"]["model_config"]["model_python"]),
        "The model worker requires the exact preflight and registered bundle",
    )
    return bundle, preflight_result


def model_worker():
    from scripts.wam.tracking.preview import reset_live_request

    original = reset_live_request.serve_one

    def serve_task(**kwargs):
        return original(
            **kwargs, request_builder=lambda template, image: tasks.make_task_request(template, image, TASK)
        )

    reset_live_request.serve_one = serve_task
    live.OUTPUT, live.GPU, live.model_worker_inputs = OUTPUT, GPU, model_worker_inputs
    live.model_worker()


def request_worker():
    directory = OUTPUT / "producer/simulation"
    result = dict(
        complete=False,
        passed=False,
        evidence_kind="offline_saved_capture",
        physics_steps=0,
        automatic_retry=False,
    )
    channel = None
    try:
        prefix.environment()
        inputs = load_inputs()
        registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
        bundle = prefix.read(registration["model_bundle_ref"])
        prefix.require(
            registration["inputs"] == input_receipt(inputs) == bundle["input_receipt"]
            and registration["source_refs"] == source_refs()
            and bundle["model_registration"] == model_registration(inputs),
            "The request must use the registered capture, task and model",
        )
        ledger = CausalStreamLedger(**bundle["model_registration"])
        image = dict(path="anchor0048/current.png", sha256=IMAGE_SHA)
        current = inputs["initial"]
        request = ledger.issue(before_render=current, after_render=current, image=image)
        result["request_ref"] = publish(directory / "request.json", request)
        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
        )
        channel.send(request)
        response = channel.receive()
        ledger.accept(response, after_wait=current)
        result["response_ref"] = publish(directory / "response.json", response)
        prefix.require(ledger.accepted_requests == 1 and ledger.phase == "executing", "One response must be accepted")
        result.update(complete=True, passed=True, requests_sent=1, responses_accepted=1)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if channel is not None:
            channel.close()
        print(publish(directory / "result.json", result), flush=True)


def execute():
    prefix.environment()
    inputs = load_inputs()
    before = check_qa_and_dry(inputs)
    check_preflight(inputs)
    sources = source_refs()
    directory = OUTPUT / "producer"
    directory.mkdir(exist_ok=False)
    simulation, model = directory / "simulation", directory / "model"
    simulation.mkdir()
    model.mkdir()
    image_dir = simulation / "runtime/anchor0048"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "current.png"
    with image_path.open("xb") as destination, Path(inputs["capture"]["canvas_png"]["path"]).open("rb") as origin:
        shutil.copyfileobj(origin, destination)
    prefix.require(repro.sha256(image_path) == IMAGE_SHA, "The staged image must match the verified capture")
    bundle_ref = publish(model / "registered_inputs.json", registered_bundle(inputs))
    registration = publish(
        directory / "registration.json",
        dict(inputs=input_receipt(inputs), source_refs=sources, model_bundle_ref=bundle_ref),
    )
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        evidence_kind="offline_saved_capture",
        registration_ref=registration,
        gpu_uuid=GPU,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        result["gpu_before"] = gpu_observation()
        prefix.require(
            result["gpu_before"]["memory_mib"] == 0 and result["gpu_before"]["compute_apps"] == [],
            "The registered GPU must be idle before starting",
        )
        simulation_env = physical._cpu_environment(simulation)
        model_env = model_owner._model_environment(model)
        model_env["CUDA_VISIBLE_DEVICES"] = GPU
        module = "scripts.simulation.runtime.run_reset_bowl_h40_reobserve_model"
        specs = {
            "simulation": dict(
                argv=[sys.executable, "-m", module, "--worker", "request"],
                env=simulation_env,
                log=str((directory / "request.log").relative_to(ROOT)),
            ),
            "model": dict(
                argv=[inputs["prepared"]["model_config"]["model_python"], "-m", module, "--worker", "model"],
                env=model_env,
                log=str((directory / "model.log").relative_to(ROOT)),
            ),
        }
        try:
            lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        except OwnedSocketPairInterrupted as error:
            result["workers_ref"] = publish(directory / "workers.json", error.result)
            raise
        result["workers_ref"] = publish(directory / "workers.json", lifecycle)
        prefix.require(lifecycle["passed"] is True, "Both workers must exit and be reaped")
        model_ref = prefix.reference(model / "result.json")
        request_ref = prefix.reference(simulation / "result.json")
        model_result, request_result = prefix.read(model_ref), prefix.read(request_ref)
        response = prefix.read(request_result["response_ref"])
        raw = prefix.read(model_result["responses"][0]["actual_response_ref"])
        arrays = repro.response_arrays(raw)
        np.testing.assert_array_equal(arrays["action"], response["action"])
        np.testing.assert_array_equal(arrays["normalized_action"], response["normalized_action"])
        result["gpu_after"] = gpu_observation()
        result["gpu_released"] = result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
        prefix.require(
            model_result["complete"]
            is model_result["passed"]
            is request_result["complete"]
            is request_result["passed"]
            is True
            and model_result["model_initializations"] == model_result["model_requests_completed"] == 1
            and model_result["denoise_calls_completed"] == 35
            and request_result["requests_sent"] == request_result["responses_accepted"] == 1
            and result["gpu_released"] is True
            and snapshot(ROOT)["sources"] == before
            and source_refs() == sources,
            "The single real response, cleanup and sources must match registration",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            model_ref=model_ref,
            request_ref=request_ref,
            response_ref=request_result["response_ref"],
            raw_response_ref=model_result["responses"][0]["actual_response_ref"],
            model_initializations=model_result["model_initializations"],
            model_requests_attempted=model_result["model_requests_attempted"],
            model_requests_completed=model_result["model_requests_completed"],
            denoise_calls_attempted=model_result["denoise_calls_attempted"],
            denoise_calls_completed=model_result["denoise_calls_completed"],
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        model_path = model / "result.json"
        if model_path.is_file():
            result["available_model_ref"] = prefix.reference(model_path)
            available = json.loads(model_path.read_text())
            for key in (
                "model_initialization_attempts",
                "model_initializations",
                "model_requests_attempted",
                "model_requests_completed",
                "denoise_calls_attempted",
                "denoise_calls_completed",
            ):
                result[key] = available.get(key)
        request_path = simulation / "result.json"
        if request_path.is_file():
            result["available_request_ref"] = prefix.reference(request_path)
        if "gpu_after" not in result:
            try:
                result["gpu_after"] = gpu_observation()
                result["gpu_released"] = (
                    result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
                )
            except Exception as error:
                result["gpu_after_error"] = {"type": type(error).__name__, "message": str(error)}
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", choices=("request", "model"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.preflight:
        preflight()
    elif args.execute:
        execute()
    elif args.worker == "request":
        request_worker()
    else:
        model_worker()


if __name__ == "__main__":
    main()
