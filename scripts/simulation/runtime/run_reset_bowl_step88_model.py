"""Reuse the audited single-image model owner for the verified step-88 image."""

import copy
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_bowl_h40_reobserve_model as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_step88_model_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_step88_model_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_step88_model_20261112"
QA = ROOT / "outputs/qa/reset_bowl_step88_model_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_step88_model_dry_20261112/result.json"
CAPTURE = {
    "path": str(ROOT / "outputs/reset_bowl_step88_capture_20261112/result.json"),
    "sha256": "8a8175014b84494291498e9c8b706b9450d0ee5fabef320b6d1014a28c41553c",
}
IMAGE_SHA = "e3418ff98474b44c3d139772d3e9be9baee581686ec29934c1a9c82e05bb4a4f"
STATE_SHA = "64f5fd441e30498b43753f91725ddefe625c016c1253cd5f99fceebf844ea789"
SESSION = "reset_bowl_h40_step88_offline_s42_20261112"
MODULE = "scripts.simulation.runtime.run_reset_bowl_step88_model"
OLD_MODULE = "scripts.simulation.runtime.run_reset_bowl_h40_reobserve_model"
OLD_SOURCE_SHA = "59a27d66f7058941d067fd17838af725f40c454396c4455d00fb5bee0d3e8a88"
FIXED = dict(
    schema="s2a.reset-bowl-step88-model.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    task=previous.TASK,
    seed=42,
    capture_step=88,
    image_sha256=IMAGE_SHA,
    state_sha256=STATE_SHA,
    gpu_uuid=previous.GPU,
    model_sha256=previous.MODEL_SHA,
    max_model_initializations=1,
    max_model_requests=1,
    denoise_steps=35,
    physics_steps=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)
_OLD_SOURCES = previous.source_refs
_OWNED_SOCKET_PAIR = previous.run_owned_socket_pair


def validate_capture(result):
    capture = result["terminal_capture"]["capture"]
    snapshot = result["terminal_capture"]["snapshot"]
    endpoint = capture["current_endpoint"]
    binding = {"sim_time": snapshot["sim_time_s"], "sim_state_sha256": STATE_SHA}
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False
        and result["counts"]["control_steps"] == 88
        and result["counts"]["native_steps"] == 2200
        and result["counts"]["stop"] is None
        and all(value == 0 for value in result["independent_max_absolute_difference"].values())
        and snapshot["step"] == capture["frame_index"] == endpoint["endpoint_step"] == 88
        and snapshot["state_sha256"] == endpoint["sim_state_sha256"] == STATE_SHA
        and endpoint["sim_time"] == snapshot["sim_time_s"]
        and capture["render_before"] == capture["render_after"] == binding
        and capture["future_rgb_frames"] == 0
        and capture["canvas_png"]["sha256"] == IMAGE_SHA
        and capture["camera_roles"] == ["auxiliary_0", "left_wrist"]
        and capture["canvas_shape"] == [960, 640, 3],
        "A complete independently replayed step-88 image and state are required",
    )
    return dict(snapshot), capture


def load_inputs():
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact image, model, device and budgets are required",
    )
    initial, capture = validate_capture(prefix.read(CAPTURE))
    prefix.checked(capture["canvas_png"])
    prepared, bridge, _completed, template, _response = previous.engine.previous_inputs()
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
    prefix.require(
        len(sources) >= 100 and len(weights) == 163 and fingerprint == previous.MODEL_SHA, "Model manifests differ"
    )
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


def source_refs():
    paths = (
        Path(__file__).resolve(),
        ROOT / "tests/research/wam/test_reset_bowl_step88_model.py",
        ROOT / "scripts/simulation/runtime/run_reset_bowl_step88_capture.py",
        ROOT / "src/s2a/research/native_runtime/verified_terminal_capture.py",
    )
    return _OLD_SOURCES() + [prefix.reference(path) for path in paths]


def stage_current_image(runtime_root):
    """Give the verified copied image a step-neutral request path."""
    runtime_root = Path(runtime_root).resolve()
    prefix.require(
        runtime_root.is_relative_to(ROOT / "outputs") and runtime_root.is_dir(), "Local runtime root required"
    )
    staged, current = runtime_root / "anchor0048/current.png", runtime_root / "current.png"
    prefix.require(previous.repro.sha256(staged) == IMAGE_SHA, "The compatibility staging image differs")
    with staged.open("rb") as origin, current.open("xb") as destination:
        shutil.copyfileobj(origin, destination)
    prefix.require(
        previous.repro.sha256(staged) == previous.repro.sha256(current) == IMAGE_SHA,
        "The model input must match the verified current image",
    )
    return dict(path="current.png", sha256=IMAGE_SHA)


def request_worker():
    directory = OUTPUT / "producer/simulation"
    result = dict(
        complete=False, passed=False, evidence_kind="offline_saved_capture", physics_steps=0, automatic_retry=False
    )
    channel = None
    try:
        prefix.environment()
        inputs = load_inputs()
        registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
        bundle = prefix.read(registration["model_bundle_ref"])
        prefix.require(
            registration["inputs"] == previous.input_receipt(inputs) == bundle["input_receipt"]
            and registration["source_refs"] == source_refs()
            and bundle["model_registration"] == previous.model_registration(inputs),
            "The request must use the registered capture, task and model",
        )
        ledger = CausalStreamLedger(**bundle["model_registration"])
        image = stage_current_image(OUTPUT / "producer/simulation/runtime")
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


def redirect_workers(specs, *, timeout_seconds):
    """Run the existing owner with this adapter in both fresh worker processes."""
    specs = copy.deepcopy(specs)
    prefix.require(type(specs) is dict and set(specs) == {"simulation", "model"}, "Exactly two workers are required")
    for spec in specs.values():
        prefix.require(
            type(spec.get("argv")) is list and spec["argv"][1:3] == ["-m", OLD_MODULE],
            "Only the registered model-owner workers may be redirected",
        )
        spec["argv"][2] = MODULE
    return _OWNED_SOCKET_PAIR(specs, timeout_seconds=timeout_seconds)


def bind():
    prefix.require(previous.repro.sha256(previous.__file__) == OLD_SOURCE_SHA, "The original model owner changed")
    previous.CONFIG, previous.PLAN, previous.OUTPUT, previous.QA, previous.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    previous.CAPTURE, previous.IMAGE_SHA, previous.STATE_SHA, previous.SESSION, previous.FIXED = (
        CAPTURE,
        IMAGE_SHA,
        STATE_SHA,
        SESSION,
        FIXED,
    )
    previous.load_inputs, previous.source_refs = load_inputs, source_refs
    previous.request_worker, previous.run_owned_socket_pair = request_worker, redirect_workers
    return previous


def main():
    bind().main()


if __name__ == "__main__":
    main()
