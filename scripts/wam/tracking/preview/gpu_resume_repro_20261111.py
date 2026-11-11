"""One registered, isolated raw WAM reproduction after GPU access resumed."""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "configs/tracking/gpu_resume_repro_20261111.json"
OUTPUT = ROOT / "outputs/gpu_resume_repro_20261111"
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"
EXPECTED_KEYS = {
    "name",
    "model_mode",
    "prompt",
    "vision_path",
    "domain_name",
    "image_size",
    "action_chunk_size",
    "fps",
    "num_frames",
    "seed",
    "num_steps",
    "guidance",
    "shift",
    "extra",
}
EXTRA_KEYS = {
    "action_domain",
    "action_dim_mask",
    "action_fps",
    "task_title",
    "selected_roles",
    "camera_layout",
    "return_video",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checked(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise ValueError(f"SHA256 mismatch: {path}")
    return json.loads(path.read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_config(config):
    if (
        config["schema"] != "s2a.gpu-resume-repro.v1"
        or ROOT / config["output"] != OUTPUT
        or config["gpu_uuid"] != GPU
        or GPU not in config["visible_authorized_uuids"]
        or [
            config[k]
            for k in ("model_initializations", "model_requests", "denoise_calls", "physics_steps", "fresh_cases")
        ]
        != [1, 1, 35, 0, 0]
        or config["automatic_retry"] is not False
    ):
        raise ValueError("Registered GPU, path or budget differs")


def validate_request(request, image_ref):
    if set(request) != EXPECTED_KEYS or set(request["extra"]) != {"action128"}:
        raise ValueError("Unexpected request conditioning")
    extra = request["extra"]["action128"]
    if (
        set(extra) != EXTRA_KEYS
        or request["name"] != "s2a_task0_init0_s42_0"
        or request["prompt"] != "put the bowl on the plate"
        or request["domain_name"] != "franka_panda__parallel_gripper"
        or request["model_mode"] != "wam"
        or [request[k] for k in ("image_size", "action_chunk_size", "fps", "num_frames", "seed", "num_steps")]
        != [960, 40, 20, 41, 42, 35]
        or request["guidance"] != 1.0
        or request["shift"] != 5.0
        or extra["action_domain"]
        != "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
        or extra["action_dim_mask"] != [True] * 10 + [False] * 118
        or any(type(value) is not bool for value in extra["action_dim_mask"])
        or extra["action_fps"] != 20
        or extra["task_title"] != request["prompt"]
        or extra["return_video"] is not False
        or request["vision_path"] != image_ref["path"]
        or not Path(image_ref["path"]).resolve().is_relative_to(ROOT / "outputs")
    ):
        raise ValueError("Request differs from fixed development H40 input")


def response_arrays(response):
    import numpy as np

    if response["status"] != "success" or len(response["outputs"]) != 1:
        raise ValueError("Model response was not a single success")
    content = response["outputs"][0]["content"]
    arrays = {
        "action": np.asarray(content["action"], dtype=np.float64),
        "normalized_action": np.asarray(content["action128"]["normalized_action"], dtype=np.float64),
    }
    if any(a.shape != (40, 128) or not np.isfinite(a).all() or np.any(a[:, 10:]) for a in arrays.values()):
        raise ValueError("Model output must be finite H40 with 118 inactive slots")
    return arrays


def compare_responses(actual, expected):
    import numpy as np

    left, right = response_arrays(actual), response_arrays(expected)
    comparison = {}
    for key in left:
        difference = left[key] - right[key]
        comparison[key] = {
            "elementwise_equal": bool(np.array_equal(left[key], right[key])),
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "different_elements": int(np.count_nonzero(difference)),
        }
    return comparison


def inputs(config):
    prepared = read_checked(ROOT / config["prepared"], config["prepared_sha256"])
    completed = read_checked(ROOT / config["historical_completed"], config["historical_completed_sha256"])
    if completed["complete"] is not True or completed["variant"] != "raw" or completed["readout_ref"] is not None:
        raise ValueError("Historical raw request was not completed")
    for key, expected in (("request_ref", "request_sha256"), ("response_ref", "response_sha256")):
        if completed[key]["sha256"] != config[expected]:
            raise ValueError("Historical request/response binding changed")
    if completed["image_ref"]["sha256"] != config["image_sha256"]:
        raise ValueError("Historical image binding changed")
    request = read_checked(**{"path": completed["request_ref"]["path"], "digest": config["request_sha256"]})
    response = read_checked(**{"path": completed["response_ref"]["path"], "digest": config["response_sha256"]})
    validate_request(request, completed["image_ref"])
    if sha256(completed["image_ref"]["path"]) != config["image_sha256"]:
        raise ValueError("Historical image changed")
    bridge_ref = prepared["prerequisite_refs"]["bridge_prepared"]
    bridge = read_checked(bridge_ref["path"], bridge_ref["sha256"])
    return prepared, bridge, completed, request, response


def verify_manifest(expected, *, workers=12):
    def verify(item):
        path, digest = item
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"Manifest SHA256 mismatch: {path}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(verify, expected.items()))


def preflight(config):
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new(OUTPUT / "preflight_started.json", {"config_sha256": sha256(CONFIG), "pid": os.getpid()})
    try:
        prepared, bridge, completed, request, response = inputs(config)
        response_arrays(response)
        cosmos = Path(prepared["model_config"]["cosmos_source_root"]).resolve()
        if cosmos != Path(bridge["config"]["cosmos_source_root"]).resolve() or not cosmos.is_relative_to(
            ROOT / "outputs"
        ):
            raise ValueError("Frozen Cosmos source root differs")
        sources = {
            path: digest for path, digest in bridge["source_sha256"].items() if Path(path).is_relative_to(cosmos)
        }
        weights = {path: entry["sha256"] for path, entry in prepared["weight_files"].items()}
        if len(sources) < 100 or len(weights) != 163:
            raise ValueError("Incomplete source or checkpoint manifest")
        verify_manifest(sources)
        verify_manifest(weights)
        local_sources = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/s2a/backends/wam/runtime.py",
                ROOT / "scripts/wam/readout/run_wam_readout_features.py",
            )
        }
        result = {
            "complete": True,
            "config_sha256": sha256(CONFIG),
            "gpu_uuid": GPU,
            "source_files_verified": len(sources),
            "weight_files_verified": len(weights),
            "weight_bytes": sum(entry["bytes"] for entry in prepared["weight_files"].values()),
            "local_sources": local_sources,
            "request_sha256": config["request_sha256"],
            "response_sha256": config["response_sha256"],
            "image_sha256": config["image_sha256"],
            "model_initializations": 0,
            "model_requests": 0,
            "physics_steps": 0,
        }
        write_new(OUTPUT / "preflight.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write_new(OUTPUT / "preflight_failed.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


def run(config):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("Worker must expose exactly its registered GPU UUID and empty LD")
    preflight_result = json.loads((OUTPUT / "preflight.json").read_text())
    if preflight_result["complete"] is not True or preflight_result["config_sha256"] != sha256(CONFIG):
        raise ValueError("Preflight is missing or its configuration changed")
    for path, digest in preflight_result["local_sources"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Local source changed after preflight: {path}")
    prepared, bridge, _completed, request, expected = inputs(config)
    model_config = prepared["model_config"]
    if Path(sys.executable).resolve() != Path(model_config["model_python"]).resolve():
        raise ValueError("Unexpected model Python")
    if any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules):
        raise ValueError("Cosmos was imported before source admission")
    cosmos = Path(model_config["cosmos_source_root"]).resolve()
    sys.path.insert(0, str(cosmos))
    directory = OUTPUT / "model_run"
    directory.mkdir(exist_ok=False)
    result = {"complete": False, "gpu_uuid": GPU, "model_initialization_attempts": 0, "model_requests": 0}
    write_new(directory / "started.json", {**result, "pid": os.getpid()})
    try:
        import numpy as np
        import torch

        from s2a.backends.wam.runtime import install_active_slot_mask, load_service
        from scripts.verification.wam.model_contracts.audit_wam_current_numerics import require_eval
        from scripts.wam.readout.run_wam_readout_features import BoundaryCounter, actual_context
        from scripts.wam.training.clean.qualify_wam_clean_bridge import imported_cosmos_sources

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one actual CUDA device is required")
        device = torch.cuda.get_device_properties(0)
        if str(device.uuid).removeprefix("GPU-") != GPU.removeprefix("GPU-"):
            raise ValueError("Actual device UUID differs")
        result.update(torch_version=torch.__version__, compiled_cuda=torch.version.cuda, device_name=device.name)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        (directory / "model").mkdir()
        result["model_initialization_attempts"] = 1
        checkpoint = model_config["checkpoint"]
        service = load_service({"checkpoints": [checkpoint]}, "8400_masked", directory / "model")
        result["model_initializations"] = 1
        expected_checkpoint = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
        if (
            Path(service.info["checkpoint_path"]).resolve() != expected_checkpoint
            or service.info["ema_weights"] is not False
        ):
            raise ValueError("Loaded checkpoint/EMA differs")
        model = service.model
        require_eval(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        evidence = install_active_slot_mask(model, [True] * 10 + [False] * 118)
        counter = BoundaryCounter(model, 35)
        imported_cosmos_sources(bridge)
        from cosmos_framework.inference.args import OmniSampleOverrides

        sample, batch = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(request)), directory)
        context = actual_context(service, sample, batch)
        caption = json.loads(batch["ai_caption"][0])
        if (
            not context.is_libero
            or caption["task_title"] != request["prompt"]
            or caption["command"] != request["prompt"]
            or caption["task_plan"] != []
        ):
            raise ValueError("Actual conditioning is not current-image and task-only")
        result["model_requests"] = 1
        with torch.inference_mode(), counter.scope(request["name"], 35):
            response = service.execute(sample, batch).model_dump(mode="json")
        write_new(directory / "response.json", response)
        result["comparison"] = compare_responses(response, expected)
        if len(counter.calls) != 35 or not all(call["completed"] for call in counter.calls) or len(evidence) != 1:
            raise ValueError("Actual denoise or mask evidence differs")
        result.update(
            complete=True,
            denoise_calls=len(counter.calls),
            mask_evidence=evidence,
            imported_cosmos_sources=imported_cosmos_sources(bridge),
            response_sha256=sha256(directory / "response.json"),
            physics_steps=0,
            fresh_cases=0,
            automatic_retry=False,
        )
        write_new(directory / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "imported_cosmos_sources"}), flush=True)
    except BaseException as error:
        write_new(directory / "failed.json", {**result, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if "torch" in locals() and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    if args.mode == "preflight":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
            raise ValueError("Preflight is CPU-only with empty CUDA and LD")
        preflight(config)
    else:
        run(config)


if __name__ == "__main__":
    main()
