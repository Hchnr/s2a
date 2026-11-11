"""Run two bounded raw WAM requests on verified current feedback observations."""

import argparse
import copy
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = repro.ROOT
CONFIG = ROOT / "configs/tracking/references/ordered_feedback_wam_inference_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/ordered_feedback_wam_inference_20261111.md"
OUTPUT = ROOT / "outputs/ordered_feedback_wam_observation_v2_20261111/inference"
QA = ROOT / "outputs/qa/ordered_feedback_wam_inference_v3_20261111/result.json"
GPU = repro.GPU
TASK = "put the cream cheese in the bowl"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    fixed = {
        "schema": "s2a.ordered-feedback-wam-inference.v1",
        "output": str(OUTPUT.relative_to(ROOT)),
        "task": TASK,
        "gpu_uuid": GPU,
        "seed": 42,
        "horizon": 40,
        "denoise_steps_per_request": 35,
        "model_initializations": 1,
        "model_requests": 2,
        "denoise_calls": 70,
        "physics_steps": 0,
        "fresh_cases": 0,
        "fits": 0,
        "automatic_retry": False,
    }
    require(
        all(config.get(k) == v and type(config.get(k)) is type(v) for k, v in fixed.items())
        and [image["step"] for image in config["images"]] == [80, 88],
        "Fixed two-request inference budget required",
    )
    for image in config["images"]:
        path = (ROOT / image["path"]).resolve()
        require(
            path.is_relative_to(ROOT / "outputs/ordered_feedback_wam_observation_v2_20261111/producer")
            and path.name == "current.png"
            and len(image["sha256"]) == 64,
            "Registered current image required",
        )


def read_ref(config, name):
    return repro.read_checked(ROOT / config[name], config[name + "_sha256"])


def capture_inputs(config):
    summary = read_ref(config, "capture_summary")
    branch = read_ref(config, "capture_branch")
    qa = read_ref(config, "capture_qa")
    prior_preflight = read_ref(config, "model_preflight")
    expected_branch = {"path": str(ROOT / config["capture_branch"]), "sha256": config["capture_branch_sha256"]}
    counts = summary["counts"]
    require(
        summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
        and summary["branch_refs"] == [expected_branch]
        and counts["control_steps"] == 88
        and counts["native_steps"] == 2200
        and counts["continuation_steps"] == 8
        and counts["captured_steps"] == [80, 88]
        and counts["newly_selected_commands"] == 0
        and branch["complete"] is branch["passed"] is True
        and branch["runtime"]["cleanup_complete"] is True
        and qa["complete"] is qa["passed"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["skipped_count"] == 0
        and prior_preflight["complete"] is True
        and prior_preflight["gpu_uuid"] == GPU,
        "Completed capture, source checks, and previous model preflight required",
    )
    worker_ref = summary["workers_ref"]
    workers = repro.read_checked(worker_ref["path"], worker_ref["sha256"])["workers"]
    require(
        len(workers) == 1
        and workers[0]["returncode"] == 0
        and workers[0]["actual_wait"] is workers[0]["reaped"] is True
        and workers[0]["signal_sent"] is False,
        "Capture worker must actually have exited and been reaped",
    )
    captures = branch["runtime"]["captures"]
    require(len(captures) == 2, "Exactly two actual observations required")
    for image, capture in zip(config["images"], captures, strict=True):
        expected = {"path": str(ROOT / image["path"]), "sha256": image["sha256"]}
        binding = capture["render_before"]
        require(
            capture["frame_index"] == image["step"]
            and capture["canvas_png"] == expected
            and capture["canvas_shape"] == [960, 640, 3]
            and capture["camera_roles"] == ["auxiliary_0", "left_wrist"]
            and capture["future_rgb_frames"] == 0
            and binding == capture["render_after"]
            and binding["sim_state_sha256"] == capture["current_endpoint"]["sim_state_sha256"]
            and binding["sim_time"] == capture["current_endpoint"]["sim_time"]
            and repro.sha256(expected["path"]) == expected["sha256"]
            and repro.sha256(capture["raw_cameras"]["path"]) == capture["raw_cameras"]["sha256"],
            "Current image and unchanged physical endpoint must match the capture",
        )
    require(
        abs(captures[1]["render_before"]["sim_time"] - captures[0]["render_before"]["sim_time"] - 0.4) < 1e-9,
        "Capture endpoints must be eight control periods apart",
    )
    return captures


def make_request(template, image):
    request = copy.deepcopy(template)
    request["name"] = f"s2a_correct08_t{image['step']}_s42"
    request["prompt"] = TASK
    request["vision_path"] = str(ROOT / image["path"])
    request["extra"]["action128"]["task_title"] = TASK
    validate_request(request, image)
    return request


def validate_request(request, image):
    require(
        request.get("name") == f"s2a_correct08_t{image['step']}_s42"
        and request.get("prompt") == TASK
        and request.get("vision_path") == str(ROOT / image["path"])
        and request.get("extra", {}).get("action128", {}).get("task_title") == TASK,
        "Current-image and fixed task conditioning required",
    )
    extra = request["extra"]["action128"]
    require(
        extra.get("selected_roles") == ["auxiliary_0", "left_wrist"]
        and extra.get("camera_layout") == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
        "Original dual-camera layout required",
    )
    normalized = copy.deepcopy(request)
    normalized["name"] = "s2a_task0_init0_s42_0"
    normalized["prompt"] = normalized["extra"]["action128"]["task_title"] = "put the bowl on the plate"
    repro.validate_request(normalized, {"path": str(ROOT / image["path"])})


def previous_inputs():
    old_config = json.loads(repro.CONFIG.read_text())
    repro.validate_config(old_config)
    return repro.inputs(old_config)


def local_sources():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        Path(repro.__file__).resolve(),
        ROOT / "src/s2a/backends/wam/runtime.py",
        ROOT / "scripts/wam/readout/run_wam_readout_features.py",
        ROOT / "scripts/wam/training/clean/qualify_wam_clean_bridge.py",
        ROOT / "tests/research/wam/test_ordered_feedback_wam_inference.py",
    )
    return {str(path.relative_to(ROOT)): repro.sha256(path) for path in paths}


def check_cpu():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("LD_LIBRARY_PATH") == "",
        "CPU owner must have empty CUDA and LD",
    )
    require(Path.cwd().resolve() == ROOT, "Repository working directory required")


def validate_qa_snapshot(snapshot, sources):
    require(
        all(snapshot["sources"].get(path) == digest for path, digest in sources.items()),
        "CPU regression must have covered the current inference sources",
    )


def validate_mask_evidence(evidence, request_count):
    require(
        len(evidence) == request_count
        and all(
            row.get("stage") == "initial_noise"
            and isinstance(row.get("inactive_rms_before"), (int, float))
            and math.isfinite(row["inactive_rms_before"])
            and row["inactive_rms_before"] >= 0
            and row.get("inactive_rms_after") == 0.0
            for row in evidence
        ),
        "One valid initial-noise mask record per completed request is required",
    )


def preflight(config):
    check_cpu()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    repro.write_new(OUTPUT / "preflight_started.json", {"pid": os.getpid(), "config_sha256": repro.sha256(CONFIG)})
    try:
        captures = capture_inputs(config)
        qa = json.loads(QA.read_text())
        require(
            qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
            and qa["skipped_count"] == 0,
            "Passing inference CPU regression and reaped workers required",
        )
        snapshot_ref = qa["source_before_ref"]
        snapshot = repro.read_checked(snapshot_ref["path"], snapshot_ref["sha256"])
        validate_qa_snapshot(snapshot, local_sources())
        prepared, bridge, _completed, template, _response = previous_inputs()
        requests = [make_request(template, image) for image in config["images"]]
        cosmos = Path(prepared["model_config"]["cosmos_source_root"]).resolve()
        require(
            cosmos == Path(bridge["config"]["cosmos_source_root"]).resolve()
            and cosmos.is_relative_to(ROOT / "outputs"),
            "Frozen Cosmos source required",
        )
        sources = {p: digest for p, digest in bridge["source_sha256"].items() if Path(p).is_relative_to(cosmos)}
        weights = {p: entry["sha256"] for p, entry in prepared["weight_files"].items()}
        require(len(sources) >= 100 and len(weights) == 163, "Complete model manifests required")
        repro.verify_manifest(sources)
        repro.verify_manifest(weights)
        result = {
            "complete": True,
            "config_sha256": repro.sha256(CONFIG),
            "qa_sha256": repro.sha256(QA),
            "gpu_uuid": GPU,
            "local_sources": local_sources(),
            "source_files_verified": len(sources),
            "weight_files_verified": len(weights),
            "weight_bytes": sum(entry["bytes"] for entry in prepared["weight_files"].values()),
            "images": [capture["canvas_png"] for capture in captures],
            "request_names": [request["name"] for request in requests],
            "model_initializations": 0,
            "model_requests": 0,
            "physics_steps": 0,
        }
        repro.write_new(OUTPUT / "preflight.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        repro.write_new(OUTPUT / "preflight_failed.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


def checked_preflight():
    result = json.loads((OUTPUT / "preflight.json").read_text())
    require(
        result["complete"] is True
        and result["config_sha256"] == repro.sha256(CONFIG)
        and result["qa_sha256"] == repro.sha256(QA)
        and result["local_sources"] == local_sources(),
        "Inference preflight, QA, and source identities must remain unchanged",
    )
    return result


def run(config):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and os.environ.get("LD_LIBRARY_PATH") == "",
        "Worker must expose its registered GPU UUID and empty LD",
    )
    checked_preflight()
    capture_inputs(config)
    prepared, bridge, _completed, template, _response = previous_inputs()
    model_config = prepared["model_config"]
    require(Path(sys.executable).resolve() == Path(model_config["model_python"]).resolve(), "Model Python differs")
    require(
        not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules),
        "Cosmos was imported before source admission",
    )
    sys.path.insert(0, str(Path(model_config["cosmos_source_root"]).resolve()))
    directory = OUTPUT / "model_run"
    directory.mkdir(exist_ok=False)
    result = {"complete": False, "gpu_uuid": GPU, "model_initialization_attempts": 0, "model_requests": 0}
    repro.write_new(directory / "started.json", {**result, "pid": os.getpid()})
    try:
        import numpy as np
        import torch

        from s2a.backends.wam.runtime import install_active_slot_mask, load_service
        from scripts.verification.wam.model_contracts.audit_wam_current_numerics import require_eval
        from scripts.wam.readout.run_wam_readout_features import BoundaryCounter, actual_context
        from scripts.wam.training.clean.qualify_wam_clean_bridge import imported_cosmos_sources

        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device required")
        device = torch.cuda.get_device_properties(0)
        require(str(device.uuid).removeprefix("GPU-") == GPU.removeprefix("GPU-"), "Actual CUDA UUID differs")
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
        require(
            Path(service.info["checkpoint_path"]).resolve() == expected_checkpoint
            and service.info["ema_weights"] is False,
            "Loaded checkpoint or EMA differs",
        )
        model = service.model
        require_eval(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        evidence = install_active_slot_mask(model, [True] * 10 + [False] * 118)
        counter = BoundaryCounter(model, 70)
        imported_cosmos_sources(bridge)
        from cosmos_framework.inference.args import OmniSampleOverrides

        responses = []
        for image in config["images"]:
            request = make_request(template, image)
            require(repro.sha256(ROOT / image["path"]) == image["sha256"], "Current image changed")
            item = directory / f"step{image['step']}"
            item.mkdir()
            repro.write_new(item / "request.json", request)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            started = time.monotonic()
            sample, batch = service.prepare(OmniSampleOverrides.model_validate(copy.deepcopy(request)), item)
            context = actual_context(service, sample, batch)
            caption = json.loads(batch["ai_caption"][0])
            require(
                context.is_libero
                and caption["task_title"] == caption["command"] == TASK
                and caption["task_plan"] == [],
                "Actual conditioning differs from current image and task",
            )
            repro.write_new(item / "conditioning.json", {"caption": caption, "image_sha256": image["sha256"]})
            result["model_requests"] += 1
            with torch.inference_mode(), counter.scope(request["name"], 35):
                response = service.execute(sample, batch).model_dump(mode="json")
            elapsed = time.monotonic() - started
            repro.response_arrays(response)
            repro.write_new(item / "response.json", response)
            require(repro.sha256(ROOT / image["path"]) == image["sha256"], "Current image changed during inference")
            responses.append(
                {
                    "step": image["step"],
                    "response_sha256": repro.sha256(item / "response.json"),
                    "request_sha256": repro.sha256(item / "request.json"),
                    "elapsed_seconds": elapsed,
                }
            )
        require(
            len(counter.calls) == 70 and all(call["completed"] for call in counter.calls),
            "Denoise evidence differs",
        )
        validate_mask_evidence(evidence, len(config["images"]))
        checked_preflight()
        result.update(
            complete=True,
            responses=responses,
            denoise_calls=len(counter.calls),
            denoise_evidence=counter.calls,
            mask_evidence=evidence,
            imported_cosmos_sources=imported_cosmos_sources(bridge),
            physics_steps=0,
            fresh_cases=0,
            automatic_retry=False,
        )
        repro.write_new(directory / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "imported_cosmos_sources"}), flush=True)
    except BaseException as error:
        if "counter" in locals():
            result["denoise_calls_attempted"] = len(counter.calls)
            result["denoise_calls_completed"] = sum(call["completed"] for call in counter.calls)
            result["denoise_evidence"] = counter.calls
        if "evidence" in locals():
            result["mask_evidence"] = evidence
        if "responses" in locals():
            result["completed_responses"] = responses
        repro.write_new(
            directory / "failed.json", {**result, "error": repr(error), "traceback": traceback.format_exc()}
        )
        raise
    finally:
        if "torch" in locals() and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def gpu_observation():
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=20
    )
    devices = [line.strip().split(", ") for line in query.splitlines()]
    selected = [int(memory) for uuid, memory in devices if uuid == GPU]
    require(len(selected) == 1, "Registered GPU is not uniquely visible")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        timeout=20,
    )
    selected_apps = [line.strip() for line in apps.splitlines() if line.strip().startswith(GPU + ",")]
    return {"memory_mib": selected[0], "compute_apps": selected_apps}


def launch(config):
    check_cpu()
    checked_preflight()
    prepared, _bridge, _completed, _template, _response = previous_inputs()
    before = gpu_observation()
    require(before["memory_mib"] == 0 and before["compute_apps"] == [], "Registered GPU is not idle")
    owner = OUTPUT / "owner"
    owner.mkdir(exist_ok=False)
    cache = OUTPUT / "cache"
    cache.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=GPU,
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:.",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        COSMOS_TRAINING="0",
        TOKENIZERS_PARALLELISM="false",
        MASTER_ADDR="127.0.0.1",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
    )
    for name in (
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "NUMBA_CACHE_DIR",
        "MPLCONFIGDIR",
        "TMPDIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    ):
        path = cache / name.lower()
        path.mkdir()
        env[name] = str(path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        env["MASTER_PORT"] = str(sock.getsockname()[1])
    command = [
        prepared["model_config"]["model_python"],
        "-m",
        "scripts.wam.tracking.preview.ordered_feedback_wam_inference_20261111",
        "run",
    ]
    result = {
        "complete": False,
        "gpu_uuid": GPU,
        "before": before,
        "actual_wait": False,
        "reaped": False,
        "signal_sent": False,
        "automatic_retry": False,
    }
    with (owner / "worker.log").open("xb") as stream:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        result["pid"] = child.pid
        try:
            child.wait(timeout=1800)
        except BaseException:
            result["signal_sent"] = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            result.update(actual_wait=True, reaped=True, worker_exit_code=child.returncode)
            result["after"] = gpu_observation()
            result["released"] = not any(f", {child.pid}," in row for row in result["after"]["compute_apps"])
            result["complete"] = child.returncode == 0 and result["released"] and not result["signal_sent"]
            repro.write_new(owner / "result.json", result)
    print(json.dumps(result), flush=True)
    require(result["complete"], "Model worker did not complete and release its GPU")


def score_prediction(anchor, actions, targets, openings):
    import numpy as np

    from s2a.geometry import error_summary
    from s2a.references.action_reference import decode_action_targets

    predicted, predicted_opening = decode_action_targets(anchor, actions, horizon=40)
    result = {}
    for horizon in (8, 40):
        row = error_summary(targets[:horizon], predicted[:horizon])
        difference = predicted_opening[:horizon] - openings[:horizon]
        row["opening_rmse"] = float(np.sqrt(np.mean(difference**2)))
        row["opening_max_absolute"] = float(np.max(np.abs(difference)))
        result[f"h{horizon}"] = row
    return result


def score(config):
    check_cpu()
    checked_preflight()
    owner = json.loads((OUTPUT / "owner/result.json").read_text())
    result = json.loads((OUTPUT / "model_run/result.json").read_text())
    require(
        owner["complete"] is owner["actual_wait"] is owner["reaped"] is owner["released"] is True
        and owner["worker_exit_code"] == 0
        and result["complete"] is True
        and result["model_requests"] == 2
        and result["denoise_calls"] == 70,
        "Both model responses and an exited GPU worker are required before scoring",
    )
    captures = capture_inputs(config)
    from scripts.simulation.runtime import prepare_ordered_correct_stream as binding

    inputs = binding.load_inputs()
    require(inputs["previous"]["record"]["task"] == TASK, "Task identity differs")
    rows = []
    for index, (capture, item) in enumerate(zip(captures, result["responses"], strict=True)):
        response_path = OUTPUT / f"model_run/step{item['step']}/response.json"
        response = repro.read_checked(response_path, item["response_sha256"])
        arrays = repro.response_arrays(response)
        import numpy as np

        anchor = np.asarray(capture["current_source_pose"]) @ inputs["previous"]["tool"]
        rows.append(
            {
                "step": item["step"],
                "response_sha256": item["response_sha256"],
                "errors": score_prediction(
                    anchor, arrays["action"], inputs["targets"][index], inputs["openings"][index]
                ),
            }
        )
    report = {
        "complete": True,
        "rows": rows,
        "physical_closed_loop": False,
        "fresh_cases": 0,
        "model_inputs_included_future_truth": False,
    }
    repro.write_new(OUTPUT / "score.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "launch", "run", "score"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    {"preflight": preflight, "launch": launch, "run": run, "score": score}[args.mode](config)


if __name__ == "__main__":
    main()
