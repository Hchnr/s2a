"""Generate two frozen WAM references from independently verified branch images."""

import argparse
import json
import os
import signal
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_early_tabletop_capture as capture_owner
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as clone_support
from scripts.simulation.runtime import run_tabletop_projection_feedback_probe as physical_audit
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview.live_stream_requests import _check_response_identity

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/early_tabletop_live_wam_model_20261111.json"
CONFIG_SHA = "7064f750856353c1167663c6f55ada33e6ecbd05190b62ce131915bce7a6d2a4"
PLAN = ROOT / "plans/v0.0.1/tracking/references/early_tabletop_intervention_live_wam_20261111.md"
OUTPUT = ROOT / "outputs/early_tabletop_intervention_live_wam_20261111/model"
DRY = ROOT / "outputs/early_tabletop_live_wam_model_dry_20261111/result.json"
QA = ROOT / "outputs/qa/early_tabletop_live_wam_model_20261111/result.json"
MODULE = "scripts.wam.tracking.preview.early_tabletop_live_wam"
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"


def load_config():
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA})
    prefix.require(
        config["schema"] == "s2a.early-tabletop-live-wam-model.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["task"] == engine.TASK
        and config["gpu_uuid"] == GPU == engine.GPU
        and (config["seed"], config["horizon"], config["denoise_steps_per_request"]) == (42, 40, 35)
        and (config["model_initializations"], config["model_requests"], config["denoise_calls"]) == (1, 2, 70)
        and config["physics_steps"] == config["fresh_cases"] == config["fits"] == 0
        and config["automatic_retry"] is False
        and [item["mode"] for item in config["images"]] == ["original", "pose_bounded"]
        and [item["step"] for item in config["images"]] == [120, 120],
        "The exact branch-specific images, GPU and fixed model budget are required",
    )
    return config


def validate_binding(image, capture):
    expected = {"path": str(ROOT / image["path"]), "sha256": image["sha256"]}
    prefix.require(
        image["mode"] in ("original", "pose_bounded")
        and image["step"] == 120
        and capture["frame_index"] == 120
        and capture["canvas_png"] == expected
        and capture["current_endpoint"]["sim_state_sha256"] == image["state_sha256"]
        and capture["render_before"] == capture["render_after"]
        and capture["render_before"]["sim_state_sha256"] == image["state_sha256"]
        and capture["render_before"]["sim_time"] == capture["current_endpoint"]["sim_time"]
        and capture["future_rgb_frames"] == 0
        and capture["camera_roles"] == ["auxiliary_0", "left_wrist"]
        and capture["canvas_shape"] == [960, 640, 3]
        and capture["canvas_dtype"] == "uint8",
        "A branch's exact unchanged state and own current dual-camera image are required",
    )
    return expected


def capture_inputs(config):
    """Check both physical producers, isolated replays and exact image identities."""
    owner_ref = {"path": str(ROOT / config["capture_owner"]), "sha256": config["capture_owner_sha256"]}
    owner = prefix.read(owner_ref)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is owner["ready_for_model"] is True
        and owner["control_qualified"] is False
        and owner["total_native_steps"] == 12000
        and owner["newly_selected_commands"] == 4
        and owner["world_model_calls"] == owner["fresh_cases"] == 0,
        "The complete independently replayed early capture is required",
    )
    for group in owner["groups"]:
        lifecycle = prefix.read(group["workers_ref"])
        prefix.require(
            lifecycle["launch_error"] is None
            and lifecycle["workers"]
            and all(
                row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0 and row["signal_sent"] is False
                for row in lifecycle["workers"]
            ),
            "All capture workers must have exited and been reaped",
        )
    captures = []
    for image in config["images"]:
        mode = image["mode"]
        producer = prefix.read(owner["producer_refs"][mode])
        checker = prefix.read(owner["checker_refs"][mode])
        prefix.require(
            producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True,
            "Both physical branches must be complete",
        )
        physical_audit.check_pair(producer, checker)
        paired = capture_owner.compare_capture(producer, checker)
        capture = producer["runtime"]["continuation"]["next_current_image"]["capture"]
        expected = validate_binding(image, capture)
        prefix.require(
            paired == owner["captures"][mode] and paired["image_ref"] == expected,
            "Capture owner and replay binding must match",
        )
        prefix.checked(expected)
        captures.append(capture)
    return captures


def response_guard(image, response, item):
    request = json.loads((item / "request.json").read_text())
    prepared, _bridge, _completed, _template, _response = engine.previous_inputs()
    _check_response_identity(
        response, model_request=request, task=engine.TASK, checkpoint=prepared["model_config"]["checkpoint"]
    )
    return {"mode": image["mode"], "image_sha256": image["sha256"], "state_sha256": image["state_sha256"]}


def context():
    return engine.InferenceContext(
        config_path=CONFIG,
        plan_path=PLAN,
        output=OUTPUT,
        qa=QA,
        capture_loader=capture_inputs,
        additional_sources=(
            Path(__file__).resolve(),
            ROOT / "configs/tracking/references/early_tabletop_intervention_live_wam_20261111.json",
            ROOT / "scripts/simulation/runtime/run_early_tabletop_capture.py",
            ROOT / "scripts/simulation/runtime/run_tabletop_projection_feedback_probe.py",
            ROOT / "scripts/simulation/runtime/replay_multicase_development_prefix.py",
            ROOT / "scripts/wam/tracking/preview/live_stream_requests.py",
            ROOT / "tests/research/wam/test_early_tabletop_live_wam.py",
        ),
        worker_module=MODULE,
        response_guard=response_guard,
    )


def split_branch_requests(
    config,
    ctx,
    service,
    torch,
    counter,
    overrides,
    domain_reader,
    directory,
    template,
    result,
    responses,
    *,
    delegate=None,
):
    """Give the two same-step images distinct output directories and identities."""
    delegate = engine.execute_requests if delegate is None else delegate
    for image in config["images"]:
        mode = image["mode"]
        prefix.require(mode in ("original", "pose_bounded"), "A registered branch mode is required")
        branch = directory / mode
        branch.mkdir()
        count = len(responses)
        delegate(
            {**config, "images": [image]},
            ctx,
            service,
            torch,
            counter,
            overrides,
            domain_reader,
            branch,
            template,
            result,
            responses,
        )
        prefix.require(len(responses) == count + 1, "Exactly one response per branch is required")
        responses[-1]["mode"] = mode
    return responses


def dry_run(config):
    prefix.environment()
    captures = capture_inputs(config)
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
                "capture_owner_ref": {
                    "path": str(ROOT / config["capture_owner"]),
                    "sha256": config["capture_owner_sha256"],
                },
                "images": [capture["canvas_png"] for capture in captures],
                "local_sources": engine.local_sources(context()),
                "model_initializations": 0,
                "model_requests": 0,
                "physics_steps": 0,
                "fresh_cases": 0,
            },
        ),
        flush=True,
    )


def preflight(config):
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == snapshot(ROOT)["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["config_ref"]["sha256"] == CONFIG_SHA
        and dry["local_sources"] == engine.local_sources(context())
        and dry["model_initializations"] == dry["model_requests"] == dry["physics_steps"] == 0,
        "Source-matched 32-worker QA and zero-work dry are required",
    )
    gpu = engine.gpu_observation()
    prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "The registered authorized GPU must be idle")
    proxy = SimpleNamespace(
        **{**vars(repro), "verify_manifest": lambda values: repro.verify_manifest(values, workers=32)}
    )
    run = clone_support.clone(engine.preflight, {"repro": proxy})
    run(config, context())
    prepared = engine.previous_inputs()[0]
    print(
        publish(
            OUTPUT / "device_preflight.json",
            {
                "complete": True,
                "gpu_uuid": GPU,
                "gpu_before": gpu,
                "model_python": prepared["model_config"]["model_python"],
                "model_python_sha256": repro.sha256(prepared["model_config"]["model_python"]),
                "model_initializations": 0,
                "model_requests": 0,
                "physics_steps": 0,
            },
        ),
        flush=True,
    )


def worker(config):
    prefix.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and all(os.environ.get(key) == "1" for key in prefix.THREAD_VARS),
        "Exactly the registered GPU and single-thread libraries are required",
    )
    device = prefix.read(prefix.reference(OUTPUT / "device_preflight.json"))
    prefix.require(
        device["complete"] is True
        and device["gpu_uuid"] == GPU
        and device["model_python_sha256"] == repro.sha256(device["model_python"]),
        "Model Python and GPU preflight must match",
    )
    run = clone_support.clone(engine.run, {"execute_requests": split_branch_requests})
    return run(config, context())


def launch(config):
    prefix.environment()
    engine.checked_preflight(context())
    device = prefix.read(prefix.reference(OUTPUT / "device_preflight.json"))
    prefix.require(device["complete"] is True and device["gpu_uuid"] == GPU, "Device preflight is required")
    before = engine.gpu_observation()
    prefix.require(before["memory_mib"] == 0 and before["compute_apps"] == [], "The registered GPU must still be idle")
    owner, cache = OUTPUT / "owner", OUTPUT / "cache"
    owner.mkdir()
    cache.mkdir()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=GPU,
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:.",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        LP_NUM_THREADS="1",
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
        directory = cache / name.lower()
        directory.mkdir()
        env[name] = str(directory)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        env["MASTER_PORT"] = str(sock.getsockname()[1])
    result = {
        "complete": False,
        "gpu_uuid": GPU,
        "before": before,
        "actual_wait": False,
        "reaped": False,
        "signal_sent": False,
        "automatic_retry": False,
    }
    child = None
    try:
        with (owner / "worker.log").open("xb") as stream:
            child = subprocess.Popen(
                [device["model_python"], "-m", MODULE, "run"],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["pid"] = child.pid
            try:
                child.wait(timeout=1800)
            except BaseException:
                result["signal_sent"] = True
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                raise
            finally:
                result.update(actual_wait=True, reaped=True, worker_exit_code=child.returncode)
        after = engine.gpu_observation()
        result["after"] = after
        result["released"] = not any(f", {child.pid}," in row for row in after["compute_apps"])
        result["complete"] = child.returncode == 0 and result["released"] and not result["signal_sent"]
        prefix.require(result["complete"], "The model worker must complete and release its GPU")
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if child is not None and "after" not in result:
            try:
                result["after"] = engine.gpu_observation()
                result["released"] = not any(f", {child.pid}," in row for row in result["after"]["compute_apps"])
            except BaseException as error:
                result["gpu_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        print(publish(owner / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry", "preflight", "launch", "run"))
    args = parser.parse_args()
    config = load_config()
    if args.mode == "dry":
        dry_run(config)
    elif args.mode == "preflight":
        preflight(config)
    elif args.mode == "launch":
        launch(config)
    else:
        worker(config)


if __name__ == "__main__":
    main()
