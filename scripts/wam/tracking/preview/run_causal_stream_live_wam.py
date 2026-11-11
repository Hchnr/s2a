"""Own one bounded live WAM stream and its independent CPU checker."""

import argparse
import json
import os
import random
import socket
import sys
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import causal_stream_live_admission as admission
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine

ROOT, OUTPUT, QA, DRY, GPU = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY, admission.GPU


def source_refs():
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/wam/tracking/preview/causal_stream_live_admission.py",
        ROOT / "scripts/wam/tracking/preview/live_stream_requests.py",
        ROOT / "scripts/wam/tracking/preview/live_cv2_admission.py",
        ROOT / "scripts/wam/tracking/preview/ordered_feedback_wam_inference_20261111.py",
        ROOT / "scripts/wam/tracking/preview/gpu_resume_repro_20261111.py",
        ROOT / "src/s2a/backends/wam/runtime.py",
        ROOT / "scripts/wam/readout/run_wam_readout_features.py",
        ROOT / "scripts/wam/training/clean/qualify_wam_clean_bridge.py",
        ROOT / "tests/research/wam/test_causal_stream_live_admission.py",
        ROOT / "tests/research/wam/test_live_stream_requests.py",
        ROOT / "tests/research/wam/test_live_cv2_admission.py",
        ROOT / "tests/research/wam/test_causal_stream_live_owner.py",
    )
    return physical.source_refs() + [prefix.reference(path) for path in names]


def dry_run():
    inputs = admission.load_inputs()
    result = {
        "complete": True,
        "passed": True,
        "inputs": admission.input_receipt(inputs),
        "source_refs": source_refs(),
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    print(publish(DRY, result), flush=True)


def check_cpu_admission(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and qa["configured_parallelism"] == 64,
        "Complete 64-worker CPU QA without skips or collection errors is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Live QA source identity changed")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["new_physics_steps"] == dry["new_model_requests"] == 0,
        "Matching zero-model and zero-physics dry admission required",
    )
    return before


def file_stats(paths, *, stat_fn=os.stat):
    result = {}
    for path in sorted(paths):
        stat = stat_fn(path)
        result[path] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def verify_model_manifests(inputs, *, verify=repro.verify_manifest, stat_fn=os.stat):
    """Hash the registered sources/weights and reject changes during hashing."""
    sources, weights = inputs["model_source_hashes"], inputs["weight_hashes"]
    prefix.require(len(sources) >= 100 and len(weights) == 163, "Complete model source and weight manifests required")
    before = file_stats(weights, stat_fn=stat_fn)
    verify(sources, workers=32)
    verify(weights, workers=32)
    after = file_stats(weights, stat_fn=stat_fn)
    prefix.require(before == after, "Checkpoint files changed during SHA verification")
    return {"source_files_verified": len(sources), "weight_files_verified": len(weights), "weight_stats": after}


def preflight():
    prefix.environment()
    inputs = admission.load_inputs()
    before = check_cpu_admission(inputs)
    prefix.require(not OUTPUT.exists(), "Preflight and physical outputs must be fresh")
    OUTPUT.mkdir()
    started = {"complete": False, "pid": os.getpid(), "config_ref": inputs["live_config_ref"]}
    publish(OUTPUT / "preflight_started.json", started)
    try:
        sources = source_refs()
        gpu = engine.gpu_observation()
        prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "Registered GPU must be idle")
        manifests = verify_model_manifests(inputs)
        probe_dir = OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
        probe_ref = publish(probe_dir / "result.json", probe)
        prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Preflight sources changed")
        result = {
            "complete": True,
            "passed": True,
            "inputs": admission.input_receipt(inputs),
            "source_refs": source_refs(),
            "qa_ref": prefix.reference(QA),
            "dry_ref": prefix.reference(DRY),
            "gpu_uuid": GPU,
            "gpu_before": gpu,
            "cv2_probe_ref": probe_ref,
            **manifests,
            "model_python_sha256": repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
            "model_initializations": 0,
            "model_requests": 0,
            "denoise_calls": 0,
            "physics_steps": 0,
            "automatic_retry": False,
        }
        print(publish(OUTPUT / "preflight.json", result), flush=True)
    except BaseException as error:
        publish(
            OUTPUT / "preflight_failed.json",
            {**started, "error": {"type": type(error).__name__, "message": str(error)}},
        )
        raise


def checked_preflight(inputs, *, recheck_weights=True):
    result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    prefix.require(
        result["complete"] is result["passed"] is True
        and result["inputs"] == admission.input_receipt(inputs)
        and result["source_refs"] == source_refs()
        and result["qa_ref"] == prefix.reference(QA)
        and result["dry_ref"] == prefix.reference(DRY)
        and result["gpu_uuid"] == GPU
        and result["source_files_verified"] >= 100
        and result["weight_files_verified"] == 163
        and result["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
        "The complete preflight and local model source identity must remain unchanged",
    )
    if recheck_weights:
        prefix.require(file_stats(inputs["weight_hashes"]) == result["weight_stats"], "Checkpoint file stats changed")
    probe = prefix.read(result["cv2_probe_ref"])
    prefix.require(
        probe["actual_wait"] is probe["reaped"] is True and probe["returncode"] == 0,
        "A completed CPU OpenCV probe is required",
    )
    return result


def worker_inputs(phase):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
    inputs = admission.load_inputs()
    checked_preflight(inputs)
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == source_refs(),
        "Live worker inputs and sources must match the owner registration",
    )
    return inputs, registration


def model_bundle(inputs):
    """Freeze only the model-side configuration, current gates and source roster."""
    return {
        "input_receipt": admission.input_receipt(inputs),
        "config": inputs["config"],
        "live_config": inputs["live_config"],
        "model_sha256": inputs["model_sha256"],
        "records": inputs["records"],
        "prepared": inputs["prepared"],
        "bridge": inputs["bridge"],
        "template": inputs["template"],
        "weight_hashes": inputs["weight_hashes"],
    }


def model_worker_inputs():
    """Read the owner's frozen bundle without pretending the GPU worker is CPU-only."""
    prefix.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and all(os.environ.get(key) == "1" for key in prefix.THREAD_VARS)
        and Path.cwd().resolve() == ROOT,
        "The model worker must expose exactly its registered GPU and repository environment",
    )
    registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
    bundle = prefix.read(registration["model_bundle_ref"])
    preflight_result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    admission.validate_config(bundle["live_config"])
    expected_weights = {path: detail["sha256"] for path, detail in bundle["prepared"]["weight_files"].items()}
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == bundle["input_receipt"] == preflight_result["inputs"]
        and registration["source_refs"] == preflight_result["source_refs"] == source_refs()
        and preflight_result["complete"] is preflight_result["passed"] is True
        and preflight_result["gpu_uuid"] == GPU
        and preflight_result["source_files_verified"] >= 100
        and preflight_result["weight_files_verified"] == 163
        and preflight_result["qa_ref"] == prefix.reference(QA)
        and preflight_result["dry_ref"] == prefix.reference(DRY)
        and bundle["weight_hashes"] == expected_weights
        and bundle["model_sha256"] == physical.admission.FIXED["historical_weight_manifest_sha256"]
        and preflight_result["model_python_sha256"] == repro.sha256(bundle["prepared"]["model_config"]["model_python"])
        and file_stats(bundle["weight_hashes"]) == preflight_result["weight_stats"],
        "The GPU worker requires its owner's frozen bundle and unchanged preflight",
    )
    probe = prefix.read(preflight_result["cv2_probe_ref"])
    prefix.require(
        probe["actual_wait"] is probe["reaped"] is True and probe["returncode"] == 0,
        "The model worker requires the completed CPU OpenCV probe",
    )
    return bundle


def model_worker():
    directory = OUTPUT / "producer/model"
    result = {
        "complete": False,
        "passed": False,
        "gpu_uuid": GPU,
        "model_initialization_attempts": 0,
        "model_initializations": 0,
        "model_requests_attempted": 0,
        "model_requests_completed": 0,
        "denoise_calls_attempted": 0,
        "denoise_calls_completed": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    channel, counter, evidence, torch = None, None, None, None
    try:
        inputs = model_worker_inputs()
        prefix.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and os.environ.get("LD_LIBRARY_PATH") == "",
            "The model worker must expose only its registered GPU UUID",
        )
        model_config = inputs["prepared"]["model_config"]
        prefix.require(
            Path(sys.executable).resolve() == Path(model_config["model_python"]).resolve(), "Model Python differs"
        )
        prefix.require(
            not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules),
            "Cosmos was imported before its source admission",
        )
        probe_ref = prefix.read(prefix.reference(OUTPUT / "preflight.json"))["cv2_probe_ref"]
        result["cv2_admission"] = live_cv2_admission.admit_model_cv2(prefix.read(probe_ref), gpu_uuid=GPU)
        sys.path.insert(0, str(Path(model_config["cosmos_source_root"]).resolve()))
        run_dir = directory / "model_run"
        run_dir.mkdir()
        import numpy as np
        import torch as torch_module

        from s2a.backends.wam.runtime import install_active_slot_mask, load_service
        from scripts.verification.wam.model_contracts.audit_wam_current_numerics import require_eval
        from scripts.wam.readout.run_wam_readout_features import BoundaryCounter, actual_context
        from scripts.wam.tracking.preview.live_stream_requests import execute_live_requests
        from scripts.wam.training.clean.qualify_wam_clean_bridge import imported_cosmos_sources

        torch = torch_module
        prefix.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device required")
        device = torch.cuda.get_device_properties(0)
        prefix.require(str(device.uuid).removeprefix("GPU-") == GPU.removeprefix("GPU-"), "Actual GPU UUID differs")
        result.update(torch_version=torch.__version__, compiled_cuda=torch.version.cuda, device_name=device.name)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        (run_dir / "model").mkdir()
        result["model_initialization_attempts"] = 1
        checkpoint = model_config["checkpoint"]
        service = load_service({"checkpoints": [checkpoint]}, "8400_masked", run_dir / "model")
        result["model_initializations"] = 1
        expected = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
        prefix.require(
            Path(service.info["checkpoint_path"]).resolve() == expected and service.info["ema_weights"] is False,
            "Loaded WAM checkpoint or EMA differs",
        )
        require_eval(service.model)
        for parameter in service.model.parameters():
            parameter.requires_grad_(False)
        evidence = install_active_slot_mask(service.model, [True] * 10 + [False] * 118)
        counter = BoundaryCounter(service.model, 70)
        imported_cosmos_sources(inputs["bridge"])
        from cosmos_framework.inference.args import OmniSampleOverrides

        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
        )
        execute_live_requests(
            inputs,
            channel=channel,
            image_root=OUTPUT / "producer/simulation/runtime",
            directory=run_dir,
            service=service,
            torch=torch,
            counter=counter,
            overrides=OmniSampleOverrides,
            domain_reader=actual_context,
            mask_evidence=evidence,
            result=result,
        )
        prefix.require(
            result["model_requests_attempted"] == result["model_requests_completed"] == result["responses_sent"] == 2
            and result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 70,
            "Both actual model requests and all denoise calls must complete",
        )
        model_worker_inputs()
        result["imported_cosmos_sources"] = imported_cosmos_sources(inputs["bridge"])
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if counter is not None:
            result["denoise_calls_attempted"] = len(counter.calls)
            result["denoise_calls_completed"] = sum(bool(call["completed"]) for call in counter.calls)
            result["denoise_evidence"] = counter.calls
        if evidence is not None:
            result["mask_evidence"] = evidence
        cleanup_error = None
        try:
            if channel is not None:
                channel.close()
            if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except BaseException as error:
            cleanup_error = error
            result.update(
                complete=False, passed=False, cleanup_error={"type": type(error).__name__, "message": str(error)}
            )
        print(publish(directory / "result.json", result), flush=True)
        if cleanup_error is not None:
            raise cleanup_error


def _model_environment(directory):
    env = physical._cpu_environment(directory)
    env.update(
        CUDA_VISIBLE_DEVICES=GPU,
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
        path = directory / "cache" / name.lower()
        path.mkdir()
        env[name] = str(path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        env["MASTER_PORT"] = str(sock.getsockname()[1])
    return env


def execute(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered phase required")
    inputs = admission.load_inputs()
    before = check_cpu_admission(inputs)
    checked_preflight(inputs)
    sources = source_refs()
    producer_ref = None
    if phase == "checker":
        prior = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            prior["complete"] is prior["passed"] is prior["sources_unchanged"] is prior["gpu_released"] is True,
            "The producer and its GPU worker must be fully closed",
        )
        producer_ref = prior["branch_ref"]
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    (directory / "simulation").mkdir()
    bundle_ref = None
    if phase == "producer":
        (directory / "model").mkdir()
        bundle_ref = publish(directory / "model/registered_inputs.json", model_bundle(inputs))
    registration = publish(
        directory / "registration.json",
        {
            "phase": phase,
            "inputs": admission.input_receipt(inputs),
            "source_refs": sources,
            "producer_branch_ref": producer_ref,
            "model_bundle_ref": bundle_ref,
        },
    )
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "control_qualified": False,
        "registration_ref": registration,
        "gpu_uuid": GPU,
        "automatic_retry": False,
        "model_initializations": None,
        "model_requests_attempted": None,
        "denoise_calls_attempted": None,
    }
    try:
        simulation_env = physical._cpu_environment(directory / "simulation")
        command = [sys.executable, "-m", "scripts.wam.tracking.preview.run_causal_stream_live_wam"]
        if phase == "producer":
            before_gpu = engine.gpu_observation()
            prefix.require(
                before_gpu["memory_mib"] == 0 and before_gpu["compute_apps"] == [], "Registered GPU must be idle"
            )
            result["gpu_before"] = before_gpu
            model_env = _model_environment(directory / "model")
            model_python = inputs["prepared"]["model_config"]["model_python"]
            specs = {
                "simulation": {
                    "argv": command + ["--worker", "simulation"],
                    "env": simulation_env,
                    "log": str((directory / "simulation.log").relative_to(ROOT)),
                },
                "model": {
                    "argv": [
                        model_python,
                        "-m",
                        "scripts.wam.tracking.preview.run_causal_stream_live_wam",
                        "--worker",
                        "model",
                    ],
                    "env": model_env,
                    "log": str((directory / "model.log").relative_to(ROOT)),
                },
            }
            try:
                lifecycle = run_owned_socket_pair(
                    specs, timeout_seconds=inputs["live_config"]["worker_timeout_seconds"]
                )
            except OwnedSocketPairInterrupted as error:
                result["workers_ref"] = publish(directory / "workers.json", error.result)
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            prefix.require(lifecycle["passed"] is True, "Both live stream workers must exit and be reaped")
            model_ref = prefix.reference(directory / "model/result.json")
            model = prefix.read(model_ref)
            result["model_ref"] = model_ref
            result.update(
                model_initializations=model["model_initializations"],
                model_requests_attempted=model["model_requests_attempted"],
                denoise_calls_attempted=model["denoise_calls_attempted"],
            )
            prefix.require(
                model["complete"] is model["passed"] is True
                and model["model_initializations"] == 1
                and model["model_requests_attempted"]
                == model["model_requests_completed"]
                == model["responses_sent"]
                == 2
                and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 70,
                "The registered model execution must complete exactly once",
            )
            after_gpu = engine.gpu_observation()
            result["gpu_after"] = after_gpu
            result["gpu_released"] = after_gpu["memory_mib"] == 0 and after_gpu["compute_apps"] == []
            prefix.require(result["gpu_released"] is True, "The selected GPU must be released")
        else:
            spec = {"key": "checker", "argv": command + ["--worker", "checker"], "log": directory / "checker.log"}
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=inputs["live_config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            physical.historical.registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
            result.update(
                gpu_released=True, model_initializations=0, model_requests_attempted=0, denoise_calls_attempted=0
            )
        branch_ref = prefix.reference(directory / "simulation/result.json")
        branch = prefix.read(branch_ref)
        counts = physical.summarize_branch(branch, phase=phase)
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Sources changed during live execution"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    finally:
        if phase == "producer":
            model_path = directory / "model/result.json"
            if model_path.is_file():
                result["available_model_ref"] = prefix.reference(model_path)
                model = json.loads(model_path.read_text())
                result.update(
                    model_initializations=model.get("model_initializations"),
                    model_requests_attempted=model.get("model_requests_attempted"),
                    denoise_calls_attempted=model.get("denoise_calls_attempted"),
                )
            if "gpu_after" not in result:
                try:
                    result["gpu_after"] = engine.gpu_observation()
                    result["gpu_released"] = (
                        result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
                    )
                except Exception as error:
                    result["gpu_after_error"] = {"type": type(error).__name__, "message": str(error)}
        branch_path = directory / "simulation/result.json"
        if branch_path.is_file():
            result["available_branch_ref"] = prefix.reference(branch_path)
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", choices=("simulation", "model", "checker"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.preflight:
        preflight()
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    elif args.worker == "simulation":
        physical.simulation_worker(output=OUTPUT, loader=worker_inputs, channel_timeout=900)
    elif args.worker == "checker":
        physical.checker_worker(output=OUTPUT, loader=worker_inputs)
    else:
        model_worker()


if __name__ == "__main__":
    main()
