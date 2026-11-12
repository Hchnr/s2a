"""Preflight and supervise the registered adaptive model/simulation trial."""

import hashlib
import json
import os
import sys
from pathlib import Path

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_audit as audit
from scripts.simulation.runtime import adaptive_live_bowl_worker as simulation
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_replan_step107_execution as previous
from scripts.simulation.runtime import run_causal_replan_step107_model as previous_model
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.simulation.runtime.run_adaptive_live_bowl"
REQUIRED_QA_TARGETS = (
    "tests/research/native_runtime/test_adaptive_live_bowl_admission.py",
    "tests/research/native_runtime/test_adaptive_live_bowl_worker.py",
    "tests/research/native_runtime/test_adaptive_live_bowl_audit.py",
    "tests/research/native_runtime/test_adaptive_live_bowl.py",
    "tests/research/native_runtime/test_completed_period_boundary.py",
    "tests/research/native_runtime/test_adaptive_streaming_replay.py",
    "tests/research/native_runtime/test_adaptive_stream_reference.py",
    "tests/research/native_runtime/test_adaptive_streaming_continuation.py",
    "tests/research/wam/test_adaptive_model_worker.py",
    "tests/research/wam/test_adaptive_live_requests.py",
    "tests/research/wam/test_adaptive_live_pipeline.py",
    "tests/unit/infrastructure/test_adaptive_stream_protocol.py",
    "tests/research/joint_control/preview/test_causal_reference_replan.py",
    "tests/research/native_runtime/test_adaptive_live_bowl_supervisor.py",
)


def source_refs():
    paths = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_adaptive_live_bowl.py",
        ROOT / "scripts/simulation/runtime/adaptive_live_bowl_admission.py",
        ROOT / "scripts/simulation/runtime/adaptive_live_bowl_worker.py",
        ROOT / "scripts/simulation/runtime/adaptive_live_bowl_audit.py",
        ROOT / "scripts/simulation/runtime/adaptive_live_bowl.py",
        ROOT / "scripts/wam/tracking/preview/adaptive_model_worker.py",
        ROOT / "scripts/wam/tracking/preview/adaptive_live_requests.py",
        ROOT / "src/s2a/research/native_runtime/completed_period_boundary.py",
        ROOT / "src/s2a/research/native_runtime/adaptive_streaming_continuation.py",
        ROOT / "src/s2a/research/native_runtime/adaptive_streaming_replay.py",
        ROOT / "src/s2a/research/native_runtime/streaming_dispatch.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/adaptive_stream_protocol.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/adaptive_stream_reference.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/adaptive_stream_control.py",
        *(ROOT / path for path in REQUIRED_QA_TARGETS),
    )
    return previous.source_refs() + previous_model.source_refs() + [prefix.reference(path) for path in paths]


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_bundle(inputs):
    """Freeze the exact model inputs and phase-local current-image registration."""
    return dict(
        input_receipt=admission.input_receipt(inputs),
        prepared=inputs["prepared"],
        bridge=inputs["bridge"],
        template=inputs["template"],
        weight_hashes=inputs["weight_hashes"],
        model_registration=admission.model_registration(inputs, OUTPUT / "producer/simulation"),
    )


def dry_run():
    inputs = admission.load_inputs()
    result = dict(
        complete=True,
        passed=True,
        inputs=admission.input_receipt(inputs),
        source_refs=source_refs(),
        model_bundle_sha256=_digest(model_bundle(inputs)),
        native_models=0,
        physics_steps=0,
        model_initializations=0,
        model_requests=0,
        denoise_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(DRY, result), flush=True)


def check_cpu_admission(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 64
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and set(REQUIRED_QA_TARGETS).issubset(set(qa["targets"]))
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and dry["model_bundle_sha256"] == _digest(model_bundle(inputs))
        and dry["native_models"]
        == dry["physics_steps"]
        == dry["model_initializations"]
        == dry["model_requests"]
        == dry["denoise_calls"]
        == dry["fresh_cases"]
        == 0,
        "Matching 64-worker CPU regression and zero-physics dry admission are required",
    )
    return before


def preflight():
    prefix.environment()
    inputs = admission.load_inputs()
    before, sources = check_cpu_admission(inputs), source_refs()
    prefix.require(engine.GPU == inputs["config"]["gpu_uuid"], "GPU observation must target the registered UUID")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = dict(complete=False, config=prefix.reference(admission.CONFIG), pid=os.getpid())
    publish(OUTPUT / "preflight_started.json", started)
    try:
        gpu = engine.gpu_observation()
        prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "The registered GPU must be idle")
        manifests = model_owner.verify_model_manifests(inputs)
        probe_dir = OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
        probe_ref = publish(probe_dir / "result.json", probe)
        prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Preflight sources changed")
        result = dict(
            complete=True,
            passed=True,
            inputs=admission.input_receipt(inputs),
            source_refs=sources,
            qa_ref=prefix.reference(QA),
            dry_ref=prefix.reference(DRY),
            model_bundle_sha256=_digest(model_bundle(inputs)),
            gpu_uuid=inputs["config"]["gpu_uuid"],
            gpu_before=gpu,
            cv2_probe_ref=probe_ref,
            model_python_sha256=repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
            model_initializations=0,
            model_requests=0,
            physics_steps=0,
            automatic_retry=False,
            **manifests,
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
        and value["inputs"] == admission.input_receipt(inputs)
        and value["source_refs"] == source_refs()
        and value["qa_ref"] == prefix.reference(QA)
        and value["dry_ref"] == prefix.reference(DRY)
        and value["model_bundle_sha256"] == _digest(model_bundle(inputs))
        and value["gpu_uuid"] == inputs["config"]["gpu_uuid"]
        and value["source_files_verified"] == len(inputs["model_source_hashes"])
        and value["weight_files_verified"] == len(inputs["weight_hashes"]) == 163
        and value["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"])
        and value["weight_stats"] == model_owner.file_stats(inputs["weight_hashes"]),
        "The source-matched GPU and checkpoint preflight is required",
    )
    probe = prefix.read(value["cv2_probe_ref"])
    prefix.require(probe["actual_wait"] is probe["reaped"] is True and probe["returncode"] == 0, "CPU OpenCV probe")
    return value


def model_worker_inputs():
    """Revalidate the registered bundle without importing the model or simulator."""
    registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
    bundle = prefix.read(registration["model_bundle_ref"])
    preflight_result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    weights = {path: detail["sha256"] for path, detail in bundle["prepared"]["weight_files"].items()}
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == bundle["input_receipt"] == preflight_result["inputs"]
        and registration["source_refs"] == preflight_result["source_refs"] == source_refs()
        and preflight_result["complete"] is preflight_result["passed"] is True
        and preflight_result["qa_ref"] == prefix.reference(QA)
        and preflight_result["dry_ref"] == prefix.reference(DRY)
        and preflight_result["model_bundle_sha256"] == _digest(bundle)
        and preflight_result["gpu_uuid"] == engine.GPU
        and preflight_result["weight_files_verified"] == len(weights) == 163
        and bundle["weight_hashes"] == weights
        and preflight_result["weight_stats"] == model_owner.file_stats(weights)
        and preflight_result["model_python_sha256"] == repro.sha256(bundle["prepared"]["model_config"]["model_python"]),
        "The model worker requires the exact preflight and registered bundle",
    )
    return bundle, preflight_result


def _record_worker_logs(lifecycle):
    for item in lifecycle["workers"]:
        path = Path(item.pop("log"))
        item["log_ref"] = prefix.reference(path if path.is_absolute() else ROOT / path)
    return lifecycle


def execute(phase):
    """Run the one registered producer or its independent replay exactly once."""
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "A registered physical phase is required")
    inputs = admission.load_inputs()
    before, sources = check_cpu_admission(inputs), source_refs()
    check_preflight(inputs)
    previous_result = None
    if phase == "checker":
        previous_result = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            previous_result["complete"]
            is previous_result["passed"]
            is previous_result["sources_unchanged"]
            is previous_result["gpu_released"]
            is True,
            "A completed producer with released GPU is required",
        )
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    simulation_dir = directory / "simulation"
    simulation_dir.mkdir()
    model_ref = None
    if phase == "producer":
        model_dir = directory / "model"
        model_dir.mkdir()
        model_ref = publish(model_dir / "registered_inputs.json", model_bundle(inputs))
    registration = publish(
        directory / "registration.json",
        dict(
            phase=phase,
            inputs=admission.input_receipt(inputs),
            source_refs=sources,
            model_bundle_ref=model_ref,
            producer_branch_ref=None if previous_result is None else previous_result["branch_ref"],
            producer_model_ref=None if previous_result is None else previous_result["model_ref"],
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        control_qualified=False,
        registration_ref=registration,
        gpu_uuid=inputs["config"]["gpu_uuid"],
        automatic_retry=False,
        model_initializations=None,
        model_requests_attempted=None,
        denoise_calls_attempted=None,
    )
    try:
        simulation_env = physical._cpu_environment(simulation_dir)
        command = [sys.executable, "-m", MODULE]
        if phase == "producer":
            result["gpu_before"] = engine.gpu_observation()
            prefix.require(
                result["gpu_before"]["memory_mib"] == 0 and result["gpu_before"]["compute_apps"] == [],
                "The registered GPU must be idle before producer startup",
            )
            model_env = model_owner._model_environment(model_dir)
            model_env["CUDA_VISIBLE_DEVICES"] = inputs["config"]["gpu_uuid"]
            specs = {
                "simulation": dict(
                    argv=command + ["--worker", "producer"],
                    env=simulation_env,
                    log=str((directory / "simulation.log").relative_to(ROOT)),
                ),
                "model": dict(
                    argv=[inputs["prepared"]["model_config"]["model_python"], "-m", MODULE, "--worker", "model"],
                    env=model_env,
                    log=str((directory / "model.log").relative_to(ROOT)),
                ),
            }
            try:
                lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
            except OwnedSocketPairInterrupted as error:
                result["workers_ref"] = publish(directory / "workers.json", _record_worker_logs(error.result))
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", _record_worker_logs(lifecycle))
            prefix.require(lifecycle["passed"] is True, "The model and simulation workers must both be reaped")
            actual_model_ref = prefix.reference(model_dir / "result.json")
            actual_model = prefix.read(actual_model_ref)
            result["model_ref"] = actual_model_ref
            prefix.require(
                actual_model["complete"] is actual_model["passed"] is True
                and actual_model["model_initializations"] == 1
                and 0 <= actual_model["responses_sent"] <= inputs["config"]["max_model_requests"]
                and actual_model["model_requests_attempted"]
                == actual_model["model_requests_completed"]
                == actual_model["responses_sent"]
                and actual_model["denoise_calls_attempted"]
                == actual_model["denoise_calls_completed"]
                == 35 * actual_model["responses_sent"],
                "The actual model calls and denoise counts must remain within registration",
            )
            result["gpu_after"] = engine.gpu_observation()
            result["gpu_released"] = (
                result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
            )
            prefix.require(result["gpu_released"] is True, "The registered GPU must be released")
        else:
            spec = {"key": "checker", "argv": command + ["--worker", "checker"], "log": directory / "checker.log"}
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", _record_worker_logs(lifecycle))
            lifecycle_check.require_closed_workers(lifecycle, ["checker"])
            result.update(
                gpu_released=True, model_initializations=0, model_requests_attempted=0, denoise_calls_attempted=0
            )
        branch_ref = prefix.reference(simulation_dir / "result.json")
        branch = prefix.read(branch_ref)
        counts = audit.summarize_branch(branch, phase, inputs["config"])
        if phase == "producer":
            model_result = prefix.read(result["model_ref"])
            simulation.load_recorded_responses(branch, model_result)
            prefix.require(
                counts["adaptive_steps"] == branch["runtime"]["continuation"]["adaptive"]["control_steps_completed"]
                and model_result["responses_sent"] == len(branch["runtime"]["adaptive_model_exchanges"]),
                "Producer physics and accepted model exchanges must agree",
            )
        else:
            producer = prefix.read(previous_result["branch_ref"])
            counts["independent_max_absolute_difference"] = audit.compare_branches(producer, branch)
            prefix.require(
                counts["control_steps"] == previous_result["counts"]["control_steps"], "Replay count differs"
            )
            counts.update(
                cumulative_control_steps=inputs["config"]["previous_control_steps"]
                + previous_result["counts"]["control_steps"]
                + counts["control_steps"],
                cumulative_native_steps=inputs["config"]["previous_native_steps"]
                + previous_result["counts"]["native_steps"]
                + counts["native_steps"],
                cumulative_models=inputs["config"]["previous_models"]
                + previous_result["counts"]["models"]
                + counts["models"],
                cumulative_contexts=inputs["config"]["previous_contexts"]
                + previous_result["counts"]["contexts"]
                + counts["contexts"],
            )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(ROOT)["sources"] == before, "Trial sources changed")
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if phase == "producer":
            model_path = directory / "model/result.json"
            if model_path.is_file():
                result["available_model_ref"] = prefix.reference(model_path)
                model = prefix.read(result["available_model_ref"])
                for key in (
                    "model_initialization_attempts",
                    "model_initializations",
                    "model_requests_attempted",
                    "model_requests_completed",
                    "denoise_calls_attempted",
                    "denoise_calls_completed",
                ):
                    result[key] = model.get(key)
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
