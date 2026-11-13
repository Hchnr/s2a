"""Supervise one registered real-model replay of a saved development image."""

import copy
import shutil
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as historical
from scripts.simulation.runtime import adaptive_multiseed_real43_admission as admission
from scripts.simulation.runtime import adaptive_multiseed_real43_replay as replay
from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import registered_model_bundle
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import adaptive_multiseed_contract as contract
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner

MODULE = "scripts.simulation.runtime.run_adaptive_multiseed_real43"
REQUIRED_QA_TARGETS = (
    "tests/research/native_runtime/test_adaptive_multiseed_real43_admission.py",
    "tests/research/native_runtime/test_adaptive_multiseed_real43_replay.py",
    "tests/research/native_runtime/test_adaptive_multiseed_real43_owner.py",
    "tests/research/native_runtime/test_registered_adaptive_trial.py",
    "tests/research/wam/test_adaptive_multiseed_contract.py",
    "tests/research/wam/test_adaptive_multiseed_requests.py",
    "tests/research/wam/test_adaptive_multiseed_model_worker.py",
)


def source_refs():
    """Bind every new entry point and its shared model/transport dependencies."""
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(admission.__file__),
        Path(replay.__file__),
        Path(__file__),
        admission.ROOT / "scripts/simulation/runtime/run_adaptive_multiseed_real43.py",
        Path(registered.__file__),
        Path(registered_model_bundle.__file__),
        Path(physical.__file__),
        Path(model_owner.__file__),
        Path(repro.__file__),
        Path(contract.__file__),
        admission.ROOT / "scripts/wam/tracking/preview/adaptive_multiseed_model_worker.py",
        admission.ROOT / "scripts/wam/tracking/preview/adaptive_multiseed_requests.py",
        admission.ROOT / "scripts/wam/tracking/preview/live_cv2_admission.py",
        admission.ROOT / "scripts/wam/tracking/preview/ordered_feedback_wam_inference_20261111.py",
        admission.ROOT / "scripts/wam/readout/run_wam_readout_features.py",
        admission.ROOT / "scripts/wam/training/clean/qualify_wam_clean_bridge.py",
        admission.ROOT / "scripts/verification/wam/model_contracts/audit_wam_current_numerics.py",
        admission.ROOT / "src/s2a/backends/wam/runtime.py",
        admission.ROOT / "src/s2a/research/wam/tracking/runtime/sampling_seed.py",
        admission.ROOT / "src/s2a/research/wam/tracking/runtime/adaptive_stream_protocol.py",
        admission.ROOT / "src/s2a/research/wam/tracking/runtime/causal_stream_protocol.py",
        admission.ROOT / "src/s2a/infrastructure/framed_channel.py",
        admission.ROOT / "src/s2a/infrastructure/lifecycle/owned_socket_pair.py",
        *(admission.ROOT / path for path in REQUIRED_QA_TARGETS),
    )
    return [prefix.reference(path.resolve()) for path in names]


def make_trial():
    return registered.RegisteredAdaptiveTrial(
        output=admission.OUTPUT,
        qa=admission.QA,
        dry=admission.DRY,
        module=MODULE,
        gpu_uuid=admission.GPU,
        required_qa_targets=REQUIRED_QA_TARGETS,
        load_inputs=admission.load_inputs,
        input_receipt=admission.input_receipt,
        source_refs=source_refs,
        model_registration=admission.model_registration,
    )


def replay_worker_inputs():
    """Check the same owner registration before replaying any frame."""
    trial = make_trial()
    inputs = admission.load_inputs()
    trial.check_preflight(inputs)
    registered_phase = prefix.read(prefix.reference(admission.OUTPUT / "producer/registration.json"))
    prefix.require(
        registered_phase["phase"] == "producer"
        and registered_phase["inputs"] == admission.input_receipt(inputs)
        and registered_phase["source_refs"] == source_refs()
        and type(registered_phase["model_bundle_ref"]) is dict,
        "The saved-frame client must share the owner's registered inputs and source identity",
    )
    return inputs, admission.model_registration(inputs, admission.OUTPUT / "producer/simulation")


def model_cost_bounds(lifecycle, model, *, launch_attempted, sources_match):
    """Use exact final counters only for a reaped source-matched worker."""
    limits = {
        "model_initialization_attempts": 1,
        "model_initializations": 1,
        "model_requests_attempted": 1,
        "model_requests_completed": 1,
        "denoise_calls_attempted": 35,
        "denoise_calls_completed": 35,
    }
    rows = [] if lifecycle is None else [row for row in lifecycle["workers"] if row["name"] == "model"]
    prefix.require(len(rows) <= 1, "At most one registered model worker is allowed")
    if not launch_attempted or (lifecycle is not None and not rows):
        return {name: {"lower": 0, "upper": 0} for name in limits}
    final = (
        len(rows) == 1
        and rows[0].get("actual_wait") is rows[0].get("reaped") is True
        and sources_match is True
        and type(model) is dict
    )
    if not final:
        return {name: {"lower": 0, "upper": limit} for name, limit in limits.items()}
    prefix.require(
        all(type(model.get(name)) is int and 0 <= model[name] <= limit for name, limit in limits.items())
        and model["model_initializations"] <= model["model_initialization_attempts"]
        and model["model_requests_completed"] <= model["model_requests_attempted"]
        and model["denoise_calls_completed"] <= model["denoise_calls_attempted"]
        and (model["model_requests_attempted"] == 0 or model["model_initializations"] == 1)
        and (model["denoise_calls_attempted"] == 0 or model["model_requests_attempted"] == 1),
        "The final worker counters must be internally consistent and within budget",
    )
    return {name: {"lower": model[name], "upper": model[name]} for name in limits}


def audit_exchange(inputs, client, model):
    """Independently compare client envelope to the saved actual model response."""
    config = inputs["config"]
    prefix.require(
        client["complete"] is client["passed"] is model["complete"] is model["passed"] is True
        and client["evidence_level"] == "saved_development_image_model_replay"
        and client["physics_steps"] == client["native_models"] == client["action_dispatches"] == 0
        and model["gpu_uuid"] == config["gpu_uuid"]
        and model["sampling_seed"] == config["seed"] == 43
        and model["model_initialization_attempts"] == model["model_initializations"] == 1
        and model["model_requests_attempted"] == model["model_requests_completed"] == model["responses_sent"] == 1
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 35
        and len(model["denoise_evidence"]) == 35
        and all(call["completed"] is True for call in model["denoise_evidence"])
        and model["physics_steps"] == model["fresh_cases"] == 0
        and model["automatic_retry"] is False
        and model["model_sequence_complete"] is True
        and len(model["responses"]) == 1
        and model["responses"][0]["reference_sent"] is True,
        "Exactly one complete registered model exchange without physics is required",
    )
    engine.validate_mask_evidence(model["mask_evidence"], 1)
    exchange = client["exchange"]
    frame, envelope = exchange["request"], exchange["response"]
    row = model["responses"][0]
    request = prefix.read(row["model_request_ref"])
    response = prefix.read(row["actual_response_ref"])
    contract.validate_response(
        response,
        model_request=request,
        task=config["task"],
        checkpoint=inputs["prepared"]["model_config"]["checkpoint"],
        seed=43,
    )
    arrays = repro.response_arrays(response)
    expected_snapshot = inputs["initial"]
    identity_keys = ("version", "session", "request_id", "step", "sim_time_s", "state_sha256", "seed", "model_sha256")
    prefix.require(
        all(type(frame[key]) is type(envelope[key]) and frame[key] == envelope[key] for key in identity_keys)
        and frame["session"] == config["session"]
        and frame["request_id"] == 0
        and frame["step"] == row["step"] == expected_snapshot["step"]
        and frame["sim_time_s"] == expected_snapshot["sim_time_s"]
        and frame["state_sha256"] == expected_snapshot["state_sha256"]
        and frame["seed"] == envelope["seed"] == request["seed"] == 43
        and frame["task"] == request["prompt"] == config["task"]
        and frame["model_sha256"] == envelope["model_sha256"] == config["model_sha256"]
        and frame["image"]["path"] == "current.png"
        and frame["image"]["sha256"] == envelope["image_sha256"] == config["image_sha256"]
        and Path(request["vision_path"]).resolve().is_relative_to(admission.OUTPUT / "producer/model")
        and repro.sha256(request["vision_path"]) == config["image_sha256"]
        and envelope["denoise_calls"] == 35
        and envelope["action_dim_mask"] == [True] * 10 + [False] * 118
        and np.array_equal(np.asarray(envelope["action"]), arrays["action"])
        and np.array_equal(np.asarray(envelope["normalized_action"]), arrays["normalized_action"])
        and exchange["accepted_requests"] == 1
        and exchange["completed_periods"]
        == exchange["physics_steps"]
        == exchange["native_models"]
        == exchange["action_dispatches"]
        == 0
        and exchange["snapshot_scope"] == "historical_recorded_endpoint_not_live",
        "The client envelope must match the recorded image and actual model arrays",
    )
    comparison = repro.compare_responses(response, inputs["previous_response"])
    prefix.require(comparison == exchange["compared_to_saved_seed42"], "Independent seed-42 comparison differs")
    return dict(
        sampling_seed=43,
        task=config["task"],
        image_sha256=config["image_sha256"],
        recorded_initial=copy.deepcopy(expected_snapshot),
        actual_model_request_ref=row["model_request_ref"],
        actual_model_response_ref=row["actual_response_ref"],
        model_arrays_match_client_envelope=True,
        compared_to_saved_seed42=comparison,
    )


def execute():
    """Run one pair of owned workers, preserving interruption and cost bounds."""
    prefix.environment()
    trial = make_trial()
    inputs = admission.load_inputs()
    trial.require_config(inputs)
    before, sources = trial.check_cpu_admission(inputs), source_refs()
    trial.check_preflight(inputs)
    directory = admission.OUTPUT / "producer"
    directory.mkdir(exist_ok=False)
    client_dir, model_dir = directory / "simulation", directory / "model"
    client_dir.mkdir()
    model_dir.mkdir()
    result = dict(
        complete=False,
        passed=False,
        evidence_level="saved_development_image_model_replay",
        control_qualified=False,
        gpu_uuid=admission.GPU,
        physics_steps=0,
        native_models=0,
        action_dispatches=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    lifecycle, model = None, None
    launch_attempted = False
    try:
        prefix.checked(inputs["refs"]["historical_image"])
        with (
            Path(inputs["refs"]["historical_image"]["path"]).open("rb") as source,
            (client_dir / "current.png").open("xb") as target,
        ):
            shutil.copyfileobj(source, target)
        prefix.checked(inputs["refs"]["historical_image"])
        prefix.require(repro.sha256(client_dir / "current.png") == inputs["config"]["image_sha256"], "Copied image SHA")
        bundle_ref = publish(model_dir / "registered_inputs.json", trial.model_bundle(inputs))
        result["registration_ref"] = publish(
            directory / "registration.json",
            dict(
                phase="producer",
                inputs=admission.input_receipt(inputs),
                source_refs=sources,
                model_bundle_ref=bundle_ref,
            ),
        )
        result["gpu_before"] = registered.gpu_observation(admission.GPU)
        prefix.require(
            result["gpu_before"]["memory_mib"] == 0 and result["gpu_before"]["compute_apps"] == [],
            "The registered GPU must be idle before model launch",
        )
        client_env = physical._cpu_environment(client_dir)
        model_env = model_owner._model_environment(model_dir)
        model_env["CUDA_VISIBLE_DEVICES"] = admission.GPU
        specs = {
            "simulation": dict(
                argv=[sys.executable, "-m", MODULE, "--worker", "replay"],
                env=client_env,
                log=str((directory / "replay.log").relative_to(admission.ROOT)),
            ),
            "model": dict(
                argv=[inputs["prepared"]["model_config"]["model_python"], "-m", MODULE, "--worker", "model"],
                env=model_env,
                log=str((directory / "model.log").relative_to(admission.ROOT)),
            ),
        }
        launch_attempted = True
        try:
            lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
        except OwnedSocketPairInterrupted as error:
            lifecycle = error.result
            result["interrupted"] = True
            raise
        finally:
            if lifecycle is not None:
                result["workers_ref"] = publish(directory / "workers.json", historical._record_worker_logs(lifecycle))
        prefix.require(lifecycle["passed"] is True, "Both replay and model workers must exit and be reaped")
        result["client_ref"] = prefix.reference(client_dir / "result.json")
        result["model_ref"] = prefix.reference(model_dir / "result.json")
        client, model = prefix.read(result["client_ref"]), prefix.read(result["model_ref"])
        result["audit"] = audit_exchange(inputs, client, model)
        prefix.require(
            source_refs() == sources and snapshot(admission.ROOT)["sources"] == before, "Replay sources changed"
        )
        for ref in inputs["refs"].values():
            prefix.checked(ref)
        result["gpu_after"] = registered.gpu_observation(admission.GPU)
        result["gpu_released"] = result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
        prefix.require(result["gpu_released"] is True, "The registered GPU must be released")
        result.update(complete=True, passed=True, sources_unchanged=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            sources_match = source_refs() == sources
            model_path = model_dir / "result.json"
            if model_path.is_file():
                result["available_model_ref"] = prefix.reference(model_path)
                model = prefix.read(result["available_model_ref"])
            result["model_cost_bounds"] = model_cost_bounds(
                lifecycle, model, launch_attempted=launch_attempted, sources_match=sources_match
            )
        except Exception as error:
            result.update(
                complete=False,
                passed=False,
                cost_accounting_error={"type": type(error).__name__, "message": str(error)},
                model_cost_bounds=model_cost_bounds(None, None, launch_attempted=launch_attempted, sources_match=False),
            )
        if "gpu_after" not in result:
            try:
                result["gpu_after"] = registered.gpu_observation(admission.GPU)
                result["gpu_released"] = (
                    result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
                )
            except Exception as error:
                result["gpu_after_error"] = {"type": type(error).__name__, "message": str(error)}
        print(publish(directory / "result.json", result), flush=True)
        if "cost_accounting_error" in result:
            raise ValueError("Model replay cost accounting failed")
