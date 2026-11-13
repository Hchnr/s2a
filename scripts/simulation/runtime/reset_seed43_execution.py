"""Own the registered reset producer and its independent CPU replay."""

import sys

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as historical
from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_costs as costs
from scripts.simulation.runtime import reset_seed43_feedback_audit as audit
from scripts.simulation.runtime import reset_seed43_supervision as supervision
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner


def require_limits(config):
    """Check the preregistered work bounds before starting either phase."""
    expected = {
        "first_window_steps": 8,
        "live_window_steps": 8,
        "max_control_steps_per_phase": 16,
        "max_native_steps_per_phase": 400,
        "max_models_per_phase": 3,
        "max_contexts_per_phase": 3,
        "max_model_initializations": 1,
        "max_model_requests": 1,
        "denoise_calls_per_request": 35,
        "fresh_cases": 0,
        "fits": 0,
        "automatic_retry": False,
    }
    prefix.require(
        all(type(config.get(key)) is type(value) and config[key] == value for key, value in expected.items())
        and type(config.get("worker_timeout_seconds")) is int
        and config["worker_timeout_seconds"] > 0,
        "The exact bounded two-window experiment is required",
    )


def execute(trial, phase):
    """Run a producer or checker once, retaining partial work on failure."""
    prefix.environment()
    prefix.require(isinstance(trial, registered.RegisteredAdaptiveTrial), "A registered trial is required")
    prefix.require(phase in ("producer", "checker"), "A producer or checker phase is required")
    inputs = trial.load_inputs()
    trial.require_config(inputs)
    require_limits(inputs["config"])
    before, sources = trial.check_cpu_admission(inputs), trial.source_refs()
    trial.check_preflight(inputs)
    prior_physical, prior_model = costs.zero_bounds()
    previous = None
    if phase == "checker":
        previous = prefix.read(prefix.reference(trial.output / "producer/result.json"))
        prior_physical, prior_model = supervision.verify_closed_producer(trial, inputs, previous, sources)

    directory = trial.output / phase
    directory.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        control_qualified=False,
        gpu_uuid=trial.gpu_uuid,
        automatic_retry=False,
    )
    lifecycle = None
    try:
        simulation_dir = directory / "simulation"
        simulation_dir.mkdir()
        model_ref = None
        if phase == "producer":
            model_dir = directory / "model"
            model_dir.mkdir()
            model_ref = publish(model_dir / "registered_inputs.json", trial.model_bundle(inputs))
        result["registration_ref"] = publish(
            directory / "registration.json",
            dict(
                phase=phase,
                inputs=trial.input_receipt(inputs),
                source_refs=sources,
                model_bundle_ref=model_ref,
                producer_branch_ref=None if previous is None else previous["branch_ref"],
                producer_model_ref=None if previous is None else previous["model_ref"],
            ),
        )
        simulation_env = physical._cpu_environment(simulation_dir)
        command = [sys.executable, "-m", trial.module]
        if phase == "producer":
            result["gpu_before"] = registered.gpu_observation(trial.gpu_uuid)
            prefix.require(
                result["gpu_before"]["memory_mib"] == 0 and result["gpu_before"]["compute_apps"] == [],
                "The registered GPU must be idle before producer startup",
            )
            model_env = model_owner._model_environment(model_dir)
            model_env["CUDA_VISIBLE_DEVICES"] = trial.gpu_uuid
            specs = {
                "simulation": dict(
                    argv=command + ["--worker", "producer"],
                    env=simulation_env,
                    log=str((directory / "simulation.log").relative_to(prefix.REPO)),
                ),
                "model": dict(
                    argv=[inputs["prepared"]["model_config"]["model_python"], "-m", trial.module, "--worker", "model"],
                    env=model_env,
                    log=str((directory / "model.log").relative_to(prefix.REPO)),
                ),
            }
            try:
                lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
            except OwnedSocketPairInterrupted as error:
                lifecycle = error.result
                result["workers_ref"] = publish(directory / "workers.json", historical._record_worker_logs(lifecycle))
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", historical._record_worker_logs(lifecycle))
            prefix.require(lifecycle["passed"] is True, "Both model and simulation workers must exit and be reaped")
            result["model_ref"] = prefix.reference(model_dir / "result.json")
            model = prefix.read(result["model_ref"])
            accounting.require_completed_model(model, inputs["config"])
            result["gpu_after"] = registered.gpu_observation(trial.gpu_uuid)
            result["gpu_released"] = (
                result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
            )
            prefix.require(result["gpu_released"] is True, "The registered GPU must be released")
        else:
            spec = {"key": "checker", "argv": command + ["--worker", "checker"], "log": directory / "checker.log"}
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [spec], cwd=prefix.REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", historical._record_worker_logs(lifecycle))
            lifecycle_check.require_closed_workers(lifecycle, ["checker"])
            result["gpu_released"] = True

        branch_ref = prefix.reference(simulation_dir / "result.json")
        branch = prefix.read(branch_ref)
        producer = None if previous is None else prefix.read(previous["branch_ref"])
        counts = audit.summarize_branch(branch, phase, inputs["config"], producer=producer)
        if phase == "producer":
            result["model_exchange"] = audit.verify_model_exchange(
                branch, model, trial.model_bundle(inputs), inputs["case"]["tool"]
            )
            prefix.require(
                counts["model_requests"] == result["model_exchange"]["requests"] == model["responses_sent"],
                "Producer physics and actual model exchanges must agree",
            )
        else:
            counts["independent_max_absolute_difference"] = audit.compare_branches(producer, branch)
            prefix.require(counts["control_steps"] == previous["counts"]["control_steps"], "Replay count differs")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            trial.source_refs() == sources and snapshot(prefix.REPO)["sources"] == before, "Trial sources changed"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            costs.phase_costs(
                trial,
                result,
                phase=phase,
                directory=directory,
                inputs=inputs,
                sources=sources,
                lifecycle=lifecycle,
                prior_physical=prior_physical,
                prior_model=prior_model,
            )
            if result["complete"] is True:
                accounting.require_complete_phase_costs(result)
        except Exception as error:
            result.update(
                complete=False,
                passed=False,
                cost_accounting_error={"type": type(error).__name__, "message": str(error)},
            )
        if phase == "producer" and "gpu_after" not in result:
            try:
                result["gpu_after"] = registered.gpu_observation(trial.gpu_uuid)
                result["gpu_released"] = (
                    result["gpu_after"]["memory_mib"] == 0 and result["gpu_after"]["compute_apps"] == []
                )
            except Exception as error:
                result["gpu_after_error"] = {"type": type(error).__name__, "message": str(error)}
        result_ref = publish(directory / "result.json", result)
        print(result_ref, flush=True)
        if "cost_accounting_error" in result:
            raise ValueError("Reset trial cost accounting failed")
    return result_ref
