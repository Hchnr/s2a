"""Own adaptive trial workers and preserve actual or bounded partial costs."""

import sys

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.physical_cost_bounds import physical_cost_bounds
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as audit
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as historical
from scripts.simulation.runtime import adaptive_live_bowl_worker as simulation
from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner


def require_limits(inputs):
    """Reject inconsistent or unbounded work before creating a phase directory."""
    config = inputs["config"]
    keys = (
        "prefix_steps",
        "max_adaptive_periods",
        "max_control_steps_per_phase",
        "max_native_steps_per_phase",
        "max_models_per_phase",
        "max_contexts_per_phase",
        "max_model_initializations",
        "max_model_requests",
        "denoise_calls_per_request",
        "max_denoise_calls",
        "worker_timeout_seconds",
    )
    prefix.require(
        all(type(config.get(key)) is int for key in keys)
        and config["prefix_steps"] >= 0
        and config["max_adaptive_periods"] >= 1
        and config["prefix_steps"] + config["max_adaptive_periods"] == config["max_control_steps_per_phase"] <= 600
        and config["max_native_steps_per_phase"] == 25 * config["max_control_steps_per_phase"]
        and config["max_models_per_phase"] > 0
        and config["max_contexts_per_phase"] > 0
        and config["max_model_initializations"] == 1
        and 1 <= config["max_model_requests"] <= 75
        and config["denoise_calls_per_request"] == 35
        and config["max_denoise_calls"] == 35 * config["max_model_requests"]
        and config["worker_timeout_seconds"] > 0,
        "Explicit consistent physical and model work limits are required",
    )
    zero_physical = {key: 0 for key in ("control_steps", "native_steps", "models", "contexts")}
    audit.add_cost_bounds(inputs["previous_physical_bounds"], audit.exact_cost_bounds(zero_physical))
    zero_model = dict(
        model_initialization_attempts=0,
        model_initializations=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        denoise_calls_attempted=0,
        denoise_calls_completed=0,
    )
    accounting.add_model_cost_bounds(
        inputs["previous_model_bounds"], accounting.model_cost_bounds(zero_model, config=config, receipt_final=True)
    )


def phase_costs(trial, result, *, phase, directory, inputs, sources, lifecycle, prior_physical, prior_model):
    """Keep full uncertainty when a source or final worker receipt is missing."""
    source_match = False
    try:
        source_match = trial.source_refs() == sources
    except Exception as error:
        result["partial_source_check_error"] = {"type": type(error).__name__, "message": str(error)}
    result["partial_source_match"] = source_match
    branch = model = None
    branch_path, model_path = directory / "simulation/result.json", directory / "model/result.json"
    if branch_path.is_file():
        try:
            result["available_branch_ref"] = prefix.reference(branch_path)
            branch = prefix.read(result["available_branch_ref"])
        except Exception as error:
            result["branch_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
    if phase == "producer" and model_path.is_file():
        try:
            result["available_model_ref"] = prefix.reference(model_path)
            model = prefix.read(result["available_model_ref"])
        except Exception as error:
            result["model_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
    config = inputs["config"]
    limits = dict(
        control_steps=config["max_control_steps_per_phase"],
        native_steps=config["max_native_steps_per_phase"],
        models=config["max_models_per_phase"],
        contexts=config["max_contexts_per_phase"],
    )
    worker_name = "simulation" if phase == "producer" else "checker"
    branch_final = source_match and accounting._closed_worker(lifecycle, worker_name) and type(branch) is dict
    try:
        physical_bounds = physical_cost_bounds(
            branch.get("runtime") if branch_final else None, limits=limits, receipt_final=branch_final
        )
    except (TypeError, ValueError, KeyError) as error:
        result["physical_receipt_error"] = {"type": type(error).__name__, "message": str(error)}
        physical_bounds = physical_cost_bounds(None, limits=limits, receipt_final=False)
    if phase == "checker":
        model = dict(
            model_initialization_attempts=0,
            model_initializations=0,
            model_requests_attempted=0,
            model_requests_completed=0,
            denoise_calls_attempted=0,
            denoise_calls_completed=0,
        )
        model_final = True
    else:
        model_final = source_match and accounting._closed_worker(lifecycle, "model") and type(model) is dict
        if not model_final:
            model = None
    try:
        model_bounds = accounting.model_cost_bounds(model, config=config, receipt_final=model_final)
    except (TypeError, ValueError, KeyError) as error:
        result["model_counter_error"] = {"type": type(error).__name__, "message": str(error)}
        model_bounds = accounting.model_cost_bounds(None, config=config, receipt_final=False)
    result.update(
        phase_physical_bounds=physical_bounds,
        cumulative_physical_bounds=audit.add_cost_bounds(prior_physical, physical_bounds),
        phase_model_bounds=model_bounds,
        cumulative_model_bounds=accounting.add_model_cost_bounds(prior_model, model_bounds),
    )


def verified_producer_bounds(trial, inputs, result, sources):
    """Recompute producer evidence before an independent checker is dispatched."""
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is result["gpu_released"] is True
        and result["phase"] == "producer"
        and result["branch_ref"] == result["available_branch_ref"]
        and result["model_ref"] == result["available_model_ref"],
        "A completed producer with released GPU and final receipts is required",
    )
    registration = prefix.read(result["registration_ref"])
    bundle = prefix.read(registration["model_bundle_ref"])
    lifecycle = prefix.read(result["workers_ref"])
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == trial.input_receipt(inputs)
        and registration["source_refs"] == sources
        and bundle == trial.model_bundle(inputs)
        and lifecycle["passed"] is True
        and lifecycle["launch_error"] is None
        and lifecycle["timed_out"] is lifecycle["interrupted"] is False
        and len(lifecycle["workers"]) == 2
        and {row["name"] for row in lifecycle["workers"]} == {"simulation", "model"}
        and all(
            row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0 and row["signal_sent"] is False
            for row in lifecycle["workers"]
        ),
        "Producer inputs, sources and reaped worker lifecycle must remain exact",
    )
    for row in lifecycle["workers"]:
        prefix.checked(row["log_ref"])
    producer, model = prefix.read(result["branch_ref"]), prefix.read(result["model_ref"])
    accounting.require_completed_model(model, inputs["config"])
    simulation.load_recorded_responses(producer, model)
    counts = audit.summarize_branch(producer, "producer", inputs["config"])
    prefix.require(
        counts == result["counts"]
        and counts["adaptive_steps"] == producer["runtime"]["continuation"]["adaptive"]["control_steps_completed"]
        and model["responses_sent"] == len(producer["runtime"]["adaptive_model_exchanges"])
        and (
            trial.compare_previous is None or result["previous_comparison"] == trial.compare_previous(inputs, producer)
        ),
        "Producer receipts, comparison and completed model exchanges must agree",
    )
    physical_bounds = audit.add_cost_bounds(inputs["previous_physical_bounds"], audit.exact_cost_bounds(counts))
    model_bounds = accounting.add_model_cost_bounds(
        inputs["previous_model_bounds"],
        accounting.model_cost_bounds(model, config=inputs["config"], receipt_final=True),
    )
    accounting.require_complete_phase_costs(result)
    prefix.require(
        result["cumulative_physical_bounds"] == physical_bounds and result["cumulative_model_bounds"] == model_bounds,
        "Producer cumulative ranges must preserve all prior work",
    )
    return physical_bounds, model_bounds


def execute(trial, phase):
    """Run one registered producer or its independent replay without retries."""
    prefix.environment()
    prefix.require(isinstance(trial, registered.RegisteredAdaptiveTrial), "A registered adaptive trial is required")
    prefix.require(phase in ("producer", "checker"), "A registered physical phase is required")
    inputs = trial.load_inputs()
    trial.require_config(inputs)
    require_limits(inputs)
    before, sources = trial.check_cpu_admission(inputs), trial.source_refs()
    trial.check_preflight(inputs)
    previous_result = None
    prior_physical, prior_model = inputs["previous_physical_bounds"], inputs["previous_model_bounds"]
    if phase == "checker":
        previous_result = prefix.read(prefix.reference(trial.output / "producer/result.json"))
        prior_physical, prior_model = verified_producer_bounds(trial, inputs, previous_result, sources)
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
                producer_branch_ref=None if previous_result is None else previous_result["branch_ref"],
                producer_model_ref=None if previous_result is None else previous_result["model_ref"],
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
            accounting.require_completed_model(prefix.read(result["model_ref"]), inputs["config"])
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
        counts = audit.summarize_branch(branch, phase, inputs["config"])
        if phase == "producer":
            model = prefix.read(result["model_ref"])
            simulation.load_recorded_responses(branch, model)
            prefix.require(
                counts["adaptive_steps"] == branch["runtime"]["continuation"]["adaptive"]["control_steps_completed"]
                and model["responses_sent"] == len(branch["runtime"]["adaptive_model_exchanges"]),
                "Producer physics and accepted model exchanges must agree",
            )
            if trial.compare_previous is not None:
                result["previous_comparison"] = trial.compare_previous(inputs, branch)
        else:
            producer = prefix.read(previous_result["branch_ref"])
            counts["independent_max_absolute_difference"] = audit.compare_branches(producer, branch)
            prefix.require(
                counts["control_steps"] == previous_result["counts"]["control_steps"], "Replay count differs"
            )
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
            phase_costs(
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
            raise ValueError("Adaptive trial cost accounting failed")
    return result_ref
