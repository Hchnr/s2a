"""Own and audit one case of the shared reset-origin adaptive matrix."""

import sys

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as historical
from scripts.simulation.runtime import registered_adaptive_execution as common
from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_audit as audit
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner


def verify_closed_producer(trial, inputs, result, sources):
    """Recheck the actual model, physical receipts and reaped workers before replay."""
    directory = trial.output / "producer"
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is result["gpu_released"] is True
        and result["phase"] == "producer"
        and result["registration_ref"] == prefix.reference(directory / "registration.json")
        and result["workers_ref"] == prefix.reference(directory / "workers.json")
        and result["branch_ref"]
        == result["available_branch_ref"]
        == prefix.reference(directory / "simulation/result.json")
        and result["model_ref"] == result["available_model_ref"] == prefix.reference(directory / "model/result.json")
        and result["gpu_after"]["memory_mib"] == 0
        and result["gpu_after"]["compute_apps"] == [],
        "The matching producer and GPU must be fully closed",
    )
    registration = prefix.read(result["registration_ref"])
    bundle = prefix.read(registration["model_bundle_ref"])
    lifecycle = prefix.read(result["workers_ref"])
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == trial.input_receipt(inputs)
        and registration["source_refs"] == sources == trial.source_refs()
        and registration["model_bundle_ref"] == prefix.reference(directory / "model/registered_inputs.json")
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
        "The producer inputs and reaped worker lifecycle must remain exact",
    )
    for row in lifecycle["workers"]:
        prefix.checked(row["log_ref"])
    branch, model = prefix.read(result["branch_ref"]), prefix.read(result["model_ref"])
    accounting.require_completed_model(model, inputs["config"])
    counts = audit.summarize_branch(branch, "producer", inputs["config"])
    exchange = audit.verify_model_exchange(branch, model, bundle)
    prefix.require(
        counts == result["counts"]
        and exchange == result["model_exchange"]
        and counts["model_requests"] == exchange["requests"] == model["responses_sent"],
        "Producer feedback and every actual model exchange must agree",
    )
    recomputed = dict(phase="producer", counts=counts)
    common.phase_costs(
        trial,
        recomputed,
        phase="producer",
        directory=directory,
        inputs=inputs,
        sources=sources,
        lifecycle=lifecycle,
        prior_physical=inputs["previous_physical_bounds"],
        prior_model=inputs["previous_model_bounds"],
    )
    accounting.require_complete_phase_costs(recomputed)
    keys = ("phase_physical_bounds", "cumulative_physical_bounds", "phase_model_bounds", "cumulative_model_bounds")
    prefix.require(all(result[key] == recomputed[key] for key in keys), "Producer costs must remain exact")
    return result["cumulative_physical_bounds"], result["cumulative_model_bounds"]


def execute(trial, phase):
    """Run one registered phase once, retaining actual or bounded partial costs."""
    prefix.environment()
    prefix.require(isinstance(trial, registered.RegisteredAdaptiveTrial), "A registered trial is required")
    prefix.require(phase in ("producer", "checker"), "One registered phase is required")
    inputs = trial.load_inputs()
    trial.require_config(inputs)
    common.require_limits(inputs)
    before, sources = trial.check_cpu_admission(inputs), trial.source_refs()
    trial.check_preflight(inputs)
    prior_physical, prior_model = inputs["previous_physical_bounds"], inputs["previous_model_bounds"]
    previous = None
    if phase == "checker":
        previous = prefix.read(prefix.reference(trial.output / "producer/result.json"))
        prior_physical, prior_model = verify_closed_producer(trial, inputs, previous, sources)

    directory = trial.output / phase
    directory.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        case_index=inputs["config"]["case_index"],
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
        index = str(inputs["config"]["case_index"])
        command = [sys.executable, "-m", trial.module, "--case-index", index]
        if phase == "producer":
            result["gpu_before"] = registered.gpu_observation(trial.gpu_uuid)
            prefix.require(
                result["gpu_before"]["memory_mib"] == 0 and result["gpu_before"]["compute_apps"] == [],
                "The registered GPU must be idle before the case starts",
            )
            model_env = model_owner._model_environment(model_dir)
            model_env["CUDA_VISIBLE_DEVICES"] = trial.gpu_uuid
            model_python = inputs["prepared"]["model_config"]["model_python"]
            specs = {
                "simulation": dict(
                    argv=command + ["--worker", "producer"],
                    env=simulation_env,
                    log=str((directory / "simulation.log").relative_to(prefix.REPO)),
                ),
                "model": dict(
                    argv=[model_python, "-m", trial.module, "--case-index", index, "--worker", "model"],
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
            prefix.require(lifecycle["passed"] is True, "The model and simulator must exit and be reaped")
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
        counts = audit.summarize_branch(branch, phase, inputs["config"])
        if phase == "producer":
            result["model_exchange"] = audit.verify_model_exchange(branch, model, trial.model_bundle(inputs))
            prefix.require(
                counts["model_requests"] == result["model_exchange"]["requests"] == model["responses_sent"],
                "Physical feedback must use each actual model response exactly once",
            )
        else:
            producer = prefix.read(previous["branch_ref"])
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
            common.phase_costs(
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
            raise ValueError("Adaptive development cost accounting failed")
    return result_ref
