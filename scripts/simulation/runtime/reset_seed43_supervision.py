"""Verify a reaped reset producer before admitting its independent checker."""

from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_costs as costs
from scripts.simulation.runtime import reset_seed43_feedback_audit as audit


def verify_closed_producer(trial, inputs, result, sources):
    """Recompute inputs, lifecycle, model exchange, physical costs and counts."""
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
        "The producer, source identities, model and GPU must be fully closed",
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
        "Producer registration, bundle and reaped worker lifecycle must match",
    )
    for row in lifecycle["workers"]:
        prefix.checked(row["log_ref"])
    producer, model = prefix.read(result["branch_ref"]), prefix.read(result["model_ref"])
    accounting.require_completed_model(model, inputs["config"])
    counts = audit.summarize_branch(producer, "producer", inputs["config"])
    exchange = audit.verify_model_exchange(producer, model, bundle, inputs["case"]["tool"])
    prefix.require(
        counts == result["counts"]
        and exchange == result["model_exchange"]
        and counts["model_requests"] == exchange["requests"] == model["responses_sent"],
        "The physical and model receipts must match the published producer result",
    )
    prior_physical, prior_model = costs.zero_bounds()
    recomputed = dict(phase="producer", counts=counts)
    costs.phase_costs(
        trial,
        recomputed,
        phase="producer",
        directory=directory,
        inputs=inputs,
        sources=sources,
        lifecycle=lifecycle,
        prior_physical=prior_physical,
        prior_model=prior_model,
    )
    accounting.require_complete_phase_costs(recomputed)
    keys = ("phase_physical_bounds", "cumulative_physical_bounds", "phase_model_bounds", "cumulative_model_bounds")
    prefix.require(
        all(result[key] == recomputed[key] for key in keys),
        "The original producer costs must remain exact and unchanged",
    )
    return result["cumulative_physical_bounds"], result["cumulative_model_bounds"]
