"""Preserve exact or bounded resource costs for the registered reset trial."""

from scripts.simulation.runtime import registered_adaptive_execution as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

PHYSICAL_FIELDS = ("control_steps", "native_steps", "models", "contexts")
MODEL_FIELDS = ("initializations_completed", "requests_attempted", "requests_completed", "denoise_calls_completed")


def zero_bounds():
    """Start this trial's own cumulative ledger at zero, without rewriting history."""
    return (
        {name: dict(lower=0, upper=0) for name in PHYSICAL_FIELDS},
        {name: dict(lower=0, upper=0) for name in MODEL_FIELDS},
    )


def phase_costs(trial, result, *, phase, directory, inputs, sources, lifecycle, prior_physical, prior_model):
    """Use final reaped receipts, retaining registered upper bounds when absent."""
    config = inputs["config"]
    fields = (
        "max_control_steps_per_phase",
        "max_native_steps_per_phase",
        "max_models_per_phase",
        "max_contexts_per_phase",
        "max_model_initializations",
        "max_model_requests",
        "denoise_calls_per_request",
    )
    prefix.require(
        phase in ("producer", "checker")
        and all(type(config.get(key)) is int for key in fields)
        and config["max_control_steps_per_phase"] == 16
        and config["max_native_steps_per_phase"] == 400
        and config["max_models_per_phase"] == config["max_contexts_per_phase"] == 3
        and config["max_model_initializations"] == config["max_model_requests"] == 1
        and config["denoise_calls_per_request"] == 35,
        "The fixed two-window physical and model budgets are required",
    )
    adapted = dict(inputs, config=dict(config, max_denoise_calls=35))
    registered.phase_costs(
        trial,
        result,
        phase=phase,
        directory=directory,
        inputs=adapted,
        sources=sources,
        lifecycle=lifecycle,
        prior_physical=prior_physical,
        prior_model=prior_model,
    )
