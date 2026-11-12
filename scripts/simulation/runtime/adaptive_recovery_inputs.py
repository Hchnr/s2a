"""Revalidate adaptive development inputs and the incomplete prior attempt."""

import numpy as np

from s2a.research.native_runtime.physical_cost_bounds import physical_cost_bounds
from scripts.simulation.runtime import adaptive_live_bowl_admission as historical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_replan_step107_execution as execution
from scripts.simulation.runtime import step107_model_inputs
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

FAILURE = {
    "path": str(historical.OUTPUT / "producer/result.json"),
    "sha256": "71db04713a43b74d97f0792c4a6d09b96270b48ba336affb8d85bf7c1d145776",
}


def load_recorded_inputs():
    """Reproduce the original input bundle without binding model globals."""
    prefix.environment()
    refs = {
        "config": {"path": str(historical.CONFIG), "sha256": historical.CONFIG_SHA},
        "plan": {"path": str(historical.PLAN), "sha256": historical.PLAN_SHA},
        "prior_checker": historical.PRIOR_CHECKER,
    }
    config = prefix.read(refs["config"])
    historical.validate_config(config)
    prefix.checked(refs["plan"])
    for name in ("prior_capture", "prior_model", "prior_execution"):
        refs[name] = {"path": str(prefix.REPO / config[name]), "sha256": config[name + "_sha256"]}
    capture, old_model, old_branch, old_checker = (
        prefix.read(refs[name]) for name in ("prior_capture", "prior_model", "prior_execution", "prior_checker")
    )
    counts = old_checker["counts"]
    prefix.require(
        all(
            item.get("complete") is item.get("passed") is True for item in (capture, old_model, old_branch, old_checker)
        )
        and old_model["gpu_released"] is True
        and old_branch["case"]["case_id"] == config["case_id"]
        and all(value == 0 for value in counts["independent_max_absolute_difference"].values())
        and counts["control_steps"] == 113
        and counts["cumulative_control_steps"] == config["previous_control_steps"]
        and counts["cumulative_native_steps"] == config["previous_native_steps"]
        and counts["cumulative_models"] == config["previous_models"]
        and counts["cumulative_contexts"] == config["previous_contexts"]
        and old_model["model_initializations"] == old_model["model_requests_completed"] == 1
        and old_model["denoise_calls_completed"] == 35,
        "The closed prior producer, checker, model and cumulative costs are required",
    )
    prior = execution.load_inputs()
    model_inputs = step107_model_inputs.load_inputs()
    initial, expected_capture, response = (
        model_inputs["initial"],
        capture["terminal_capture"]["capture"],
        prior["envelope"],
    )
    prefix.require(
        initial == capture["terminal_capture"]["snapshot"]
        and initial["step"] == config["prefix_steps"]
        and initial["state_sha256"] == config["initial_state_sha256"]
        and expected_capture["canvas_png"]["sha256"] == config["initial_image_sha256"]
        and response["step"] == initial["step"]
        and response["sim_time_s"] == initial["sim_time_s"]
        and response["state_sha256"] == initial["state_sha256"]
        and response["image_sha256"] == config["initial_image_sha256"]
        and response["model_sha256"] == config["model_sha256"]
        and response["seed"] == config["seed"]
        and response["denoise_calls"] == config["denoise_calls_per_request"]
        and len(model_inputs["weight_hashes"]) == 163
        and len(model_inputs["model_source_hashes"]) >= 100,
        "The exact first current state, image and historical response are required",
    )
    arrays = repro.response_arrays(prior["raw"])
    np.testing.assert_array_equal(arrays["action"], response["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], response["normalized_action"])
    consumed = dict(prior["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        config=config,
        refs=refs,
        prior=prior,
        prior_branch=old_branch,
        prior_checker=old_checker,
        initial=initial,
        expected_capture=expected_capture,
        expected_response=response,
        prepared=model_inputs["prepared"],
        bridge=model_inputs["bridge"],
        template=model_inputs["template"],
        model_source_hashes=model_inputs["model_source_hashes"],
        weight_hashes=model_inputs["weight_hashes"],
        source_sha256=consumed,
    )


def load_failed_attempt(config):
    """Bind the prior failed workers and retain physical uncertainty as ranges."""
    prefix.environment()
    historical.validate_config(config)
    failure = prefix.read(FAILURE)
    workers = prefix.read(failure["workers_ref"])
    model = prefix.read(failure["available_model_ref"])
    by_name = {row["name"]: row for row in workers["workers"]}
    prefix.require(
        failure["complete"] is failure["passed"] is False
        and failure["phase"] == "producer"
        and failure["gpu_uuid"] == model["gpu_uuid"] == config["gpu_uuid"]
        and failure["gpu_released"] is True
        and "available_branch_ref" not in failure
        and workers["passed"] is False
        and workers["launch_error"] is None
        and workers["timed_out"] is workers["interrupted"] is False
        and len(workers["workers"]) == len(by_name) == 2
        and set(by_name) == {"simulation", "model"}
        and all(row["actual_wait"] is row["reaped"] is True for row in by_name.values())
        and by_name["simulation"]["returncode"] == -15
        and by_name["simulation"]["signal_sent"] is True
        and by_name["model"]["returncode"] == 1
        and by_name["model"]["signal_sent"] is False
        and model["complete"] is model["passed"] is False
        and model["error"]
        == {"type": "ValueError", "message": "The model worker requires the exact preflight and registered bundle"}
        and all(
            model[key] == failure[key] == 0
            for key in (
                "model_initialization_attempts",
                "model_initializations",
                "model_requests_attempted",
                "model_requests_completed",
                "denoise_calls_attempted",
                "denoise_calls_completed",
            )
        )
        and not (historical.OUTPUT / "producer/simulation/result.json").exists()
        and not (historical.OUTPUT / "checker").exists(),
        "The exact failed worker lifecycle and missing simulation receipt are required",
    )
    for row in workers["workers"]:
        prefix.checked(row["log_ref"])
    limits = dict(
        control_steps=config["max_control_steps_per_phase"],
        native_steps=config["max_native_steps_per_phase"],
        models=config["max_models_per_phase"],
        contexts=config["max_contexts_per_phase"],
    )
    unknown = physical_cost_bounds(None, limits=limits, receipt_final=False)
    known = dict(
        control_steps=config["previous_control_steps"],
        native_steps=config["previous_native_steps"],
        models=config["previous_models"],
        contexts=config["previous_contexts"],
    )
    cumulative = {
        key: dict(lower=known[key] + value["lower"], upper=known[key] + value["upper"])
        for key, value in unknown.items()
    }
    return dict(
        failure_ref=FAILURE,
        workers_ref=failure["workers_ref"],
        model_ref=failure["available_model_ref"],
        simulation_bounds=unknown,
        cumulative_physical_bounds=cumulative,
        previous_static_models=config["previous_static_models"],
        previous_model_initializations=config["previous_model_initializations"],
        previous_requests_attempted=config["previous_requests_attempted"],
        previous_requests_completed=config["previous_requests_completed"],
        previous_denoises=config["previous_denoises"],
    )
