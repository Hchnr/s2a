"""Admit the frozen short live-adaptive bowl trial and its historical inputs."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_replan_step107_execution as execution
from scripts.simulation.runtime import run_causal_replan_step107_model as model
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/adaptive_live_bowl_20261112.json"
CONFIG_SHA = "8b7dd9f5cadf360114ba1b047071eca970f7832ea1f959694697404f829505f5"
PLAN = ROOT / "plans/v0.0.1/tracking/references/adaptive_live_bowl_20261112.md"
PLAN_SHA = "83bc8b6b348ee80604f879169bc36d2a41318ba1b9e5e01c808eeb58c68e597f"
OUTPUT = ROOT / "outputs/adaptive_live_bowl_20261112"
QA = ROOT / "outputs/qa/adaptive_live_bowl_owner_20261112/result.json"
DRY = ROOT / "outputs/adaptive_live_bowl_dry_20261112/result.json"
SESSION = "adaptive_live_bowl_s42_20261112"
PRIOR_CHECKER = {
    "path": str(ROOT / "outputs/causal_replan_step107_execution_20261112/checker/result.json"),
    "sha256": "f7960d9fbc1f0cd08a85ac4eef40b0fcba1b00786ea972d971f0f379bf30d792",
}


def validate_config(config):
    """Check the original development identity and unchanged hard budgets."""
    prefix.require(
        type(config) is dict
        and config.get("schema") == "s2a.adaptive-live-bowl.v1"
        and ROOT / config.get("output", "") == OUTPUT
        and config.get("case_id") == "task0_init0_s42_raw"
        and config.get("task") == "put the bowl on the plate"
        and type(config.get("seed")) is int
        and config["seed"] == 42
        and type(config.get("prefix_steps")) is int
        and config["prefix_steps"] == 107
        and config.get("gpu_uuid") == "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"
        and config.get("model_sha256") == "48138bb8c95ce8cc1ca0b70f925e57c8f16bfb21be0a243495b391a06cf5b8ec"
        and all(
            type(config.get(key)) is int and config[key] == value
            for key, value in (
                ("max_model_initializations", 1),
                ("max_model_requests", 2),
                ("denoise_calls_per_request", 35),
                ("max_denoise_calls", 70),
                ("max_adaptive_periods", 8),
                ("max_control_steps_per_phase", 115),
                ("max_native_steps_per_phase", 2875),
                ("max_models_per_phase", 3),
                ("max_contexts_per_phase", 3),
                ("previous_control_steps", 1531),
                ("previous_native_steps", 38275),
                ("previous_models", 78),
                ("previous_contexts", 78),
                ("previous_static_models", 2),
                ("previous_model_initializations", 9),
                ("previous_requests_attempted", 21),
                ("previous_requests_completed", 18),
                ("previous_denoises", 630),
                ("fresh_cases", 0),
                ("worker_timeout_seconds", 3600),
            )
        )
        and config.get("automatic_retry") is False
        and config["prefix_steps"] + config["max_adaptive_periods"] == config["max_control_steps_per_phase"]
        and config["max_native_steps_per_phase"] == 25 * config["max_control_steps_per_phase"]
        and config["max_denoise_calls"] == config["max_model_requests"] * config["denoise_calls_per_request"],
        "The registered case, task, device and physical/model budgets must remain exact",
    )


def load_inputs():
    """Verify the saved endpoint, model response and prior cumulative evidence."""
    prefix.environment()
    refs = {
        "config": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "plan": {"path": str(PLAN), "sha256": PLAN_SHA},
        "prior_checker": PRIOR_CHECKER,
    }
    config = prefix.read(refs["config"])
    validate_config(config)
    prefix.checked(refs["plan"])
    for name in ("prior_capture", "prior_model", "prior_execution"):
        refs[name] = {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]}
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
    model_inputs = model.bind().load_inputs()
    initial = model_inputs["initial"]
    expected_capture = capture["terminal_capture"]["capture"]
    response = prior["envelope"]
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


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        initial=inputs["initial"],
        model_sha256=inputs["config"]["model_sha256"],
        source_sha256=inputs["source_sha256"],
    )


def model_registration(inputs, capture_root):
    """Create one phase-local capture root without changing the trial limits."""
    capture_root = Path(capture_root).resolve()
    prefix.require(
        capture_root in (OUTPUT / "producer/simulation", OUTPUT / "checker/simulation"),
        "The capture root must be one registered phase directory inside this trial",
    )
    config = inputs["config"]
    return dict(
        session=SESSION,
        initial=dict(inputs["initial"]),
        image_root=str(capture_root.relative_to(ROOT)),
        task=config["task"],
        seed=config["seed"],
        model_sha256=config["model_sha256"],
        max_requests=config["max_model_requests"],
        max_periods=config["max_adaptive_periods"],
        denoise_steps=config["denoise_calls_per_request"],
    )
