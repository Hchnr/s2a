"""Admit one separately registered adaptive recovery development trial."""

from pathlib import Path

from scripts.simulation.runtime import adaptive_live_bowl_admission as historical
from scripts.simulation.runtime import adaptive_recovery_inputs as recovery
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/adaptive_live_bowl_recovery_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/adaptive_live_bowl_recovery_20261112.md"
OUTPUT = ROOT / "outputs/adaptive_live_bowl_recovery_20261112"
QA = ROOT / "outputs/qa/adaptive_live_bowl_recovery_final_20261112/result.json"
DRY = ROOT / "outputs/adaptive_live_bowl_recovery_dry_20261112/result.json"
CONFIG_SHA = "975c0c5e2046c137e5cd09588c1c0ef8f9bd01bbee129a4178afc7cf4447ac76"
PLAN_SHA = "dbbb154675e11cf50d6503788d76123c38e66b48fb857dab507ebd3560adc4cb"
SESSION = "adaptive_live_bowl_recovery_s42_20261112"
PREVIOUS_PREFLIGHT = {
    "path": str(historical.OUTPUT / "preflight.json"),
    "sha256": "2902c6e7d320573bce6b490c61394b1dd011b325f470a3788414eb5f05836cc4",
}

INHERITED_KEYS = (
    "case_id",
    "task",
    "seed",
    "prefix_steps",
    "initial_state_sha256",
    "initial_image_sha256",
    "gpu_uuid",
    "model_sha256",
    "max_model_initializations",
    "max_model_requests",
    "denoise_calls_per_request",
    "max_denoise_calls",
    "max_adaptive_periods",
    "max_control_steps_per_phase",
    "max_native_steps_per_phase",
    "max_models_per_phase",
    "max_contexts_per_phase",
    "previous_control_steps",
    "previous_native_steps",
    "previous_models",
    "previous_contexts",
    "previous_static_models",
    "previous_model_initializations",
    "previous_requests_attempted",
    "previous_requests_completed",
    "previous_denoises",
    "fresh_cases",
    "automatic_retry",
    "worker_timeout_seconds",
)


def validate_config(config, old_config):
    """Reject changed identity, budgets, uncertainty or recovery destinations."""
    historical.validate_config(old_config)
    expected = {
        "schema": "s2a.adaptive-live-bowl-recovery.v1",
        "output": str(OUTPUT.relative_to(ROOT)),
        "qa": str(QA.relative_to(ROOT)),
        "dry": str(DRY.relative_to(ROOT)),
        "session": SESSION,
        "prior_failed_attempt": str(Path(recovery.FAILURE["path"]).relative_to(ROOT)),
        "prior_failed_attempt_sha256": recovery.FAILURE["sha256"],
        "unknown_prior_control_steps_upper": old_config["max_control_steps_per_phase"],
        "unknown_prior_native_steps_upper": old_config["max_native_steps_per_phase"],
        "unknown_prior_models_upper": old_config["max_models_per_phase"],
        "unknown_prior_contexts_upper": old_config["max_contexts_per_phase"],
        **{key: old_config[key] for key in INHERITED_KEYS},
    }
    prefix.require(
        type(config) is dict
        and set(config) == set(expected)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in expected.items())
        and config["prefix_steps"] + config["max_adaptive_periods"] == config["max_control_steps_per_phase"]
        and config["max_native_steps_per_phase"] == 25 * config["max_control_steps_per_phase"]
        and config["max_denoise_calls"] == config["max_model_requests"] * config["denoise_calls_per_request"],
        "The registered recovery identity, prior uncertainty and budgets must remain exact",
    )


def load_inputs():
    """Bind historical inputs, the failed attempt and the new immutable plan."""
    prefix.environment()
    new_refs = {
        "recovery_config": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "recovery_plan": {"path": str(PLAN), "sha256": PLAN_SHA},
        "failed_attempt": recovery.FAILURE,
        "failed_preflight": PREVIOUS_PREFLIGHT,
    }
    config = prefix.read(new_refs["recovery_config"])
    prefix.checked(new_refs["recovery_plan"])
    recorded = recovery.load_recorded_inputs()
    validate_config(config, recorded["config"])
    failed = recovery.load_failed_attempt(recorded["config"])
    prior_preflight = prefix.read(new_refs["failed_preflight"])
    failure = prefix.read(new_refs["failed_attempt"])
    new_refs.update(
        failed_workers=failed["workers_ref"],
        failed_model=failed["model_ref"],
        failed_registration=failure["registration_ref"],
    )
    expected_bounds = {
        "control_steps": ("previous_control_steps", "unknown_prior_control_steps_upper"),
        "native_steps": ("previous_native_steps", "unknown_prior_native_steps_upper"),
        "models": ("previous_models", "unknown_prior_models_upper"),
        "contexts": ("previous_contexts", "unknown_prior_contexts_upper"),
    }
    prefix.require(
        prior_preflight["complete"] is prior_preflight["passed"] is True
        and prior_preflight["gpu_uuid"] == config["gpu_uuid"]
        and prior_preflight["model_initializations"] == prior_preflight["model_requests"] == 0
        and prior_preflight["physics_steps"] == 0
        and prior_preflight["source_files_verified"] == len(recorded["model_source_hashes"])
        and prior_preflight["weight_files_verified"] == len(recorded["weight_hashes"]) == 163
        and failed["failure_ref"] == new_refs["failed_attempt"]
        and all(
            failed["cumulative_physical_bounds"][name]
            == dict(lower=config[known], upper=config[known] + config[uncertain])
            for name, (known, uncertain) in expected_bounds.items()
        )
        and all(
            failed[key] == config[key]
            for key in (
                "previous_static_models",
                "previous_model_initializations",
                "previous_requests_attempted",
                "previous_requests_completed",
                "previous_denoises",
            )
        ),
        "The previous preflight, failed attempt and cumulative ranges must agree",
    )
    consumed = dict(recorded["source_sha256"])
    for ref in new_refs.values():
        prefix.checked(ref, consumed)
    return {
        **recorded,
        "config": config,
        "refs": {**recorded["refs"], **new_refs},
        "failed_attempt": failed,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        initial=inputs["initial"],
        model_sha256=inputs["config"]["model_sha256"],
        failed_attempt=inputs["failed_attempt"],
        source_sha256=inputs["source_sha256"],
    )


def model_registration(inputs, capture_root):
    """Bind the same bounded model contract to a new phase-local capture root."""
    capture_root = Path(capture_root).resolve()
    config = inputs["config"]
    prefix.require(
        capture_root in (OUTPUT / "producer/simulation", OUTPUT / "checker/simulation")
        and config.get("session") == SESSION,
        "The capture root must be one registered recovery phase directory",
    )
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
