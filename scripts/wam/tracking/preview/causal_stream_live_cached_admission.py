"""Admit a new bounded live trial after the cached-endpoint evidence audit."""

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import causal_stream_live_48_admission as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_stream_live_cached_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_stream_live_cached_20261111.md"
OUTPUT = ROOT / "outputs/causal_stream_live_cached_20261111"
QA = ROOT / "outputs/qa/causal_stream_live_cached_20261111/result.json"
DRY = ROOT / "outputs/causal_stream_live_cached_dry_20261111/result.json"
GPU = previous.GPU

FIXED = {
    "schema": "s2a.causal-stream-live-cached.v1",
    "output": "outputs/causal_stream_live_cached_20261111",
    "prior_cache_audit": "outputs/causal_stream_endpoint_cache_fixed_audit_20261111/result.json",
    "prior_cache_audit_sha256": "961f59575825aa04cac820abb8e39815c7da04e731aff43f9ee3310ed755c3b2",
    "gpu_uuid": GPU,
    "case_index": 8,
    "seed": 42,
    "task": "put the cream cheese in the bowl",
    "prefix_steps": 80,
    "window_steps": 8,
    "max_windows": 6,
    "max_model_initializations": 1,
    "max_model_requests": 6,
    "denoise_calls_per_request": 35,
    "max_denoise_calls": 210,
    "max_control_steps_per_phase": 128,
    "max_native_steps_per_phase": 3200,
    "fresh_cases": 0,
    "fits": 0,
    "worker_timeout_seconds": 2400,
    "automatic_retry": False,
}


def validate_cached_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The complete cached live model and physical budgets must remain fixed",
    )


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_cached_config(config)
    inputs = previous.load_inputs()
    previous_receipt = previous.input_receipt(inputs)
    consumed = dict(inputs["source_sha256"])
    audit_ref = {"path": str(ROOT / config["prior_cache_audit"]), "sha256": config["prior_cache_audit_sha256"]}
    audit = prefix.read(audit_ref, consumed)
    comparison = audit["comparison"]
    prefix.require(
        audit["complete"] is audit["passed"] is True
        and audit["original_owner_passed"] is audit["control_qualified"] is False
        and audit["new_model_requests"] == audit["new_physics_steps"] == 0
        and audit["task_success_tail_periods"] == 33
        and comparison["completed_actions"] == 40
        and comparison["native_steps"] == 3000
        and comparison["final_state_bytes_equal"] is False
        and all(value <= 1e-10 for value in comparison["max_difference"].values()),
        "The separate audited cached replay and preserved original failure are required",
    )
    for ref in (audit["capture_ref"], audit["prior_audit_ref"], audit["original_branch_ref"], config_ref):
        prefix.checked(ref, consumed)
    prefix.reference(PLAN)
    for key in (
        "gpu_uuid",
        "case_index",
        "seed",
        "task",
        "prefix_steps",
        "window_steps",
        "max_windows",
        "max_model_initializations",
        "max_model_requests",
        "denoise_calls_per_request",
        "max_denoise_calls",
        "max_control_steps_per_phase",
        "max_native_steps_per_phase",
        "fresh_cases",
        "fits",
        "worker_timeout_seconds",
        "automatic_retry",
    ):
        prefix.require(config[key] == inputs["live_config"][key], "Cached and underlying live budgets differ")
    return {
        **inputs,
        "previous_long_receipt": previous_receipt,
        "cached_config": config,
        "cached_config_ref": config_ref,
        "cached_audit_ref": audit_ref,
        "long_config_ref": config_ref,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "cached_config_ref": inputs["cached_config_ref"],
        "cached_audit_ref": inputs["cached_audit_ref"],
        "previous_long_receipt": inputs["previous_long_receipt"],
        "model_sha256": inputs["model_sha256"],
        "source_sha256": inputs["source_sha256"],
    }


def validate_config(config):
    """The frozen model worker still receives the original six-window config."""
    previous.validate_config(config)
