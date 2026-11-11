"""Admit a one-period physical probe with completed causal innovation."""

from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as prior
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/early_causal_innovation_physical_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/early_causal_innovation_physical_20261111.md"
OUTPUT = ROOT / "outputs/early_causal_innovation_physical_20261111"
QA = ROOT / "outputs/qa/early_causal_innovation_physical_20261111/result.json"
DRY = ROOT / "outputs/early_causal_innovation_physical_dry_20261111/result.json"
FIXED = {
    "schema": "s2a.early-causal-innovation-physical.v1",
    "output": "outputs/early_causal_innovation_physical_20261111",
    "recorded_audit": "outputs/quality_rate_repair_recorded_feedback_audit_20261111/result.json",
    "recorded_audit_sha256": "bd33ed32be8e4bb4a9ef0c3fb753a8d0883e7cb6bd3ef22fedd9729807b4c802",
    "static_early": "outputs/early_causal_innovation_quality_20261111/worker/result.json",
    "static_early_sha256": "878a290e2b4bd7a1970c4a0cd1ba9785199696b74f62f965fcdc20d94521f9ee",
    "case_index": 8,
    "seed": 42,
    "prefix_steps": 80,
    "prior_periods": 42,
    "last_completed_step": 121,
    "decision_step": 122,
    "new_periods": 1,
    "max_control_steps_per_phase": 123,
    "max_native_steps_per_phase": 3075,
    "max_decisions": 44,
    "max_baseline_model_responses": 21120,
    "max_quality_model_responses": 7040,
    "max_independent_endpoint_queries": 8600,
    "max_independent_transition_steps": 111800,
    "max_gripper_candidates": 288684,
    "new_model_initializations": 0,
    "new_model_requests": 0,
    "new_denoise_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 2400,
    "automatic_retry": False,
}


def validate_static(saved):
    analysis = saved["analysis"]
    prefix.require(
        saved["complete"] is saved["passed"] is True
        and analysis["decision_step"] == 122
        and analysis["last_completed_step"] == 121
        and analysis["generated_step"] == 120
        and analysis["reference_cursor"] == 2
        and analysis["candidates"]["innovation_quality"]["certified_model_witness"] is True
        and saved["stats"]["physics_steps_attempted"] == 0,
        "The completed, independently checked static early search is required",
    )


def load_inputs():
    inputs = prior.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact one-period physical budget is required",
    )
    consumed = dict(inputs["source_sha256"])
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("recorded_audit", "static_early")
    }
    recorded, saved = (prefix.read(refs[key], consumed) for key in ("recorded_audit", "static_early"))
    prefix.require(
        recorded["complete"] is recorded["passed"] is recorded["checker_policy_isolated"] is True
        and recorded["continuation_periods"] == 46
        and recorded["control_qualified"] is False,
        "The audited recorded-reference physical run is required",
    )
    validate_static(saved)
    prefix.require(
        len(inputs["records"]) == 6 and [row["step"] for row in inputs["prior_periods"][:42]] == list(range(80, 122)),
        "Six source-bound references and the completed 42-period prefix are required",
    )
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    inputs.update(
        feedback_config=config, one_period_config_ref=config_ref, one_period_refs=refs, source_sha256=consumed
    )
    return inputs


def input_receipt(inputs):
    return {
        "prior": prior.input_receipt(inputs),
        "one_period_config_ref": inputs["one_period_config_ref"],
        "one_period_refs": inputs["one_period_refs"],
    }
