"""Register a new bounded pilot after the audited denoise-counter failure."""

import copy
from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_live_admission as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_multitask_live_48_recovery_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_multitask_live_recovery_20261111.md"
OUTPUT = ROOT / "outputs/reset_multitask_live_48_recovery_20261111"
QA = ROOT / "outputs/qa/reset_multitask_live_48_recovery_20261111/result.json"
DRY = ROOT / "outputs/reset_multitask_live_48_recovery_dry_20261111/result.json"
INCIDENT = {
    "path": str(ROOT / "outputs/reset_multitask_live_failure_audit_20261111/result.json"),
    "sha256": "e6aa8c57a9ad299b71f7b49d0330a65b525de5247dbc41e321d04417c07e7937",
}
CASES = previous.CASES
FIXED = copy.deepcopy(previous.FIXED)
FIXED.update(
    schema="s2a.reset-multitask-live-48-recovery.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    incident_sha256=INCIDENT["sha256"],
    previously_completed_physical_periods=48,
    previously_completed_inner_steps=1200,
    previously_initialized_models=3,
    previously_attempted_model_requests=6,
    previously_completed_model_requests=3,
    previously_completed_denoise=105,
)


def validate_config(config):
    prefix.require(previous._exact(config, FIXED), "The exact recovery roster and cumulative budgets are required")


def source_refs():
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_reset_multitask_live_48_v2.py",
        ROOT / "scripts/simulation/runtime/reset_multitask_live_workers_v2.py",
        ROOT / "scripts/simulation/runtime/audit_reset_multitask_live_failure.py",
        ROOT / "scripts/wam/tracking/preview/reset_multitask_counter.py",
        ROOT / "reports/tracking/references/reset_multitask_live_failure_20261111.md",
        ROOT / "tests/research/wam/test_reset_multitask_counter.py",
        ROOT / "tests/research/native_runtime/test_audit_reset_multitask_live_failure.py",
        ROOT / "tests/research/native_runtime/test_reset_multitask_live_recovery.py",
    )
    return previous.source_refs() + [prefix.reference(path) for path in names]


def load_inputs(*, verify_assets=True):
    inputs = previous.load_inputs(verify_assets=verify_assets)
    prior_config_ref = inputs["config_ref"]
    config_ref, plan_ref = prefix.reference(CONFIG), prefix.reference(PLAN)
    config, incident = prefix.read(config_ref), prefix.read(INCIDENT)
    validate_config(config)
    prefix.require(
        incident["complete"] is incident["passed"] is True
        and incident["control_qualified"] is incident["independently_replayed"] is False
        and incident["actual_physical_periods"] == 48
        and incident["actual_inner_steps"] == 1200
        and incident["actual_sim_models"] == incident["actual_osmesa_contexts"] == 9
        and incident["model_initializations"] == 3
        and incident["model_requests_attempted"] == 6
        and incident["model_requests_completed"] == 3
        and incident["denoise_calls"] == 105
        and incident["new_native_models"] == incident["new_physics_steps"] == incident["new_model_requests"] == 0
        and incident["automatic_retry"] is False
        and [case["case_index"] for case in incident["cases"]] == [0, 4, 8],
        "The complete failed-run audit and its actual cumulative costs are required",
    )
    for ref in (config_ref, plan_ref, INCIDENT):
        prefix.checked(ref, inputs["source_sha256"])
    inputs.update(config=config, config_ref=config_ref, prior_config_ref=prior_config_ref, recovery_plan_ref=plan_ref)
    return inputs


def input_receipt(inputs):
    result = previous.input_receipt(inputs)
    result.update(
        prior_config_ref=inputs["prior_config_ref"], recovery_plan_ref=inputs["recovery_plan_ref"], incident=INCIDENT
    )
    return result


def model_registration(inputs, index, image_root):
    return previous.model_registration(inputs, index, image_root)
