"""Admit one cached checker using the preserved failed long-run evidence."""

from pathlib import Path

from scripts.simulation.runtime import adaptive_live_bowl_long_admission as previous
from scripts.simulation.runtime import audit_adaptive_long_partial as partial_audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/adaptive_long_cached_checker_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/adaptive_long_cached_checker_20261112.md"
OUTPUT = ROOT / "outputs/adaptive_long_cached_checker_20261112"
QA = ROOT / "outputs/qa/adaptive_long_cached_checker_final_20261112/result.json"
DRY = ROOT / "outputs/adaptive_long_cached_checker_dry_20261112/result.json"
CONFIG_SHA = "d21c8544e6914dce019377119e935f9a5e63fe4c69248d17031cf22c2a760894"
PLAN_SHA = "def82d32ea5ba42f03de5a4fda0d055ff7ee6f13133e4dc47ff1e789a296782c"
PARTIAL_AUDIT = {
    "path": str(ROOT / "outputs/audits/adaptive_live_bowl_long_partial_20261112/result.json"),
    "sha256": "acbbeb312be8e18cec2c7c24903985771855db8352957c8d143408c643e58421",
}


def validate_config(config, old_config):
    """Freeze the replay identity, previous uncertainty and one-run limits."""
    identity = ("case_id", "seed", "session")
    expected = dict(
        schema="s2a.adaptive-long-cached-checker.v1",
        output=str(OUTPUT.relative_to(ROOT)),
        qa=str(QA.relative_to(ROOT)),
        dry=str(DRY.relative_to(ROOT)),
        prior_producer=str(Path(partial_audit.PRODUCER["path"]).relative_to(ROOT)),
        prior_producer_sha256=partial_audit.PRODUCER["sha256"],
        failed_checker=str(Path(partial_audit.CHECKER["path"]).relative_to(ROOT)),
        failed_checker_sha256=partial_audit.CHECKER["sha256"],
        partial_audit=str(Path(PARTIAL_AUDIT["path"]).relative_to(ROOT)),
        partial_audit_sha256=PARTIAL_AUDIT["sha256"],
        max_control_steps=147,
        max_native_steps=3675,
        max_models=3,
        max_contexts=3,
        previous_physical_bounds=dict(
            control_steps=dict(lower=2055, upper=2170),
            native_steps=dict(lower=51375, upper=54250),
            models=dict(lower=90, upper=93),
            contexts=dict(lower=90, upper=93),
        ),
        previous_static_models=2,
        previous_model_bounds=dict(
            initializations_completed=dict(lower=11, upper=11),
            requests_attempted=dict(lower=30, upper=30),
            requests_completed=dict(lower=27, upper=27),
            denoise_calls_completed=dict(lower=945, upper=945),
        ),
        new_model_initializations=0,
        new_model_requests=0,
        new_denoise_calls=0,
        fresh_cases=0,
        automatic_retry=False,
        worker_timeout_seconds=3600,
        **{key: old_config[key] for key in identity},
    )
    prefix.require(
        previous._same(config, expected)
        and config["max_control_steps"] == old_config["max_control_steps_per_phase"]
        and config["max_native_steps"] == 25 * config["max_control_steps"]
        and config["max_models"] == old_config["max_models_per_phase"]
        and config["max_contexts"] == old_config["max_contexts_per_phase"],
        "The cached checker identity, accumulated costs and limits must remain exact",
    )


def load_inputs():
    """Recompute the partial audit and bind the separately registered replay."""
    prefix.environment()
    refs = {
        "cached_config": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "cached_plan": {"path": str(PLAN), "sha256": PLAN_SHA},
        "long_producer": partial_audit.PRODUCER,
        "failed_checker": partial_audit.CHECKER,
        "partial_audit": PARTIAL_AUDIT,
    }
    config = prefix.read(refs["cached_config"])
    prefix.checked(refs["cached_plan"])
    inputs = previous.load_inputs()
    validate_config(config, inputs["config"])
    saved = prefix.read(PARTIAL_AUDIT)
    actual = partial_audit.audit()
    prefix.require(
        saved == actual
        and saved["complete"] is saved["passed"] is True
        and saved["details"]["checker_complete"] is False
        and saved["cumulative_physical_bounds"] == config["previous_physical_bounds"]
        and saved["cumulative_model_bounds"] == config["previous_model_bounds"],
        "The exact recomputed partial audit and previous uncertainty are required",
    )
    producer = prefix.read(partial_audit.PRODUCER)
    failed = prefix.read(partial_audit.CHECKER)
    refs.update(
        producer_branch=producer["branch_ref"],
        producer_model=producer["model_ref"],
        producer_workers=producer["workers_ref"],
        failed_branch=failed["available_branch_ref"],
        failed_workers=failed["workers_ref"],
        partial_audit_source=saved["audit_source_ref"],
    )
    consumed = dict(inputs["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        config=config,
        long_inputs=inputs,
        refs={**inputs["refs"], **refs},
        producer=producer,
        failed_checker=failed,
        partial_audit=saved,
        previous_physical_bounds=saved["cumulative_physical_bounds"],
        previous_model_bounds=saved["cumulative_model_bounds"],
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        previous_physical_bounds=inputs["previous_physical_bounds"],
        previous_model_bounds=inputs["previous_model_bounds"],
        producer_branch_ref=inputs["producer"]["branch_ref"],
        producer_model_ref=inputs["producer"]["model_ref"],
        source_sha256=inputs["source_sha256"],
    )


def replay_registration(inputs, capture_root):
    """Use the original session while isolating new checker image files."""
    capture_root = Path(capture_root).resolve()
    prefix.require(
        capture_root == OUTPUT / "checker/simulation" and inputs["config"]["session"] == previous.SESSION,
        "The capture root and original recorded session must match the cached checker",
    )
    registration = previous.model_registration(inputs["long_inputs"], previous.OUTPUT / "checker/simulation")
    registration["image_root"] = str(capture_root.relative_to(ROOT))
    return registration
