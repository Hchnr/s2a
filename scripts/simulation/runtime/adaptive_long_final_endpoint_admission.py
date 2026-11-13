"""Admit one final-endpoint-only checker after two preserved failures."""

from pathlib import Path

from scripts.simulation.runtime import adaptive_long_cached_checker_admission as previous
from scripts.simulation.runtime import audit_cached_checker_incident_20261113 as incident
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/adaptive_long_final_endpoint_checker_20261113.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/adaptive_long_final_endpoint_checker_20261113.md"
OUTPUT = ROOT / "outputs/adaptive_long_final_endpoint_checker_20261113"
QA = ROOT / "outputs/qa/adaptive_long_final_endpoint_checker_revalidated_20261113/result.json"
DRY = ROOT / "outputs/adaptive_long_final_endpoint_checker_dry_revalidated_20261113/result.json"
CONFIG_SHA = "20a170d42051b0a7f782179c485c6e156dd43cd0f2213ceb317ba98900560705"
PLAN_SHA = "7b23836ea85e47ecf30f226edeab5b82e4fa6278fa49e268adc4460696d64649"
INCIDENT_AUDIT = {
    "path": str(ROOT / "outputs/audits/cached_checker_incident_20261113/result.json"),
    "sha256": "9aad9b9be9bb8cedb2a4c61295b4ff524cb52e689b3e39708651812e039c1f39",
}


def validate_config(config, previous_config):
    """Freeze the one-run identity and both accumulated failure costs."""
    identity = ("case_id", "seed", "session")
    expected = dict(
        schema="s2a.adaptive-long-final-endpoint-checker.v1",
        output=str(OUTPUT.relative_to(ROOT)),
        qa=str(QA.relative_to(ROOT)),
        dry=str(DRY.relative_to(ROOT)),
        prior_producer_sha256=previous.partial_audit.PRODUCER["sha256"],
        original_failed_checker_sha256=previous.partial_audit.CHECKER["sha256"],
        cached_failed_checker_sha256=incident.FAILURE["sha256"],
        cached_incident_audit_sha256=INCIDENT_AUDIT["sha256"],
        cache_step=147,
        max_control_steps=147,
        max_native_steps=3675,
        max_models=3,
        max_contexts=3,
        previous_physical_bounds=dict(
            control_steps=dict(lower=2174, upper=2289),
            native_steps=dict(lower=54350, upper=57225),
            models=dict(lower=93, upper=96),
            contexts=dict(lower=93, upper=96),
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
        **{key: previous_config[key] for key in identity},
    )
    prefix.require(
        previous.previous._same(config, expected)
        and config["cache_step"] == config["max_control_steps"] == previous_config["max_control_steps"]
        and config["max_native_steps"] == previous_config["max_native_steps"]
        and config["max_models"] == previous_config["max_models"]
        and config["max_contexts"] == previous_config["max_contexts"],
        "The final-endpoint replay identity, accumulated costs and limits must remain exact",
    )


def load_inputs():
    """Recompute the cached-failure audit and bind the prior registered run."""
    prefix.environment()
    refs = {
        "final_config": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "final_plan": {"path": str(PLAN), "sha256": PLAN_SHA},
        "cached_incident_audit": INCIDENT_AUDIT,
        "cached_failure": incident.FAILURE,
        "cached_failure_branch": incident.BRANCH,
        "cached_failure_workers": incident.WORKERS,
    }
    config = prefix.read(refs["final_config"])
    prefix.checked(refs["final_plan"])
    inputs = previous.load_inputs()
    validate_config(config, inputs["config"])
    saved = prefix.read(INCIDENT_AUDIT)
    prefix.require(
        saved == incident.audit()
        and saved["complete"] is saved["passed"] is True
        and saved["control_qualified"] is saved["failure_cause_proven"] is False
        and saved["failed_capture_state_digest_matches"] is False
        and saved["cumulative_physical_bounds"] == config["previous_physical_bounds"]
        and saved["cumulative_model_bounds"] == config["previous_model_bounds"],
        "The exact recomputed cached-failure audit and previous uncertainty are required",
    )
    refs["cached_incident_audit_source"] = saved["audit_source_ref"]
    consumed = dict(inputs["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        config=config,
        previous=inputs,
        refs={**inputs["refs"], **refs},
        producer=inputs["producer"],
        long_inputs=inputs["long_inputs"],
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
    """Preserve the original model session while isolating this replay's images."""
    capture_root = Path(capture_root).resolve()
    prefix.require(
        capture_root == OUTPUT / "checker/simulation" and inputs["config"]["cache_step"] == 147,
        "The final-endpoint capture root and exact cache step are required",
    )
    value = previous.replay_registration(inputs["previous"], previous.OUTPUT / "checker/simulation")
    value["image_root"] = str(capture_root.relative_to(ROOT))
    return value
