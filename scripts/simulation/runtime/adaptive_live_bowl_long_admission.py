"""Admit a longer adaptive trial from independently checked recovery evidence."""

from pathlib import Path

from scripts.simulation.runtime import adaptive_live_bowl_recovery_admission as previous
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as audit
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as previous_owner
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/adaptive_live_bowl_long_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/adaptive_live_bowl_long_20261112.md"
OUTPUT = ROOT / "outputs/adaptive_live_bowl_long_20261112"
QA = ROOT / "outputs/qa/adaptive_live_bowl_long_final_20261112/result.json"
DRY = ROOT / "outputs/adaptive_live_bowl_long_dry_20261112/result.json"
CONFIG_SHA = "f1a020680388cb08567689913e36eb098a386a2a487e4dc2850cd3659a6f20f2"
PLAN_SHA = "139e634be175e49abefe5e8f293c26153b32891980c102cbfb7466cee08a94ce"
SESSION = "adaptive_live_bowl_long_s42_20261112"
PRIOR_PRODUCER_SHA = "b2c1df055bd76fb8cdd5c0f948f8f26ae3074b9ffefcb3075a2860c37eeb57be"
PRIOR_CHECKER_SHA = "c546979a78877233fc60aa591eee4d08d6710862de31581ed001b50b97f2aadc"


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(_same(value[key], item) for key, item in expected.items())
    return value == expected


def validate_config(config, old_config):
    """Freeze identities, explicit limits, historical ranges and output paths."""
    previous.validate_config(
        old_config, prefix.read({"path": str(previous.historical.CONFIG), "sha256": previous.historical.CONFIG_SHA})
    )
    identity = (
        "case_id",
        "task",
        "seed",
        "prefix_steps",
        "initial_state_sha256",
        "initial_image_sha256",
        "gpu_uuid",
        "model_sha256",
        "denoise_calls_per_request",
    )
    expected = dict(
        schema="s2a.adaptive-live-bowl-long.v1",
        output=str(OUTPUT.relative_to(ROOT)),
        qa=str(QA.relative_to(ROOT)),
        dry=str(DRY.relative_to(ROOT)),
        session=SESSION,
        prior_producer=str((previous.OUTPUT / "producer/result.json").relative_to(ROOT)),
        prior_producer_sha256=PRIOR_PRODUCER_SHA,
        prior_checker=str((previous.OUTPUT / "checker/result.json").relative_to(ROOT)),
        prior_checker_sha256=PRIOR_CHECKER_SHA,
        max_model_initializations=1,
        max_model_requests=40,
        max_denoise_calls=1400,
        max_adaptive_periods=40,
        max_control_steps_per_phase=147,
        max_native_steps_per_phase=3675,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
        previous_physical_bounds=dict(
            control_steps=dict(lower=1761, upper=1876),
            native_steps=dict(lower=44025, upper=46900),
            models=dict(lower=84, upper=87),
            contexts=dict(lower=84, upper=87),
        ),
        previous_static_models=2,
        previous_model_initializations=10,
        previous_requests_attempted=23,
        previous_requests_completed=20,
        previous_denoises=700,
        fresh_cases=0,
        automatic_retry=False,
        worker_timeout_seconds=3600,
        **{key: old_config[key] for key in identity},
    )
    prefix.require(
        _same(config, expected)
        and config["prefix_steps"] + config["max_adaptive_periods"] == config["max_control_steps_per_phase"]
        and 25 * config["max_control_steps_per_phase"] == config["max_native_steps_per_phase"]
        and config["max_model_requests"] * config["denoise_calls_per_request"] == config["max_denoise_calls"],
        "The longer trial identity, prior ranges and budgets must remain exact",
    )


def verify_previous(inputs, producer, checker, sources):
    """Re-audit the real producer and checker, including uncertain prior costs."""
    producer_physical, producer_model = previous_owner.verified_producer_bounds(inputs, producer, sources)
    prefix.require(
        checker["complete"] is checker["passed"] is checker["sources_unchanged"] is checker["gpu_released"] is True
        and checker["phase"] == "checker"
        and checker["branch_ref"] == checker["available_branch_ref"],
        "A completed independently checked recovery is required",
    )
    registration = prefix.read(checker["registration_ref"])
    workers = prefix.read(checker["workers_ref"])
    lifecycle_check.require_closed_workers(workers, ["checker"])
    prefix.require(
        registration["phase"] == "checker"
        and registration["inputs"] == previous.input_receipt(inputs)
        and registration["source_refs"] == sources
        and registration["model_bundle_ref"] is None
        and registration["producer_branch_ref"] == producer["branch_ref"]
        and registration["producer_model_ref"] == producer["model_ref"],
        "Checker registration must bind the same producer, model and sources",
    )
    for worker in workers["workers"]:
        prefix.checked(worker["log_ref"])
    produced, replayed = prefix.read(producer["branch_ref"]), prefix.read(checker["branch_ref"])
    counts = audit.summarize_branch(replayed, "checker", inputs["config"])
    counts["independent_max_absolute_difference"] = audit.compare_branches(produced, replayed)
    prefix.require(
        counts == checker["counts"]
        and counts["control_steps"] == 115
        and counts["adaptive_steps"] == 8
        and all(value == 0 for value in counts["independent_max_absolute_difference"].values()),
        "The previous actual commands, states and independent replay must agree",
    )
    previous_owner.require_complete_phase_costs(checker)
    zero = dict(
        model_initialization_attempts=0,
        model_initializations=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        denoise_calls_attempted=0,
        denoise_calls_completed=0,
    )
    physical = audit.add_cost_bounds(producer_physical, audit.exact_cost_bounds(counts))
    model = previous_owner.add_model_cost_bounds(
        producer_model, previous_owner.model_cost_bounds(zero, config=inputs["config"], receipt_final=True)
    )
    prefix.require(
        checker["cumulative_physical_bounds"] == physical and checker["cumulative_model_bounds"] == model,
        "The independent checker's cumulative costs must retain all previous work",
    )
    return physical, model


def load_inputs():
    """Load the immutable long registration and independently verified history."""
    prefix.environment()
    refs = {
        "long_config": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "long_plan": {"path": str(PLAN), "sha256": PLAN_SHA},
        "prior_producer": {"path": str(previous.OUTPUT / "producer/result.json"), "sha256": PRIOR_PRODUCER_SHA},
        "prior_checker": {"path": str(previous.OUTPUT / "checker/result.json"), "sha256": PRIOR_CHECKER_SHA},
    }
    config = prefix.read(refs["long_config"])
    prefix.checked(refs["long_plan"])
    old_inputs = previous.load_inputs()
    validate_config(config, old_inputs["config"])
    producer, checker = prefix.read(refs["prior_producer"]), prefix.read(refs["prior_checker"])
    physical, model = verify_previous(old_inputs, producer, checker, previous_owner.source_refs())
    expected_model = previous_owner.known_model_bounds(config)
    prefix.require(
        physical == config["previous_physical_bounds"] and model == expected_model,
        "Registered cumulative physical and model bounds must match verified history",
    )
    refs.update(
        prior_producer_registration=producer["registration_ref"],
        prior_producer_branch=producer["branch_ref"],
        prior_model=producer["model_ref"],
        prior_producer_workers=producer["workers_ref"],
        prior_checker_registration=checker["registration_ref"],
        prior_checker_branch=checker["branch_ref"],
        prior_checker_workers=checker["workers_ref"],
    )
    consumed = dict(old_inputs["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return {
        **old_inputs,
        "config": config,
        "refs": {**old_inputs["refs"], **refs},
        "previous_physical_bounds": physical,
        "previous_model_bounds": model,
        "prior_recovery_producer": producer,
        "prior_recovery_checker": checker,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        initial=inputs["initial"],
        model_sha256=inputs["config"]["model_sha256"],
        previous_physical_bounds=inputs["previous_physical_bounds"],
        previous_model_bounds=inputs["previous_model_bounds"],
        source_sha256=inputs["source_sha256"],
    )


def model_registration(inputs, capture_root):
    """Bind the current-image stream to the new phase-local output."""
    capture_root = Path(capture_root).resolve()
    config = inputs["config"]
    prefix.require(
        capture_root in (OUTPUT / "producer/simulation", OUTPUT / "checker/simulation")
        and config.get("session") == SESSION,
        "The capture root must belong to the registered longer trial",
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
