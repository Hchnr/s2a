"""Compare one opt-in cached nominal candidate at the recorded wine boundary."""

import argparse
import copy
import sys
from pathlib import Path

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import recover_wine_margin_cache as recovery
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_multitask_margin_recheck as original
from scripts.simulation.runtime import run_reset_start_feedback as reset

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/wine_cached_nominal_candidate_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/wine_cached_nominal_candidate_20261116.md"
OUTPUT = ROOT / "outputs/wine_cached_nominal_candidate_20261116"
QA = ROOT / "outputs/qa/wine_cached_nominal_candidate_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_wine_cached_nominal_candidate"
CANDIDATE_MODULE = "s2a.research.joint_control.preview.cached_nominal_fallback"
CASE = recovery.CASE
BASELINE = dict(
    producer=dict(
        path="outputs/wine_margin_cache_recovery_20261116/producer/result.json",
        sha256="e58773c78b5c0845cd802ce944473d0f14f176c360f51715c2042f8fde97c82e",
    ),
    checker=dict(
        path="outputs/wine_margin_cache_recovery_20261116/checker/result.json",
        sha256="725ef6af636221a70003aa712c2ea819676ec9bdac3b64d09ec961c7baa9cfe8",
    ),
)
_prepare_recovery = recovery.prepare
_recovery_source_refs = recovery.source_refs
_validate_original_branch = original.validate_branch
_checker_isolation = original.checker_isolation
_run_original_probe = original.run_matched_reference_probe


def validate_config(config):
    fixed = dict(
        schema="s2a.wine-cached-nominal-candidate.v1",
        output="outputs/wine_cached_nominal_candidate_20261116",
        baseline_producer=BASELINE["producer"],
        baseline_checker=BASELINE["checker"],
        case_index=4,
        sampling_seed=42,
        activation_step=143,
        expected_trigger="quality_margin_consumed",
        max_deferred_triggers=1,
        max_extra_periods=1,
        max_control_steps_per_phase=144,
        max_native_steps_per_phase=3600,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
        max_workers_per_phase=1,
        max_quality_responses_per_period=320,
        max_baseline_responses_per_period=480,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if (
        not isinstance(config, dict)
        or set(config) != set(fixed)
        or any(config[key] != value or type(config[key]) is not type(value) for key, value in fixed.items())
    ):
        raise ValueError("The fixed wine case, activation and physical budgets are required")


def baseline_refs():
    return {phase: dict(path=str(ROOT / ref["path"]), sha256=ref["sha256"]) for phase, ref in BASELINE.items()}


def prepare(spec):
    prefix.require(spec == CASE, "The original wine / seed42 case is required")
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    data = _prepare_recovery(spec)
    refs = baseline_refs()
    phases = {phase: prefix.read(ref) for phase, ref in refs.items()}
    for phase, result in phases.items():
        counts = result["case"]["counts"]
        prefix.require(
            result["complete"] is result["passed"] is result["owner_closed"] is result["sources_unchanged"] is True
            and result["phase"] == phase
            and result["case"]["status"] == "closed"
            and counts["control_steps"] == 143
            and counts["native_steps"] == 3575
            and counts["probe_steps"] == 0
            and counts["cache_boundary"]["original_capture_matched"] is True,
            "The original 143-period wine baseline and independent replay must remain intact",
        )
        prefix.checked(result["case"]["branch_ref"])
    producer = prefix.read(phases["producer"]["case"]["branch_ref"])
    traces = producer["runtime"]["segment_receipts"][0]["shared_shift_traces"]
    last = traces[-1]
    prefix.require(
        len(traces) == 144
        and producer["runtime"]["continuation"]["probe"]["stop"]["reason"] == "quality_no_witness"
        and producer["runtime"]["continuation"]["probe"]["stop"]["solver_status"] == "shared_search_ineligible"
        and last["shared_search"]["primary_responses"] == 320
        and last["shared_search"]["primary_margin_evaluations"] == 0
        and last["shared_search"]["primary_phases"][-1]["phase"] == "future_quality"
        and last["shared_search"]["primary_phases"][-1]["message"] == "response_budget",
        "The original interrupted margin search and its full budget are required",
    )
    data["identity"]["candidate_config_ref"] = config_ref
    data["identity"]["baseline_refs"] = refs
    return data


def source_refs():
    return _recovery_source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/preview/cached_nominal_fallback.py",
            ROOT / "tests/research/joint_control/preview/test_cached_nominal_fallback.py",
            ROOT / "tests/research/native_runtime/test_wine_cached_nominal_candidate.py",
            ROOT / "reports/tracking/references/wine_margin_cache_recovery_20261116.md",
        )
    ]


def checker_isolation():
    _checker_isolation()
    prefix.require(CANDIDATE_MODULE not in sys.modules, "The checker cannot import the cached candidate")


def run_probe(*args, **kwargs):
    """Keep the candidate off until the registered producer probe boundary."""
    if kwargs.get("phase") == "producer":
        from s2a.research.joint_control.preview.cached_nominal_fallback import CachedNominalFallbackAudit

        holder = {}

        def factory(receipt):
            prefix.require(not holder, "Only one fresh candidate adapter is allowed")
            holder["adapter"] = CachedNominalFallbackAudit(receipt)
            return holder["adapter"]

        def activate(policy, owner, receipt, step):
            prefix.require(step == 143 and "adapter" in holder, "Only the registered completed boundary may activate")
            holder["adapter"].activate(policy, owner, receipt, step=step)

        kwargs.update(adapter_factory=factory, activate_probe=activate)
    else:
        checker_isolation()
    return _run_original_probe(*args, **kwargs)


def validate_branch(branch, phase, spec):
    counts = _validate_original_branch(branch, phase, spec)
    if phase == "checker":
        prefix.require(branch["candidate_modules_unloaded"] is True, "The checker must remain candidate-free")
        return counts
    runtime = branch["runtime"]
    activation = runtime.get("cached_nominal_activation")
    traces = runtime["segment_receipts"][0].get("cached_nominal_traces")
    prefix.require(
        activation == dict(step=143, completed_periods=143, future_observations=0)
        and isinstance(traces, list)
        and len(traces) == 1
        and traces[0]["enabled"] is True
        and traces[0]["extra_model_responses"] == 0,
        "Exactly one boundary activation and one accounted cache inspection are required",
    )
    probe = runtime["continuation"]["probe"]
    if probe["periods"]:
        verification = probe["periods"][0]["decision"]["quality"]["verification"]
        prefix.require(
            traces[0]["selected"] is True
            and verification["passed"] is verification["prefix_pose_caps_satisfied"] is True
            and verification["selection_scope"] == "cached_nominal_two_step"
            and verification["model_margin_enforced"] is False,
            "Any extra action requires the original independently audited nominal two-point witness",
        )
    baseline = prefix.read(prefix.read(baseline_refs()["producer"])["case"]["branch_ref"])
    recorded, actual = original.periods(baseline), original.periods(branch)
    prefix.require(len(recorded) == 143, "The exact original baseline prefix is required")
    counts["baseline_prefix_max_difference"] = comparison.compare_periods(recorded, actual[: len(recorded)])
    counts["cached_nominal_trace"] = copy.deepcopy(traces[0])
    return counts


def dry_run():
    prefix.environment()
    reset.require_no_simulator_import()
    config_ref = prefix.reference(CONFIG)
    validate_config(prefix.read(config_ref))
    data = prepare(CASE)
    checker_isolation()
    print(
        publish_result(
            OUTPUT / "dry.json",
            dict(
                complete=True,
                passed=True,
                config_ref=config_ref,
                identities=[data["identity"]],
                source_refs=source_refs(),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def configure():
    recovery.configure()
    recovery.CONFIG = original.CONFIG = CONFIG
    recovery.PLAN = original.PLAN = PLAN
    recovery.OUTPUT = original.OUTPUT = OUTPUT
    recovery.QA = original.QA = QA
    recovery.MODULE = original.MODULE = MODULE
    recovery.validate_config = original.validate_config = validate_config
    recovery.prepare = original.prepare = prepare
    recovery.source_refs = original.source_refs = source_refs
    original.DRY = OUTPUT / "dry.json"
    original.EXPECTED_CASES = [CASE]
    original.checker_isolation = checker_isolation
    original.run_matched_reference_probe = run_probe
    original.validate_branch = validate_branch


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.phase:
        recovery.phase_run(args.phase)
    else:
        original.worker(args.worker[0], int(args.worker[1]))


if __name__ == "__main__":
    main()
