"""Recheck the original wine margin boundary with shared endpoint caching."""

import argparse
import sys
from pathlib import Path

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as captures
from scripts.simulation.runtime import run_multitask_margin_recheck as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/wine_margin_cache_recovery_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/wine_margin_cache_recovery_20261116.md"
OUTPUT = ROOT / "outputs/wine_margin_cache_recovery_20261116"
QA = ROOT / "outputs/qa/wine_margin_cache_recovery_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.recover_wine_margin_cache"
CASE = dict(case_index=4, sampling_seed=42, trigger_step=143, trigger_reason="quality_margin_consumed")
DIAGNOSTIC = dict(
    path="outputs/wine_capture_identity_recovery_20261116/result.json",
    sha256="eba8ae3c31f8aaad77067430212124da2c4a4205a36b51963508b2fc20e5c62c",
)
_prepare_original = original.prepare
_original_source_refs = original.source_refs


def validate_config(config):
    fixed = dict(
        schema="s2a.wine-margin-cache-recovery.v1",
        output="outputs/wine_margin_cache_recovery_20261116",
        development_review=dict(
            path="outputs/reset_multitask_adaptive_complete_audit_20261113/result.json",
            sha256="7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
        ),
        capture_diagnostic=DIAGNOSTIC,
        cases=[CASE],
        max_deferred_triggers_per_case=1,
        max_extra_periods_per_case=1,
        max_total_control_steps_per_phase=144,
        max_total_native_steps_per_phase=3600,
        max_workers_per_phase=1,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
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
        or any(type(config["cases"][0][key]) is not int for key in ("case_index", "sampling_seed", "trigger_step"))
    ):
        raise ValueError("The single-case cache recovery and its original physical budgets are required")


def prepare(spec):
    prefix.require(spec == CASE, "The originally failed wine / seed42 case is required")
    data = _prepare_original(spec)
    ref = dict(path=str(ROOT / DIAGNOSTIC["path"]), sha256=DIAGNOSTIC["sha256"])
    diagnostic = prefix.read(ref)
    expected_paths = [
        "current_endpoint.sim_state_sha256",
        "render_before.sim_state_sha256",
        "render_after.sim_state_sha256",
    ]
    prefix.require(
        diagnostic["complete"] is diagnostic["passed"] is diagnostic["owner_closed"] is True
        and diagnostic["sources_unchanged"] is True
        and all(
            diagnostic["phases"][phase]["status"] == "closed"
            and diagnostic["phases"][phase]["counts"]["original_guard_passed"] is False
            and diagnostic["phases"][phase]["counts"]["differing_paths"] == expected_paths
            and diagnostic["phases"][phase]["counts"]["control_steps"] == 104
            and diagnostic["phases"][phase]["counts"]["native_steps"] == 2600
            for phase in ("producer", "checker")
        ),
        "The completed two-path identity diagnostic must preserve its mismatch and cost",
    )
    prefix.checked(diagnostic["phases"]["producer"]["branch_ref"])
    prefix.checked(diagnostic["phases"]["checker"]["branch_ref"])
    data["identity"]["capture_diagnostic_ref"] = ref
    return data


def source_refs():
    return _original_source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "reports/tracking/references/wine_capture_identity_recovery_20261116.md",
            ROOT / "tests/research/native_runtime/test_wine_margin_cache_recovery.py",
        )
    ]


def configure():
    original.CONFIG = CONFIG
    original.PLAN = PLAN
    original.OUTPUT = OUTPUT
    original.DRY = OUTPUT / "dry.json"
    original.QA = QA
    original.MODULE = MODULE
    original.EXPECTED_CASES = [CASE]
    original.validate_config = validate_config
    original.prepare = prepare
    original.source_refs = source_refs


def cached_boundary(branch, data):
    """Require an observed then same-state reused step104 endpoint and its original capture."""
    events = [item for item in branch["runtime"]["endpoint_cache_events"] if item["step"] == 104]
    expected = next(item for item in data["captures"] if item["frame_index"] == 104)
    actual = next(item for item in branch["reference_captures"] if item["frame_index"] == 104)
    captures.compare_capture(expected, actual)
    prefix.require(
        [item["kind"] for item in events] == ["observed", "reused"]
        and all(item["state_sha256"] == expected["current_endpoint"]["sim_state_sha256"] for item in events)
        and all(item["time"] == expected["current_endpoint"]["sim_time"] for item in events),
        "The original step104 identity must be observed and strictly reused",
    )
    return dict(events=events, original_capture_matched=True)


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered phase is required")
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(OUTPUT / "dry.json")
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "Passing source-bound CPU QA and zero-physics preflight are required",
    )
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    data, sources = prepare(CASE), source_refs()
    prefix.require(
        dry["config_ref"] == config_ref and dry["identities"] == [data["identity"]] and dry["source_refs"] == sources,
        "Dry inputs and source identities must match",
    )
    producer_refs = {}
    if phase == "checker":
        producer = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(producer["complete"] is producer["passed"] is True, "A closed producer is required")
        producer_refs["4"] = producer["case"]["branch_ref"]
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            identities={"4": data["identity"]},
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            eligible_cases=[4],
            producer_branch_refs=producer_refs,
            automatic_retry=False,
        ),
    )
    lifecycle = run_owned_workers(
        [
            dict(
                key="4",
                argv=[sys.executable, "-m", MODULE, "--worker", phase, "4"],
                log=directory / "case04/worker.log",
            )
        ],
        cwd=ROOT,
        timeout_seconds=config["worker_timeout_seconds"],
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(directory / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        owner_closed=False,
        sources_unchanged=False,
        phase=phase,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        case=dict(status="not_run"),
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        workers = lifecycle["workers"]
        path = directory / "case04/worker/result.json"
        if path.is_file():
            result["case"]["available_branch_ref"] = prefix.reference(path)
        prefix.require(
            len(workers) == 1
            and workers[0]["returncode"] == 0
            and workers[0]["actual_wait"] is workers[0]["reaped"] is True
            and workers[0]["signal_sent"] is False
            and path.is_file(),
            "The owned wine worker must close normally",
        )
        branch_ref = prefix.reference(path)
        branch = prefix.read(branch_ref)
        counts = original.validate_branch(branch, phase, CASE)
        baseline = data["initial"] + [item for window in data["windows"] for item in window["periods"]]
        actual = original.periods(branch)
        counts["original_prefix_max_difference"] = comparison.compare_periods(baseline, actual[: len(baseline)])
        counts["cache_boundary"] = cached_boundary(branch, data)
        if phase == "checker":
            produced = prefix.read(producer_refs["4"])
            counts["replay_max_difference"] = comparison.compare_periods(original.periods(produced), actual)
            for left, right in zip(produced["reference_captures"], branch["reference_captures"], strict=True):
                captures.compare_capture(left, right)
        result["case"].update(status="closed", branch_ref=branch_ref, counts=counts)
        result["passed"] = True
    except Exception as error:
        result["case"].update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
    finally:
        result["owner_closed"] = len(lifecycle["workers"]) == 1 and all(
            item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"]
        )
        result["sources_unchanged"] = (
            snapshot(ROOT)["sources"] == before
            and source_refs() == sources
            and prepare(CASE)["identity"] == data["identity"]
        )
        result["complete"] = (
            result["owner_closed"] and result["sources_unchanged"] and lifecycle["launch_error"] is None
        )
        result["passed"] = result["passed"] and result["complete"]
        print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The wine cache-recovery phase did not close")


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    args = parser.parse_args()
    if args.dry:
        original.dry_run()
    elif args.phase:
        phase_run(args.phase)
    else:
        original.worker(args.worker[0], int(args.worker[1]))


if __name__ == "__main__":
    main()
