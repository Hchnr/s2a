"""Run the registered, CPU-only retrospective TCP residual cohort."""

import argparse
import sys
from pathlib import Path

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import evaluate_action_conditioned_tcp as scoring

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/action_conditioned_tcp_residual_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/action_conditioned_tcp_residual_20261116.md"
OUTPUT = ROOT / "outputs/action_conditioned_tcp_residual_20261116"
QA = ROOT / "outputs/qa/action_conditioned_tcp_cohort_20261116/result.json"
CONFIG_SHA = "39c51b1b35d71410b51c5e2535908dedd55ff6a43fbd0ae3fa9043a979128dbe"
PLAN_SHA = "d2496c509d1828634ff53d609607d0d5a06c05f70b15440804b0b0848e35c4f1"
INDICES = (0, 2, 4, 6, 8, 10)
SEEDS = (42, 43, 42, 43, 42, 43)
PROTOCOLS = ("adaptive", "fixed", "correct", "wine")
EXPECTED_PERIODS = dict(adaptive=1025, fixed=270, correct=4778, wine=144)
EXPECTED_CASES = dict(adaptive=6, fixed=6, correct=18, wine=1)
MODULE = "scripts.tracking.response_models.run_action_conditioned_tcp_cohort"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/tracking/response_models/evaluate_action_conditioned_tcp.py",
            ROOT / "scripts/tracking/response_models/audit_causal_tcp_prediction.py",
            ROOT / "scripts/simulation/runtime/replay_multicase_development_prefix.py",
            ROOT / "src/s2a/infrastructure/owned_processes.py",
            ROOT / "src/s2a/infrastructure/publication.py",
            ROOT / "src/s2a/research/response_models/causal/action_conditioned_tcp_residual.py",
            ROOT / "src/s2a/research/response_models/causal/tcp_prediction_innovation.py",
            ROOT / "tests/research/response_models/test_action_conditioned_tcp_residual.py",
            ROOT / "tests/research/response_models/test_action_conditioned_tcp_evaluation.py",
            ROOT / "tests/research/response_models/test_action_conditioned_tcp_cohort.py",
        )
    ]


def zero_difference(difference):
    return isinstance(difference, dict) and bool(difference) and all(value == 0 for value in difference.values())


def load_manifest():
    """Build the exact case roster from immutable independent replay audits."""
    prefix.environment()
    consumed = {}
    config = prefix.read(dict(path=str(CONFIG), sha256=CONFIG_SHA), consumed)
    prefix.checked(dict(path=str(PLAN), sha256=PLAN_SHA), consumed)
    prefix.require(
        config["schema"] == "s2a.action-conditioned-tcp-residual.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["adaptive_case_indices"] == config["fixed_case_indices"] == list(INDICES)
        and config["correct_reference_cases"] == 18
        and config["expected_adaptive_periods"] == 1025
        and config["expected_fixed_periods"] == 270
        and config["expected_correct_reference_periods"] == 4778
        and config["expected_wine_candidate_periods"] == 144
        and config["history_window"] == 20
        and config["ridge"] == 1.0
        and config["arm_action_dimensions"] == 6
        and config["position_cap_m"] == 0.01
        and config["rotation_cap_rad"] == 0.05
        and config["max_workers"] == 31
        and config["worker_timeout_seconds"] == 1800
        and config["new_world_model_calls"] == config["new_physics_steps"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The fixed retrospective scope and zero-work budget are required",
    )
    values = {}
    for key in (
        "adaptive_audit",
        "fixed_audit",
        "correct_reference_audit",
        "wine_candidate_producer",
        "wine_candidate_checker",
    ):
        ref = config[key]
        values[key] = prefix.read(dict(path=str(ROOT / ref["path"]), sha256=ref["sha256"]), consumed)
    roster = []
    for protocol in ("adaptive", "fixed"):
        audit = values[f"{protocol}_audit"]
        prefix.require(
            audit["complete"] is True and [item["case_index"] for item in audit["cases"]] == list(INDICES),
            "The WAM audit roster must be complete and ordered",
        )
        for index, seed, item in zip(INDICES, SEEDS, audit["cases"], strict=True):
            key = "independent_max_absolute_difference" if protocol == "adaptive" else "replay_max_absolute_difference"
            prefix.require(zero_difference(item[key]), "WAM producer and independent replay must agree")
            roster.append(
                dict(
                    key=f"{protocol}_{index:02d}",
                    protocol=protocol,
                    index=index,
                    seed=seed,
                    expected_periods=item["counts"]["control_steps"],
                    producer_ref=item["producer_ref"],
                    checker_ref=item["checker_ref"],
                )
            )
    audit = values["correct_reference_audit"]
    prefix.require(
        audit["complete"] is audit["passed"] is True
        and audit["control_qualified"] is False
        and audit["registered_cases"] == 18
        and len(audit["cases"]) == 18,
        "The complete 18-case correct-reference audit is required",
    )
    for index, item in enumerate(audit["cases"]):
        case = item["case"]
        prefix.require(
            case["index"] == index
            and item["producer_status"] == item["checker_status"] == "closed"
            and item["replayed"] is True
            and zero_difference(item["replay_max_difference"]),
            "Every correct-reference result must have an independently replayed prefix",
        )
        roster.append(
            dict(
                key=f"correct_{index:02d}",
                protocol="correct",
                index=index,
                seed=case["seed"],
                task_id=case["task_id"],
                expected_periods=item["summary"]["completed_periods"],
                registered_periods=case["periods"],
                producer_ref=item["producer_ref"],
                checker_ref=item["checker_ref"],
            )
        )
    producer, checker = values["wine_candidate_producer"], values["wine_candidate_checker"]
    for owner, phase in ((producer, "producer"), (checker, "checker")):
        prefix.require(
            owner["complete"] is owner["passed"] is owner["owner_closed"] is owner["sources_unchanged"] is True
            and owner["phase"] == phase
            and owner["case"]["status"] == "closed"
            and owner["case"]["counts"]["control_steps"] == 144
            and owner["case"]["counts"]["probe_steps"] == 1
            and owner["case"]["counts"]["probe_pose_caps_satisfied"] is False,
            "The separately audited wine probe must retain its failed physical gate",
        )
    prefix.require(zero_difference(checker["case"]["counts"]["replay_max_difference"]), "Wine replay must agree")
    roster.append(
        dict(
            key="wine_04",
            protocol="wine",
            index=4,
            seed=42,
            expected_periods=144,
            producer_ref=producer["case"]["branch_ref"],
            checker_ref=checker["case"]["branch_ref"],
        )
    )
    validate_roster(roster)
    return config, roster, consumed


def validate_roster(roster):
    prefix.require(len({item["key"] for item in roster}) == 31, "All case identities must be distinct")
    for protocol in PROTOCOLS:
        cases = [item for item in roster if item["protocol"] == protocol]
        prefix.require(
            len(cases) == EXPECTED_CASES[protocol]
            and sum(item["expected_periods"] for item in cases) == EXPECTED_PERIODS[protocol]
            and all(item["expected_periods"] > 1 for item in cases),
            "Every protocol must retain its registered full observed denominator",
        )


def load_periods(spec):
    """Check both immutable inputs, then extract only observed producer periods."""
    seen = {}
    prefix.checked(spec["checker_ref"], seen)
    branch = prefix.read(spec["producer_ref"], seen)
    if spec["protocol"] == "fixed":
        prefix.require(branch["complete"] is branch["passed"] is True, "The fixed-reference owner must close")
        branch = prefix.read(branch["branch_ref"], seen)
    prefix.require(branch["complete"] is branch["passed"] is True, "The recorded producer must close")
    continuation = branch["runtime"]["continuation"]
    if spec["protocol"] in ("adaptive", "fixed"):
        prefix.require(
            branch["case"]["case_index"] == spec["index"] and branch["case"]["sampling_seed"] == spec["seed"],
            "The WAM branch identity must match the audit",
        )
        segments = continuation[spec["protocol"]]["segments"]
        periods = continuation["first_window"]["periods"] + [
            period for segment in segments for period in segment["continuation"]["periods"]
        ]
    elif spec["protocol"] == "correct":
        periods = [period for segment in continuation["segments"] for period in segment["continuation"]["periods"]]
    else:
        periods = (
            continuation["initial"]["periods"]
            + [period for window in continuation["windows"] for period in window["continuation"]["periods"]]
            + continuation["probe"]["periods"]
        )
    prefix.require(
        len(periods) == spec["expected_periods"]
        and [period["step"] for period in periods] == list(range(len(periods))),
        "Every observed period must be consecutive and remain in the denominator",
    )
    return periods, seen


def aggregate(roster, cases):
    """Reject missing cases and pool only within each registered protocol."""
    validate_roster(roster)
    prefix.require(set(cases) == {spec["key"] for spec in roster}, "All registered cases are required")
    summaries = {}
    for protocol in PROTOCOLS:
        selected = [spec for spec in roster if spec["protocol"] == protocol]
        rows = []
        for spec in selected:
            case = cases[spec["key"]]
            prefix.require(
                case["complete"] is case["passed"] is True
                and case["identity"] == spec
                and len(case["rows"]) == spec["expected_periods"],
                "Each case must close with its full observed prefix",
            )
            rows.extend(case["rows"])
        summary = scoring.summarize(rows)
        prefix.require(
            summary["recorded_periods"] == EXPECTED_PERIODS[protocol]
            and summary["scored_periods"] == EXPECTED_PERIODS[protocol] - EXPECTED_CASES[protocol]
            and summary["ridge_fits"] == summary["scored_periods"],
            "The history and ridge-fit denominators must be exact",
        )
        summaries[protocol] = summary
    return summaries


def dry_run():
    _config, roster, consumed = load_manifest()
    for spec in roster:
        for key in ("producer_ref", "checker_ref"):
            path = Path(spec[key]["path"])
            prefix.require(path.is_file() and path.resolve().is_relative_to(ROOT), "A recorded input is missing")
    result = dict(
        complete=True,
        passed=True,
        roster=roster,
        source_refs=source_refs(),
        input_sha256=consumed,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT / "dry.json", result), flush=True)


def worker(key):
    _config, roster, _consumed = load_manifest()
    spec = next((item for item in roster if item["key"] == key), None)
    prefix.require(spec is not None, "A registered case key is required")
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    prefix.require(
        registration["roster"] == roster and registration["source_refs"] == source_refs(), "Registration changed"
    )
    result = dict(complete=False, passed=False, identity=spec, rows=[], automatic_retry=False)
    try:
        periods, seen = load_periods(spec)
        result["rows"] = scoring.evaluate(periods)
        result["summary"] = scoring.summarize(result["rows"])
        for path, digest in seen.items():
            prefix.checked(dict(path=path, sha256=digest))
        prefix.require(registration["source_refs"] == source_refs(), "Scoring sources changed")
        result.update(complete=True, passed=True, input_sha256=seen)
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(OUTPUT / key / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The registered case did not close")


def execute():
    config, roster, consumed = load_manifest()
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(OUTPUT / "dry.json")
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before, sources = snapshot(ROOT)["sources"], source_refs()
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["roster"] == roster
        and dry["source_refs"] == sources
        and dry["input_sha256"] == consumed,
        "Matching CPU QA, preflight, sources and registered inputs are required",
    )
    registration_ref = publish_result(
        OUTPUT / "registration.json", dict(roster=roster, source_refs=sources, qa_ref=qa_ref, dry_ref=dry_ref)
    )
    specs = [
        dict(
            key=item["key"],
            argv=[sys.executable, "-m", MODULE, "--worker", item["key"]],
            log=OUTPUT / item["key"] / "worker.log",
        )
        for item in roster
    ]
    lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(OUTPUT / "workers.json", lifecycle)
    worker_by_key = {item["key"]: item for item in lifecycle["workers"]}
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        cases={},
        protocol_summaries={},
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
        shared_initial_states_and_prefixes=True,
        independent_samples_claimed=False,
        retrospective_selection=True,
    )
    try:
        cases = {}
        for spec in roster:
            key, path = spec["key"], OUTPUT / spec["key"] / "result.json"
            entry = result["cases"][key] = dict(status="not_run")
            if path.is_file():
                entry["result_ref"] = prefix.reference(path)
            info = worker_by_key.get(key)
            if info is None:
                entry["reason"] = "worker_not_launched"
                continue
            if (
                info["returncode"] != 0
                or not info["actual_wait"]
                or not info["reaped"]
                or info["signal_sent"]
                or not path.is_file()
            ):
                entry["reason"] = "worker_failed_or_missing_result"
                continue
            case = prefix.read(entry["result_ref"])
            if case["complete"] is not True or case["passed"] is not True:
                entry.update(status="failed", error=case.get("error"))
                continue
            cases[key] = case
            entry.update(status="closed", summary=case["summary"])
        summaries = aggregate(roster, cases)
        unchanged = snapshot(ROOT)["sources"] == before and source_refs() == sources
        _config, after_roster, after_inputs = load_manifest()
        owner_closed = len(lifecycle["workers"]) == 31 and all(
            item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"]
        )
        prefix.require(
            unchanged
            and after_roster == roster
            and after_inputs == consumed
            and owner_closed
            and lifecycle["launch_error"] is None,
            "The owner, source identities and inputs must remain complete",
        )
        result.update(
            complete=True, passed=True, owner_closed=True, sources_unchanged=True, protocol_summaries=summaries
        )
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
    print(publish_result(OUTPUT / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("The full registered retrospective cohort did not close")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--worker")
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
