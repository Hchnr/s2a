"""Evaluate the opt-in cached-seed candidate on the fixed correct-reference cohort."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_correct_reference_cohort as cohort
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/cached_seed_preference_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/cached_seed_preference_20261116.md"
OUTPUT = ROOT / "outputs/cached_seed_preference_20261116"
DRY = ROOT / "outputs/cached_seed_preference_dry_20261116.json"
QA = ROOT / "outputs/qa/cached_seed_preference_final_20261116/result.json"
STATIC = {
    "path": str(ROOT / "outputs/cached_seed_first_divergence_20261116/result.json"),
    "sha256": "d7765166c7593b10c8d7b6c23b229ef9d14fc58d193a51c24f1c600db3bd5116",
}
MODULE = "scripts.simulation.runtime.run_cached_seed_preference"
CANDIDATE_MODULE = "s2a.research.joint_control.preview.cached_seed_preference"


def validate_config(config):
    fixed = dict(
        schema="s2a.cached-seed-preference.v1",
        output="outputs/cached_seed_preference_20261116",
        selection="prefer_exact_seed_only_if_cached_joint_safe_and_margin_eligible",
        candidate_cases=18,
        registered_periods_per_phase=4907,
        max_inner_steps_per_phase=122675,
        max_workers_per_phase=18,
        max_models_per_phase=54,
        max_contexts_per_phase=54,
        max_quality_responses_per_period=320,
        max_baseline_responses_per_period=480,
        max_gripper_sequences_per_period=6561,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        joint_buffer_rad=0.05,
        arm_rate_limit=0.25,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if any(config.get(key) != value or type(config.get(key)) is not type(value) for key, value in fixed.items()):
        raise ValueError("The registered candidate, cohort, safety gates and budgets must remain fixed")


def source_refs():
    return cohort.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/preview/cached_seed_preference.py",
            ROOT / "scripts/verification/trajectory_search/audit_cached_seed_first_divergence.py",
            ROOT / "tests/research/joint_control/preview/test_cached_seed_preference.py",
            ROOT / "tests/research/native_runtime/test_cached_seed_preference_runtime.py",
        )
    ]


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    roster_ref = preparation.configured_ref(config["original_cohort_config"])
    baseline_ref = preparation.configured_ref(config["baseline_cohort_audit"])
    baseline, static = prefix.read(baseline_ref), prefix.read(STATIC)
    static_registration = prefix.read(static["registration_ref"])
    data = preparation.load_inputs()
    static_cases = [prefix.read(ref) for ref in static["case_refs"]]
    prefix.require(
        roster_ref == data["config_ref"]
        and baseline_ref == static_registration["cohort_audit_ref"]
        and baseline["complete"] is baseline["passed"] is static["complete"] is static["passed"] is True
        and baseline["registered_cases"] == len(data["cases"]) == len(static_cases) == 18
        and baseline["qualified_cases"] == 14
        and static["source_unchanged"] is True
        and static["seed_eligible"] == static["selection_changed"] == 18
        and [row["case_index"] for row in static_cases] == [case["spec"]["index"] for case in data["cases"]]
        and all(
            row["complete"] is row["passed"] is True and row["metrics"]["preferred_independent_witness"] is True
            for row in static_cases
        ),
        "The original cohort and all first-divergence cached-seed audits must be closed",
    )
    return dict(config=config, config_ref=config_ref, baseline_ref=baseline_ref, static_ref=STATIC, cohort=data)


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        baseline_ref=inputs["baseline_ref"],
        static_ref=inputs["static_ref"],
        cohort=preparation.input_receipt(inputs["cohort"]),
    )


def checker_isolation():
    original.checker_isolation()
    prefix.require(CANDIDATE_MODULE not in sys.modules, "The checker must not import the seed-preferred candidate")


def summarize_seed_traces(traces):
    """Keep the soft future-quality tradeoff and zero-extra-response claim visible."""
    if not isinstance(traces, list):
        raise ValueError("A list of actual seed-selection traces is required")
    increases = []
    for row in traces:
        if (
            type(row.get("seed_in_eligible_cache")) is not bool
            or type(row.get("selected_seed")) is not bool
            or type(row.get("selected_changed")) is not bool
            or row.get("extra_model_responses") != 0
            or (row["selected_changed"] and not row["selected_seed"])
            or row["selected_seed"] != row["seed_in_eligible_cache"]
        ):
            raise ValueError("Seed traces must preserve cache eligibility and response accounting")
        delta = row["future_slack_increase"]
        if delta is not None:
            if not np.isfinite(delta) or delta < -1e-10:
                raise ValueError("The original future-slack ordering must be recorded accurately")
            increases.append(float(delta))
    return dict(
        searches=len(traces),
        seed_in_eligible_cache=sum(row["seed_in_eligible_cache"] for row in traces),
        selected_changed=sum(row["selected_changed"] for row in traces),
        maximum_future_slack_increase=max(increases) if increases else None,
        extra_model_responses=sum(row["extra_model_responses"] for row in traces),
    )


def dry_run():
    prefix.environment()
    original.require_no_simulator_import()
    inputs = load_inputs()
    checker_isolation()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


def worker(phase, index):
    prefix.require(
        phase in ("producer", "checker") and type(index) is int and 0 <= index < 18,
        "One fixed case and physical phase are required",
    )
    print(json.dumps(dict(event="worker_started", phase=phase, case_index=index)), flush=True)
    directory = OUTPUT / phase / f"case{index:02d}" / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        prefix.require(
            registration["phase"] == phase
            and index in registration["eligible_cases"]
            and registration["source_refs"] == source_refs(),
            "The registered case and source identity must match",
        )
        shared = preparation.shared_inputs()
        prefix.require(
            shared["receipt"] == registration["inputs"]["cohort"]["shared_receipt"], "Shared assets must match"
        )
        identity = registration["inputs"]["cohort"]["cases"][index]
        inputs = preparation.case_inputs(identity, shared)
        prefix.require(
            inputs["reset_input_sha256"] == identity["reset_input_sha256"]
            and inputs["target_parity"] == identity["target_parity"],
            "The registered reset and reference must match",
        )
        recorded = None
        if phase == "checker":
            checker_isolation()
            produced = prefix.read(registration["producer_branch_refs"][str(index)])
            cohort.validate_branch(produced, "producer", inputs["spec"]["periods"])
            recorded = produced["runtime"]["continuation"]
        cv2 = shared["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        reset, case = inputs["reset"], inputs["case"]
        adapter = CorrectReferenceResetRuntime(
            expected=reset["expected"],
            expected_receipt=reset["expected_receipt"],
            controller=case["controller"],
            tool=case["tool"],
            anchor=reset["anchor"],
            opening=reset["opening"],
        )
        factory = None
        if phase == "producer":
            from s2a.research.joint_control.preview.cached_seed_preference import CachedSeedPreferredAudit
            from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
            from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

            def factory(env, terminal, receipt):
                return build_segmented_reset_policy(
                    env, policy_inputs(case, terminal), receipt, adapter_factory=CachedSeedPreferredAudit
                )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            return run_correct_reference_continuation(
                env,
                terminal,
                observers,
                receipt,
                timeline=inputs["timeline"],
                tool=case["tool"],
                target_object=inputs["target_object"],
                phase=phase,
                score_gripper=original.measured_gripper,
                compare_score=original.compare_score,
                policy_factory=factory,
                recorded=recorded,
                progress=progress,
            )

        def progress(receipt):
            print(json.dumps(dict(case_index=index, **previous.progress_event(phase, receipt))), flush=True)

        runtime = adapter.run_feedback(
            inputs["runtime_config"],
            inputs["record"],
            inputs["record_xml"],
            directory / "runtime",
            shared["asset_refs"],
            registered_seed=inputs["spec"]["seed"],
            continuation=continuation,
            max_control_steps=inputs["spec"]["periods"],
            progress=progress,
        )
        traces = None
        if phase == "checker":
            checker_isolation()
        else:
            traces = summarize_seed_traces(runtime["segment_receipts"][0]["seed_preference_traces"])
        return dict(
            complete=True,
            passed=True,
            case=inputs["spec"],
            runtime=runtime,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=phase == "checker",
            seed_trace_summary=traces,
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def eligible_checker_cases(producer, cases):
    return [
        case["spec"]["index"]
        for case in cases
        if producer["cases"].get(str(case["spec"]["index"]), {}).get("status") == "closed"
    ]


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered physical phase is required")
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "Passing source-bound CPU QA and zero-physics preflight are required",
    )
    inputs, sources = load_inputs(), source_refs()
    identity = input_receipt(inputs)
    prefix.require(dry["inputs"] == identity and dry["source_refs"] == sources, "Dry inputs and sources must match")
    cases, config = inputs["cohort"]["cases"], inputs["config"]
    producer_refs = {}
    if phase == "checker":
        producer = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            producer["owner_closed"] is producer["sources_unchanged"] is True,
            "The producer owner must close with unchanged sources",
        )
        eligible = eligible_checker_cases(producer, cases)
        producer_refs = {str(index): producer["cases"][str(index)]["branch_ref"] for index in eligible}
        for ref in producer_refs.values():
            prefix.checked(ref)
    else:
        eligible = [case["spec"]["index"] for case in cases]
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            inputs=identity,
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            eligible_cases=eligible,
            producer_branch_refs=producer_refs,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(
            key=str(index),
            argv=[sys.executable, "-m", MODULE, "--worker", phase, str(index)],
            log=directory / f"case{index:02d}/worker.log",
        )
        for index in eligible
    ]
    lifecycle = (
        run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
        if specs
        else dict(workers=[], launch_error=None)
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(directory / "workers.json", lifecycle)
    workers = {int(item["key"]): item for item in lifecycle["workers"]}
    outcomes = {}
    result = dict(
        complete=False,
        passed=False,
        owner_closed=False,
        sources_unchanged=False,
        phase=phase,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        cases=outcomes,
        eligible_cases=eligible,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for case in cases:
            index = case["spec"]["index"]
            entry = dict(status="not_run", expected_periods=case["spec"]["periods"])
            outcomes[str(index)] = entry
            if index not in eligible:
                entry["reason"] = "producer_not_closed"
                continue
            worker_info = workers.get(index)
            if worker_info is None:
                entry["reason"] = "worker_not_launched"
                continue
            result_path = directory / f"case{index:02d}/worker/result.json"
            if result_path.is_file():
                entry["available_branch_ref"] = prefix.reference(result_path)
            try:
                prefix.require(
                    worker_info["returncode"] == 0
                    and worker_info["actual_wait"] is worker_info["reaped"] is True
                    and worker_info["signal_sent"] is False
                    and result_path.is_file(),
                    "The owned worker must close",
                )
                branch_ref = prefix.reference(result_path)
                branch = prefix.read(branch_ref)
                prefix.require(branch["case"] == case["spec"], "Worker case identity must match")
                counts = cohort.validate_branch(branch, phase, case["spec"]["periods"])
                if phase == "checker":
                    candidate = prefix.read(producer_refs[str(index)])
                    counts["replay_max_difference"] = comparison.compare_periods(
                        previous.periods(candidate), previous.periods(branch)
                    )
                    prefix.require(
                        candidate["runtime"]["continuation"]["stop"] == branch["runtime"]["continuation"]["stop"],
                        "The recorded producer stop must be retained",
                    )
                else:
                    counts["seed_trace_summary"] = branch["seed_trace_summary"]
                entry.update(status="closed", branch_ref=branch_ref, counts=counts)
            except Exception as error:
                entry.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        owner_closed = all(item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"])
        source_unchanged = snapshot(ROOT)["sources"] == before and input_receipt(load_inputs()) == identity
        result.update(
            owner_closed=owner_closed,
            sources_unchanged=source_unchanged,
            complete=owner_closed and source_unchanged and lifecycle["launch_error"] is None,
            passed=owner_closed
            and source_unchanged
            and lifecycle["launch_error"] is None
            and len(outcomes) == 18
            and all(item["status"] == "closed" for item in outcomes.values()),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("One or more candidate workers did not close successfully")


def audit():
    prefix.environment()
    inputs = load_inputs()
    producer_ref, checker_ref = (
        prefix.reference(OUTPUT / "producer/result.json"),
        prefix.reference(OUTPUT / "checker/result.json"),
    )
    producer, checker, baseline = (
        prefix.read(producer_ref),
        prefix.read(checker_ref),
        prefix.read(inputs["baseline_ref"]),
    )
    prefix.require(
        producer["owner_closed"]
        is checker["owner_closed"]
        is producer["sources_unchanged"]
        is checker["sources_unchanged"]
        is True,
        "Both physical phase owners must close with unchanged sources",
    )
    regs = [prefix.read(group["registration_ref"]) for group in (producer, checker)]
    identity = input_receipt(inputs)
    prefix.require(
        all(reg["inputs"] == identity and reg["source_refs"] == source_refs() for reg in regs),
        "Both phases must use the fixed candidate and cohort sources",
    )
    cases = []
    for item, old in zip(inputs["cohort"]["cases"], baseline["cases"], strict=True):
        spec, index = item["spec"], item["spec"]["index"]
        p, c = producer["cases"][str(index)], checker["cases"][str(index)]
        replayed = p["status"] == c["status"] == "closed"
        summary = p.get("counts", {}).get("summary")
        replay_max, trace = None, None
        if replayed:
            left, right = prefix.read(p["branch_ref"]), prefix.read(c["branch_ref"])
            pc = cohort.validate_branch(left, "producer", spec["periods"])
            cc = cohort.validate_branch(right, "checker", spec["periods"])
            prefix.require(
                pc["summary"] == summary
                and cc["summary"] == c["counts"]["summary"]
                and left["runtime"]["continuation"]["stop"] == right["runtime"]["continuation"]["stop"],
                "Physical summaries and reported stop must match",
            )
            replay_max = comparison.compare_periods(previous.periods(left), previous.periods(right))
            prefix.require(replay_max == c["counts"]["replay_max_difference"], "Independent replay maxima must match")
            trace = summarize_seed_traces(left["runtime"]["segment_receipts"][0]["seed_preference_traces"])
            prefix.require(
                trace == left["seed_trace_summary"] == p["counts"]["seed_trace_summary"], "Seed accounting must match"
            )
        cases.append(
            dict(
                case=spec,
                baseline_qualified=old["qualified"],
                producer_status=p["status"],
                checker_status=c["status"],
                producer_ref=p.get("branch_ref"),
                checker_ref=c.get("branch_ref"),
                summary=summary,
                replayed=replayed,
                replay_max_difference=replay_max,
                seed_trace_summary=trace,
                qualified=cohort.qualifies(summary, replayed, inputs["config"]),
            )
        )
    count = sum(case["qualified"] for case in cases)
    print(
        publish_result(
            OUTPUT / "audit.json",
            dict(
                complete=True,
                passed=True,
                control_qualified=count == len(cases) == 18,
                registered_cases=18,
                qualified_cases=count,
                baseline_qualified_cases=baseline["qualified_cases"],
                cases=cases,
                producer_ref=producer_ref,
                checker_ref=checker_ref,
                static_ref=STATIC,
                world_model_calls=0,
                fresh_cases=0,
                physics_steps=0,
            ),
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.phase:
        phase_run(args.phase)
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        audit()


if __name__ == "__main__":
    main()
