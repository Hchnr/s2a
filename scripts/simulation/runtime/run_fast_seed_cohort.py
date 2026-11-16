"""Run the fixed 18-case correct-reference comparison with seed-first search."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_broader_recovery_pilot as broader
from scripts.simulation.runtime import run_cached_seed_preference as base
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_correct_reference_cohort as cohort
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/fast_seed_cohort_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/fast_seed_cohort_20261116.md"
OUTPUT = ROOT / "outputs/fast_seed_cohort_20261116"
DRY = ROOT / "outputs/fast_seed_cohort_dry_20261116.json"
QA = ROOT / "outputs/qa/fast_seed_cohort_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_fast_seed_cohort"
CANDIDATE_MODULE = "s2a.research.joint_control.preview.fast_seed_preference"
UNBOUNDED_AUDIT = dict(
    path="outputs/cached_seed_preference_20261116/audit.json",
    sha256="ae8358fc25a9675d208f91cd8e7c52f4d20f120e9c314ee18755b97472744525",
)
FIXED_STATE = dict(
    path="outputs/fast_seed_search_20261116/result.json",
    sha256="4f45d867b1a48cc23c2d0fd0a34365d7c9b9fc5a8e31c0b27524744238030387",
)


def validate_config(config):
    """Require the original controls and budgets plus this one fixed variant."""
    expected = prefix.read(prefix.reference(base.CONFIG)).copy()
    expected.update(
        schema="s2a.fast-seed-cohort.v1",
        output="outputs/fast_seed_cohort_20261116",
        cached_seed_audit=UNBOUNDED_AUDIT,
        fixed_state_comparison=FIXED_STATE,
        selection="early_seed_if_joint_safe_and_two_pose_margin_else_original_cached_search",
        gpu_calls=0,
    )
    if (
        not isinstance(config, dict)
        or set(config) != set(expected)
        or any(config[key] != value or type(config[key]) is not type(value) for key, value in expected.items())
    ):
        raise ValueError("The fixed cohort, safety gates, budgets and source identities must match registration")


def source_refs():
    return base.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/preview/fast_seed_preference.py",
            ROOT / "tests/research/joint_control/preview/test_fast_seed_preference.py",
            ROOT / "tests/research/native_runtime/test_fast_seed_cohort.py",
        )
    ]


def load_inputs():
    inputs = base.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    unbounded_ref = preparation.configured_ref(config["cached_seed_audit"])
    fixed_ref = preparation.configured_ref(config["fixed_state_comparison"])
    unbounded = prefix.read(unbounded_ref)
    fixed = prefix.read(fixed_ref)
    cases = inputs["cohort"]["cases"]
    prefix.require(
        unbounded["complete"] is unbounded["passed"] is True
        and fixed["complete"] is fixed["passed"] is fixed["sources_unchanged"] is True
        and fixed["totals"]["solver_completed"] == fixed["totals"]["audits_completed"] == 12
        and unbounded["registered_cases"] == len(unbounded["cases"]) == len(cases) == 18
        and unbounded["qualified_cases"] == 16
        and unbounded["baseline_qualified_cases"] == 14
        and all(
            row["case"] == case["spec"]
            and row["producer_status"] == row["checker_status"] == "closed"
            and row["replayed"] is True
            for row, case in zip(unbounded["cases"], cases, strict=True)
        ),
        "The full original cohort and its 16/18 independently replayed comparison are required",
    )
    inputs.update(config=config, config_ref=config_ref, unbounded_ref=unbounded_ref, fixed_ref=fixed_ref)
    return inputs


def input_receipt(inputs):
    return dict(**base.input_receipt(inputs), unbounded_ref=inputs["unbounded_ref"], fixed_ref=inputs["fixed_ref"])


def checker_isolation():
    base.checker_isolation()
    prefix.require(CANDIDATE_MODULE not in sys.modules, "The checker cannot load the fast candidate")


def summarize_seed_traces(traces):
    """Check early guards and retain unknown comparative quality."""
    if not isinstance(traces, list):
        raise ValueError("A list of actual seed-selection traces is required")
    for row in traces:
        early, continued = row.get("early_selected"), row.get("original_solver_continued")
        domain, margin = row.get("seed_domain_accepted"), row.get("seed_margin_accepted")
        if (
            type(early) is not bool
            or type(continued) is not bool
            or early == continued
            or type(row.get("additional_model_responses")) is not int
            or row["additional_model_responses"] != 0
        ):
            raise ValueError("Early selection, continuation and response accounting must agree")
        if early:
            quality = row.get("selected_future_slack")
            if (
                domain is not True
                or margin is not True
                or row.get("baseline_future_slack", False) is not None
                or row.get("future_slack_increase", False) is not None
                or not isinstance(quality, (int, float))
                or not np.isfinite(quality)
                or quality < 0
            ):
                raise ValueError("Early seed requires both guards and unknown comparative quality")
        else:
            reached = row.get("seed_probe_reached")
            if (
                type(reached) is not bool
                or (reached and (type(domain) is not bool or type(margin) is not bool or (domain and margin)))
                or (not reached and (domain is not None or margin is not None))
            ):
                raise ValueError("Original search continuation must accurately report the seed probe")
    return dict(
        searches=len(traces),
        early_selected=sum(row["early_selected"] for row in traces),
        original_solver_continued=sum(row["original_solver_continued"] for row in traces),
        unknown_baseline_quality=sum(row["early_selected"] for row in traces),
        additional_model_responses=sum(row["additional_model_responses"] for row in traces),
    )


def instrument_policy_selection(policy):
    """Measure the whole policy select call in this isolated producer process."""
    cls = type(policy)
    select = cls.select

    def timed_select(self, *args, **kwargs):
        started = time.perf_counter()
        result = select(self, *args, **kwargs)
        elapsed = time.perf_counter() - started
        if isinstance(result, dict) and "command" in result:
            result["full_policy_select_seconds"] = elapsed
        return result

    cls.select = timed_select


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
            from s2a.research.joint_control.preview.fast_seed_preference import FastSeedPreferredAudit
            from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
            from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

            def factory(env, terminal, receipt):
                policy, owner, prior = build_segmented_reset_policy(
                    env, policy_inputs(case, terminal), receipt, adapter_factory=FastSeedPreferredAudit
                )
                instrument_policy_selection(policy)
                return policy, owner, prior

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
            traces = summarize_seed_traces(runtime["segment_receipts"][0]["fast_seed_traces"])
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


def configured(function):
    """Reuse the recorded cohort owner without changing its historical globals."""
    namespace = function.__globals__.copy()
    namespace.update(
        OUTPUT=OUTPUT,
        DRY=DRY,
        QA=QA,
        MODULE=MODULE,
        source_refs=source_refs,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        checker_isolation=checker_isolation,
        summarize_seed_traces=summarize_seed_traces,
    )
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return clone


def strict_preflight():
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa.get("configured_parallelism") == 64
        and qa.get("all_tests_passed") is True
        and qa.get("aggregate_node_partition_exact") is True
        and qa.get("collection_node_count", 0) > 0
        and qa.get("skipped_count") == qa.get("xfail_count") == qa.get("xpass_count") == 0
        and qa.get("returncode") == 0,
        "The full registered CPU regression must pass without omissions",
    )


def record_resources(phase):
    """Keep observed resources separate from absent or partial receipts."""
    group_ref = prefix.reference(OUTPUT / phase / "result.json")
    group = prefix.read(group_ref)
    lifecycle = prefix.read(group["lifecycle_ref"])
    workers = {int(row["key"]): row for row in lifecycle["workers"]}
    rows = []
    for index in range(18):
        outcome = group["cases"][str(index)]
        ref = outcome.get("available_branch_ref")
        observed = broader.observed_resources(prefix.read(ref)) if ref is not None else None
        rows.append(
            dict(
                case_index=index,
                status=outcome["status"],
                worker_launched=index in workers,
                worker_returncode=workers[index]["returncode"] if index in workers else None,
                observed=observed,
                unobserved_after_launch=index in workers and observed is None,
            )
        )
    known = [row["observed"] for row in rows if row["observed"] is not None]
    lower_bound = dict(
        control_steps_attempted=sum(row["control_steps_attempted"] for row in known),
        control_steps_completed=sum(row["control_steps_completed"] for row in known),
        counters={
            key: {field: sum(row["counters"][key][field] for row in known) for field in ("attempted", "completed")}
            for key in ("mj_step", "mj_step2", "model_construct", "context")
        },
    )
    config = load_inputs()["config"]
    prefix.require(
        lower_bound["control_steps_attempted"] <= config["registered_periods_per_phase"]
        and lower_bound["counters"]["mj_step"]["attempted"] <= config["max_inner_steps_per_phase"]
        and lower_bound["counters"]["model_construct"]["attempted"] <= config["max_models_per_phase"]
        and lower_bound["counters"]["context"]["attempted"] <= config["max_contexts_per_phase"]
        and len(workers) <= config["max_workers_per_phase"],
        "Observed resource lower bounds must fit the registered phase budget",
    )
    print(
        publish_result(
            OUTPUT / phase / "resources.json",
            dict(
                phase=phase,
                group_ref=group_ref,
                observed_resource_lower_bound=lower_bound,
                complete_counts=not any(row["unobserved_after_launch"] for row in rows),
                all_launched_reaped=all(row["actual_wait"] is row["reaped"] is True for row in workers.values()),
                cases=rows,
            ),
        ),
        flush=True,
    )


def phase_run(phase):
    strict_preflight()
    try:
        configured(base.phase_run)(phase)
    finally:
        if (OUTPUT / phase / "result.json").is_file():
            record_resources(phase)


def compare_prior_execution(current, old):
    """Describe the observed common prefix, including differences and stops."""
    now, prior = previous.periods(current), previous.periods(old)
    maxima = {key: 0.0 for key in ("command", "state", "tcp_error", "inner_q", "inner_qvel", "inner_time", "gripper")}
    first_numeric_difference = None
    first_label_difference = None

    def compare(name, left, right):
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        prefix.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "Finite comparable values")
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    for index, (left, right) in enumerate(zip(now, prior)):
        prefix.require(left["step"] == right["step"], "Common-prefix step identity must match")
        diffs = {}
        for name, key in (("command", "command"), ("state", "actual_state"), ("tcp_error", "tcp_error")):
            diffs[name] = compare(name, left[key], right[key])
        for name, key in (("inner_q", "q"), ("inner_qvel", "qvel"), ("inner_time", "time")):
            diffs[name] = compare(name, [r[key] for r in left["actual_inner"]], [r[key] for r in right["actual_inner"]])
        a, b = left["completed_feedback"]["score"], right["completed_feedback"]["score"]
        diffs["gripper"] = max(
            compare("gripper", a[key], b[key])
            for key in ("opening", "opening_reference", "opening_error", "qpos", "qvel", "motor_target")
        )
        for key, value in diffs.items():
            maxima[key] = max(maxima[key], value)
        if first_numeric_difference is None and any(value > 1e-10 for value in diffs.values()):
            first_numeric_difference = index
        labels = (
            "endpoint_receipt",
            "pose_caps_satisfied",
            "target_grasp",
            "task_success",
            "inner_outside_samples",
            "inner_below_buffer_samples",
        )
        if first_label_difference is None and any(left[key] != right[key] for key in labels):
            first_label_difference = index
    same_stop = current["runtime"]["continuation"]["stop"] == old["runtime"]["continuation"]["stop"]
    return dict(
        current_periods=len(now),
        prior_periods=len(prior),
        common_periods=min(len(now), len(prior)),
        max_absolute_difference=maxima,
        first_numeric_difference=first_numeric_difference,
        first_label_difference=first_label_difference,
        same_stop=same_stop,
        same_complete_execution=len(now) == len(prior)
        and same_stop
        and first_numeric_difference is None
        and first_label_difference is None,
    )


def timing_summary(rows, field):
    values = [row["decision"].get(field) for row in rows]
    present = [float(value) for value in values if type(value) in (int, float) and np.isfinite(value) and value >= 0]
    return dict(
        field=field,
        periods=len(rows),
        measured=len(present),
        missing_or_invalid=len(values) - len(present),
        median_seconds=float(np.median(present)) if present else None,
        p95_seconds=float(np.percentile(present, 95)) if present else None,
        maximum_seconds=max(present) if present else None,
        above_50ms=sum(value > 0.05 for value in present),
    )


def audit():
    """Recheck both actual phases and compare each available original execution."""
    inputs = load_inputs()
    groups = [prefix.read(prefix.reference(OUTPUT / phase / "result.json")) for phase in ("producer", "checker")]
    producer, checker = groups
    identity = input_receipt(inputs)
    for phase, group in zip(("producer", "checker"), groups, strict=True):
        registration = prefix.read(group["registration_ref"])
        resources = prefix.read(prefix.reference(OUTPUT / phase / "resources.json"))
        prefix.require(
            group["owner_closed"] is group["sources_unchanged"] is resources["all_launched_reaped"] is True
            and registration["inputs"] == identity
            and registration["source_refs"] == source_refs()
            and resources["group_ref"] == prefix.reference(OUTPUT / phase / "result.json"),
            "Both isolated phase owners and their exact source identities must close",
        )
    baseline = prefix.read(inputs["baseline_ref"])
    prior = prefix.read(inputs["unbounded_ref"])
    cases = []
    all_rows, old_rows = [], []
    for item, old, historical in zip(inputs["cohort"]["cases"], prior["cases"], baseline["cases"], strict=True):
        spec, index = item["spec"], item["spec"]["index"]
        prefix.require(old["case"] == historical["case"] == spec, "Fixed case identity must agree")
        p, c = producer["cases"][str(index)], checker["cases"][str(index)]
        replayed = p["status"] == c["status"] == "closed"
        summary = p.get("counts", {}).get("summary")
        replay_max = trace = prior_comparison = timing = None
        if p["status"] == "closed":
            left, old_branch = prefix.read(p["branch_ref"]), prefix.read(old["producer_ref"])
            pc = cohort.validate_branch(left, "producer", spec["periods"])
            prefix.require(pc["summary"] == summary, "Producer summary must be reproducible")
            trace = summarize_seed_traces(left["runtime"]["segment_receipts"][0]["fast_seed_traces"])
            prefix.require(
                trace == left["seed_trace_summary"] == p["counts"]["seed_trace_summary"], "Seed trace accounting"
            )
            current_rows, prior_rows = previous.periods(left), previous.periods(old_branch)
            prior_comparison = compare_prior_execution(left, old_branch)
            timing = timing_summary(current_rows, "full_policy_select_seconds")
            prefix.require(timing["missing_or_invalid"] == 0, "Each executed action requires measured policy time")
            all_rows.extend(current_rows)
            old_rows.extend(prior_rows)
        if replayed:
            right = prefix.read(c["branch_ref"])
            cc = cohort.validate_branch(right, "checker", spec["periods"])
            prefix.require(
                cc["summary"] == c["counts"]["summary"]
                and left["runtime"]["continuation"]["stop"] == right["runtime"]["continuation"]["stop"],
                "Checker summary and reported stop must agree",
            )
            replay_max = comparison.compare_periods(previous.periods(left), previous.periods(right))
            prefix.require(replay_max == c["counts"]["replay_max_difference"], "Independent replay maxima must match")
        cases.append(
            dict(
                case=spec,
                baseline_qualified=historical["qualified"],
                cached_seed_qualified=old["qualified"],
                producer_status=p["status"],
                checker_status=c["status"],
                producer_ref=p.get("branch_ref"),
                checker_ref=c.get("branch_ref"),
                summary=summary,
                replayed=replayed,
                replay_max_difference=replay_max,
                seed_trace_summary=trace,
                prior_comparison=prior_comparison,
                full_policy_timing=timing,
                qualified=cohort.qualifies(summary, replayed, inputs["config"]),
            )
        )
    qualified = sum(row["qualified"] for row in cases)
    result = dict(
        complete=True,
        passed=True,
        control_qualified=qualified == len(cases) == 18,
        registered_cases=len(cases),
        qualified_cases=qualified,
        cached_seed_qualified_cases=prior["qualified_cases"],
        baseline_qualified_cases=baseline["qualified_cases"],
        replayed_cases=sum(row["replayed"] for row in cases),
        identical_prior_executions=sum(
            row["prior_comparison"] is not None and row["prior_comparison"]["same_complete_execution"] for row in cases
        ),
        current_full_policy_timing=timing_summary(all_rows, "full_policy_select_seconds"),
        current_partial_arm_timing=timing_summary(all_rows, "decision_seconds"),
        prior_partial_arm_timing=timing_summary(old_rows, "decision_seconds"),
        cases=cases,
        producer_ref=prefix.reference(OUTPUT / "producer/result.json"),
        checker_ref=prefix.reference(OUTPUT / "checker/result.json"),
        producer_resources_ref=prefix.reference(OUTPUT / "producer/resources.json"),
        checker_resources_ref=prefix.reference(OUTPUT / "checker/resources.json"),
        cached_seed_audit_ref=inputs["unbounded_ref"],
        physics_steps=0,
        world_model_calls=0,
        gpu_calls=0,
        fresh_cases=0,
    )
    print(publish_result(OUTPUT / "audit.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    prefix.environment()
    if args.dry:
        configured(base.dry_run)()
    elif args.phase:
        phase_run(args.phase)
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        audit()


if __name__ == "__main__":
    main()
