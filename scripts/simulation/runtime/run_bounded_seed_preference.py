"""Run the fixed 18-case correct-reference comparison with bounded seed preference."""

import argparse
import json
import sys
from pathlib import Path
from types import FunctionType

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_cached_seed_preference as base
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_correct_reference_cohort as cohort
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/bounded_seed_preference_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/bounded_seed_preference_20261116.md"
OUTPUT = ROOT / "outputs/bounded_seed_preference_20261116"
DRY = ROOT / "outputs/bounded_seed_preference_dry_20261116.json"
QA = ROOT / "outputs/qa/bounded_seed_preference_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_bounded_seed_preference"
CANDIDATE_MODULE = "s2a.research.joint_control.preview.bounded_seed_preference"
MAX_INCREASE = 0.3
TOLERANCE = 1e-10
UNBOUNDED_AUDIT = dict(
    path="outputs/cached_seed_preference_20261116/audit.json",
    sha256="ae8358fc25a9675d208f91cd8e7c52f4d20f120e9c314ee18755b97472744525",
)


def validate_config(config):
    """Require the original controls and budgets plus this one fixed variant."""
    expected = prefix.read(prefix.reference(base.CONFIG)).copy()
    expected.update(
        schema="s2a.bounded-seed-preference.v1",
        output="outputs/bounded_seed_preference_20261116",
        unbounded_preference_audit=UNBOUNDED_AUDIT,
        selection="prefer_exact_seed_if_cached_eligible_and_future_slack_increase_bounded",
        max_future_slack_increase=MAX_INCREASE,
    )
    if (
        not isinstance(config, dict)
        or set(config) != set(expected)
        or any(config[key] != value or type(config[key]) is not type(value) for key, value in expected.items())
    ):
        raise ValueError("The fixed cohort, safety gates, budgets and slack bound must match registration")


def source_refs():
    return base.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/joint_control/preview/bounded_seed_preference.py",
            ROOT / "tests/research/joint_control/preview/test_bounded_seed_preference.py",
            ROOT / "tests/research/native_runtime/test_bounded_seed_preference_runtime.py",
        )
    ]


def load_inputs():
    inputs = base.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    unbounded_ref = preparation.configured_ref(config["unbounded_preference_audit"])
    unbounded = prefix.read(unbounded_ref)
    cases = inputs["cohort"]["cases"]
    prefix.require(
        unbounded["complete"] is unbounded["passed"] is True
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
    inputs.update(config=config, config_ref=config_ref, unbounded_ref=unbounded_ref)
    return inputs


def input_receipt(inputs):
    return dict(**base.input_receipt(inputs), unbounded_ref=inputs["unbounded_ref"])


def checker_isolation():
    base.checker_isolation()
    prefix.require(CANDIDATE_MODULE not in sys.modules, "The checker cannot load the bounded candidate")


def summarize_seed_traces(traces):
    """Validate actual decisions and retain the rejected seed's quality cost."""
    if not isinstance(traces, list):
        raise ValueError("A list of actual seed-selection traces is required")
    actual, rejected = [], []
    for row in traces:
        eligible, selected, changed, blocked = (
            row.get(key)
            for key in ("seed_in_eligible_cache", "selected_seed", "selected_changed", "seed_blocked_by_future_slack")
        )
        delta, requested = row.get("future_slack_increase"), row.get("unbounded_seed_future_slack_increase")
        if (
            any(type(value) is not bool for value in (eligible, selected, changed, blocked))
            or row.get("max_future_slack_increase") != MAX_INCREASE
            or type(row.get("extra_model_responses")) is not int
            or row["extra_model_responses"] != 0
            or selected != (eligible and not blocked)
            or (changed and not selected)
            or (eligible != (requested is not None))
        ):
            raise ValueError("Seed eligibility, bounded selection and response accounting must agree")
        if requested is not None and (not np.isfinite(requested) or requested < -TOLERANCE):
            raise ValueError("The unbounded seed quality increment must be finite and nonnegative")
        if blocked != (requested is not None and requested > MAX_INCREASE + TOLERANCE):
            raise ValueError("Only a seed exceeding the fixed quality increment may be blocked")
        if delta is not None:
            if not np.isfinite(delta) or not -TOLERANCE <= delta <= MAX_INCREASE + TOLERANCE:
                raise ValueError("The selected plan must respect the fixed relative slack bound")
            if selected and abs(delta - requested) > TOLERANCE:
                raise ValueError("An accepted seed must retain its original quality increment")
            if blocked and abs(delta) > TOLERANCE:
                raise ValueError("A blocked seed must retain the original ranking")
            actual.append(float(delta))
        elif eligible or blocked:
            raise ValueError("An evaluated eligible seed needs a selected quality increment")
        if blocked:
            rejected.append(float(requested))
    return dict(
        searches=len(traces),
        seed_in_eligible_cache=sum(row["seed_in_eligible_cache"] for row in traces),
        selected_changed=sum(row["selected_changed"] for row in traces),
        blocked_by_future_slack=sum(row["seed_blocked_by_future_slack"] for row in traces),
        maximum_future_slack_increase=max(actual) if actual else None,
        maximum_blocked_seed_increase=max(rejected) if rejected else None,
        extra_model_responses=sum(row["extra_model_responses"] for row in traces),
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
            from s2a.research.joint_control.preview.bounded_seed_preference import BoundedSeedPreferredAudit
            from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
            from s2a.research.joint_control.preview.segmented_reset_policy import build_segmented_reset_policy

            def factory(env, terminal, receipt):
                return build_segmented_reset_policy(
                    env, policy_inputs(case, terminal), receipt, adapter_factory=BoundedSeedPreferredAudit
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


def compare_prior():
    """Preserve the full case denominator when comparing both variants."""
    audit_ref = prefix.reference(OUTPUT / "audit.json")
    audit = prefix.read(audit_ref)
    prior_ref = preparation.configured_ref(UNBOUNDED_AUDIT)
    prior = prefix.read(prior_ref)
    prefix.require(
        audit["complete"] is audit["passed"] is prior["complete"] is prior["passed"] is True
        and len(audit["cases"]) == len(prior["cases"]) == 18,
        "Both full cohort audits are required",
    )
    rows = []
    for current, old in zip(audit["cases"], prior["cases"], strict=True):
        prefix.require(current["case"] == old["case"], "The two cases must use the same original identity")
        rows.append(
            dict(
                case=current["case"],
                bounded_qualified=current["qualified"],
                unbounded_qualified=old["qualified"],
                baseline_qualified=current["baseline_qualified"],
                replayed=current["replayed"],
            )
        )
    result = dict(
        complete=True,
        passed=True,
        registered_cases=18,
        bounded_qualified=sum(row["bounded_qualified"] for row in rows),
        unbounded_qualified=sum(row["unbounded_qualified"] for row in rows),
        baseline_qualified=sum(row["baseline_qualified"] for row in rows),
        improved=[row["case"]["index"] for row in rows if row["bounded_qualified"] and not row["unbounded_qualified"]],
        regressed=[row["case"]["index"] for row in rows if row["unbounded_qualified"] and not row["bounded_qualified"]],
        cases=rows,
        audit_ref=audit_ref,
        unbounded_audit_ref=prior_ref,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    print(publish_result(OUTPUT / "comparison.json", result), flush=True)


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
        configured(base.phase_run)(args.phase)
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        configured(base.audit)()
        compare_prior()


if __name__ == "__main__":
    main()
