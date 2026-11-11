"""Compare shared-budget dual-first and completed-plan first-pose feedback."""

import argparse
import copy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import build_causal_quality_feedback as builder
from scripts.simulation.runtime import prepare_task2_earlier_preview as original_binding
from scripts.simulation.runtime import prepare_task2_shared_shift as static_binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_causal_shift_fallback as prior_shift
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_margin_guarded_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_shared_shift_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_shared_shift_feedback_20261111.md"
ROOT = REPO / "outputs/task2_shared_shift_feedback_20261111"
DRY = REPO / "outputs/task2_shared_shift_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_shared_shift_feedback_20261111/result.json"
STATIC = {
    "path": str(REPO / "outputs/task2_shared_shift_qualification_20261111/result.json"),
    "sha256": "281b41e4a8170547f00fd3e381ade08a4643f88399d962602cdfe4cc966603fc",
}
PRIOR_PRODUCER = static_binding.PRODUCER
PRIOR_CHECKER = static_binding.CHECKER
BRANCHES = ((10, "shared_shift_fallback"), (11, "shared_shift_fallback"))
CHECKER_FORBIDDEN = previous.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.shared_shift_fallback",
    "scripts.tracking.trajectory_search.audit_task2_shared_shift",
)


def static_source_refs():
    """Verify prior static identities without loading any candidate policy."""
    return (
        static_binding.source_refs()
        + prior_shift.static_source_refs()
        + [
            prefix.reference(REPO / path)
            for path in (
                "scripts/tracking/trajectory_search/audit_task2_shared_shift.py",
                "tests/research/native_runtime/test_task2_shared_shift_audit.py",
            )
        ]
    )


def source_refs():
    return static_source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/joint_control/preview/shared_shift_fallback.py",
            REPO / "tests/research/joint_control/preview/test_shared_shift_fallback.py",
            REPO / "tests/research/native_runtime/test_task2_shared_shift_feedback.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-shared-shift-feedback.v1",
            case_indices=[10, 11],
            branch="shared_shift_fallback",
            seed=43,
            continuation_steps=8,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            model_position_margin_m=0.0005,
            model_rotation_margin_rad=0.003,
            selection_tolerance=1e-10,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            primary_hard_pose_prefix=2,
            shifted_hard_pose_prefix=1,
            quality_response_budget=320,
            max_optimizer_iterations=120,
            max_feasibility_iterations=60,
            max_start_lp_per_call=2,
            max_start_lp_per_phase=32,
            max_control_steps_per_phase=128,
            max_native_steps_per_phase=3200,
            max_models_per_phase=6,
            max_contexts_per_phase=6,
            max_refiner_calls_per_phase=16,
            max_refiner_response_evaluations_per_phase=7680,
            max_quality_calls_per_phase=16,
            max_quality_response_evaluations_per_phase=5120,
            max_shifted_rollouts_per_phase=16,
            max_audit_endpoint_queries_per_phase=3200,
            max_audit_transition_steps_per_phase=41600,
            max_workers=2,
            worker_timeout_seconds=1800,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact shared dual-first and shifted-plan feedback budgets",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs, static_inputs = previous.load_inputs(), static_binding.load_inputs()
    prior_receipt = copy.deepcopy(previous.input_receipt(inputs))
    static_receipt = copy.deepcopy(static_binding.input_receipt(static_inputs))
    prefix.require(
        inputs["asset_refs"] == static_inputs["asset_refs"] and static_receipt["previous"] == prior_receipt,
        "The same prior feedback inputs and frozen runtime assets are required",
    )
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    static, producer, checker = (prefix.read(ref, refs) for ref in (STATIC, PRIOR_PRODUCER, PRIOR_CHECKER))
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            and item["control_qualified"] is False
            for item in (static, producer, checker)
        )
        and static["physics_steps"] == static["world_model_calls"] == static["fresh_cases"] == 0,
        "Closed prior static qualification and independently replayed feedback",
    )
    registration = prefix.read(static["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == static_receipt and registration["source_refs"] == static_source_refs(),
        "Exact branch-local shifted qualification and frozen source identities",
    )
    for path, digest in static_receipt["source_sha256"].items():
        prefix.checked({"path": path, "sha256": digest}, refs)
    cases = {}
    for ref in static["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(item["complete"] is item["passed"] is True and index not in cases, "Unique static stop case")
        cases[index] = item["data"]
    prefix.require(
        sorted(cases) == [10, 11]
        and cases[10]["quality"]["model_first_pose_candidate"] is False
        and cases[11]["quality"]["model_first_pose_candidate"] is True
        and all(cases[index]["quality"]["model_prefix_candidate"] is False for index in (10, 11))
        and all(cases[index]["independent_audit"]["status"] == "passed" for index in (10, 11)),
        "Registered first-pose qualifications and independent safety results",
    )
    for phase, result in (("producer", producer), ("checker", checker)):
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            result["phase"] == phase
            and registration["inputs"] == prior_receipt
            and registration["source_refs"] == previous.source_refs(),
            "Same prior shared-budget feedback inputs and sources",
        )
        for identity, ref in zip(previous.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity
                and item["complete"] is item["passed"] is True
                and (phase != "checker" or item["candidate_modules_unloaded"] is True),
                "Both prior shared-budget branches and independent replays are closed",
            )
    inputs.update(shared_shift_config_ref=config_ref, shared_shift_config=config, prior_receipt=prior_receipt)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=inputs["prior_receipt"],
        shared_shift_config_ref=inputs["shared_shift_config_ref"],
        static_ref=STATIC,
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered shared-shift branch")
    return f"case_{index:02d}_{mode}"


def make_shared_shift_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit

    adapter = SharedShiftFallbackAudit(receipt)
    policy, history = builder.build_refiner(env, case, receipt, lambda _receipt: adapter)
    adapter.attach_history(policy.tracking.history)
    return policy, history


def replay_action(expected):
    prefix.require(all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot import candidates")
    return previous.replay_action(expected)


def isolated_worker():
    old = previous.isolated_worker()
    if old.__code__ is not original.worker.__code__ or old.__globals__ is original.worker.__globals__:
        raise ValueError("The frozen earlier worker must remain unchanged")
    namespace = old.__globals__.copy()
    namespace.update(
        ROOT=ROOT,
        binding=SimpleNamespace(
            load_inputs=load_inputs, input_receipt=input_receipt, policy_case=original_binding.policy_case
        ),
        branch_key=branch_key,
        source_refs=source_refs,
        full=SimpleNamespace(make_full_refiner=make_shared_shift_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(item["case"]["case_index"], item["branch"]) for item in branches] != list(BRANCHES):
        raise ValueError("Shared-shift branches must match registration")
    namespace = original.summarize_phase.__globals__.copy()
    namespace["BRANCHES"] = tuple((index, "full_quality") for index, _ in BRANCHES)
    summarize = FunctionType(
        original.summarize_phase.__code__,
        namespace,
        original.summarize_phase.__name__,
        original.summarize_phase.__defaults__,
        original.summarize_phase.__closure__,
    )
    normalized = [{**item, "branch": "full_quality"} for item in branches]
    counts = summarize(normalized, phase, config)
    if phase == "producer":
        extra = dict(start_lp_calls=0, shifted_rollouts=0, margin_selected=0, guarded_selected=0, first_selected=0)
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("shared_shift_traces", [])
            causal = runtime.get("causal_shift_fallback_traces", [])
            prefix.require(
                len(traces) == len(causal) == runtime["quality_refiner"]["quality_calls_completed"],
                "One shared-shift trace per completed decision",
            )
            for trace, base in zip(traces, causal, strict=True):
                shared, scope, status = trace["shared_search"], trace["selected_scope"], trace["audit_status"]
                prefix.require(
                    trace["primary_response_evaluations"] == shared["primary_responses"] + shared["fallback_responses"]
                    and trace["shifted_rollouts"] in (0, 1)
                    and trace["response_evaluations"]
                    == trace["primary_response_evaluations"] + trace["shifted_rollouts"]
                    and trace["response_evaluations"] <= config["quality_response_budget"]
                    and trace["response_evaluations"] == base["response_evaluations"]
                    and trace["shifted_rollouts"] == base["fallback_rollouts"]
                    and status == base["audit_status"]
                    and shared["fallback_lp_attempts"] == 0
                    and shared["response_budget"] == config["quality_response_budget"]
                    and shared["feasibility_iteration_budget"] == config["max_feasibility_iterations"]
                    and shared["total_iteration_budget"] == config["max_optimizer_iterations"]
                    and scope in (None, "margin_two_step", "guarded_two_step", "shifted_plan_first_pose")
                    and status in ("passed", "rejected", "not_attempted")
                    and (scope is None) == (status == "not_attempted")
                    and (scope != "shifted_plan_first_pose" or trace["shifted_rollouts"] == 1)
                    and (scope not in ("margin_two_step", "guarded_two_step") or trace["shifted_rollouts"] == 0)
                    and (trace["shifted_rollouts"] == 0 or trace["normalized_status"] == "two_step_no_witness"),
                    "Shared budgets, causal fallback and independently audited selection",
                )
                extra["start_lp_calls"] += base["primary_initialization"]["start_lp_attempts"]
                extra["shifted_rollouts"] += trace["shifted_rollouts"]
                extra["margin_selected"] += int(scope == "margin_two_step" and status == "passed")
                extra["guarded_selected"] += int(scope == "guarded_two_step" and status == "passed")
                extra["first_selected"] += int(scope == "shifted_plan_first_pose" and status == "passed")
            prefix.require(
                sum(t["audit_status"] == "passed" for t in traces)
                == runtime["quality_refiner"]["witness_audits_passed"]
                and sum(t["response_evaluations"] for t in traces)
                == runtime["quality_refiner"]["quality_response_evaluations"],
                "Each independently audited selection and response is charged",
            )
        prefix.require(
            extra["start_lp_calls"] <= config["max_start_lp_per_phase"]
            and extra["shifted_rollouts"] <= config["max_shifted_rollouts_per_phase"],
            "Registered LP and shifted-rollout budgets",
        )
        counts.update(extra)
    return counts


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_task2_shared_shift_feedback",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="shared_shift_config",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        inputs = load_inputs()
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=input_receipt(inputs),
                    source_refs=source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "One known phase required")
        isolated_worker()(phase, int(index), mode)


if __name__ == "__main__":
    main()
