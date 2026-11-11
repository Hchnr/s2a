"""Compare shared-budget margin-first feedback at matched task-2 starts."""

import argparse
import copy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import build_causal_quality_feedback as builder
from scripts.simulation.runtime import prepare_task2_earlier_preview as original_binding
from scripts.simulation.runtime import prepare_two_step_viability as static_binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_feasible_start_restoration as feasible
from scripts.simulation.runtime import run_task2_margin_two_step_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_margin_guarded_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_margin_guarded_feedback_20261111.md"
ROOT = REPO / "outputs/task2_margin_guarded_feedback_20261111"
DRY = REPO / "outputs/task2_margin_guarded_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_margin_guarded_feedback_20261111/result.json"
STATIC = {
    "path": str(REPO / "outputs/margin_guarded_fallback_20261111/result.json"),
    "sha256": "70dfaf10dd00d6743639ed379f90f660502495bc30ff51a45f646c087ffd2c24",
}
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/task2_margin_two_step_feedback_20261111/producer/result.json"),
    "sha256": "3a94fbc681bbc360fe61b9a3b2d217f0e1d5e94a148d3c5d51505e5efbabe41e",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/task2_margin_two_step_feedback_20261111/checker/result.json"),
    "sha256": "1b5aae47c8d8be3f52ad2211ab2435c82352227d719a334acda8afce3ccf4af2",
}
BRANCHES = ((10, "margin_guarded_fallback"), (11, "margin_guarded_fallback"))
CHECKER_FORBIDDEN = previous.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.margin_guarded_feedback",
    "s2a.research.trajectory_search.margin_guarded_fallback",
    "scripts.tracking.trajectory_search.compare_margin_guarded_fallback",
)


def static_source_refs():
    """Verify static source identities without importing the candidate solver."""
    return previous.static_source_refs() + [
        prefix.reference(REPO / path)
        for path in (
            "scripts/tracking/trajectory_search/compare_margin_guarded_fallback.py",
            "configs/tracking/joint_control/margin_guarded_fallback_20261111.json",
            "plans/v0.0.1/tracking/joint_control/margin_guarded_fallback_20261111.md",
            "src/s2a/research/trajectory_search/margin_guarded_fallback.py",
            "tests/research/trajectory_search/test_margin_guarded_fallback.py",
            "tests/research/native_runtime/test_margin_guarded_fallback.py",
        )
    ]


def source_refs():
    return (
        previous.source_refs()
        + static_source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                CONFIG,
                PLAN,
                REPO / "src/s2a/research/joint_control/preview/margin_guarded_feedback.py",
                REPO / "tests/research/joint_control/preview/test_margin_guarded_feedback.py",
                REPO / "tests/research/native_runtime/test_task2_margin_guarded_feedback.py",
            )
        ]
    )


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-margin-guarded-feedback.v1",
            case_indices=[10, 11],
            branch="margin_guarded_fallback",
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
            hard_pose_prefix=2,
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
            max_audit_endpoint_queries_per_phase=3200,
            max_audit_transition_steps_per_phase=41600,
            max_workers=2,
            worker_timeout_seconds=1800,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact margin-first feedback scope and original shared budgets",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs, static_inputs = previous.load_inputs(), static_binding.load_inputs()
    prior_receipt = copy.deepcopy(previous.input_receipt(inputs))
    static_receipt = copy.deepcopy(static_binding.input_receipt(static_inputs))
    prefix.require(inputs["asset_refs"] == static_inputs["asset_refs"], "The same frozen runtime assets are required")
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    static, producer, checker = (prefix.read(ref, refs) for ref in (STATIC, PRIOR_PRODUCER, PRIOR_CHECKER))
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            and item["control_qualified"] is False
            for item in (static, producer, checker)
        ),
        "Closed shared static and independently replayed margin feedback evidence",
    )
    registration = prefix.read(static["registration_ref"], refs)
    recorded = registration["inputs"]
    original_static = recorded["previous"]["guarded"]["previous"]
    prefix.require(
        registration["source_refs"] == static_source_refs()
        and recorded["shared_config_ref"]
        == prefix.reference(REPO / "configs/tracking/joint_control/margin_guarded_fallback_20261111.json")
        and recorded["prior_ref"] == previous.STATIC
        and all(original_static[key] == value for key, value in static_receipt.items() if key != "source_sha256")
        and all(
            original_static["source_sha256"].get(path) == digest
            for path, digest in static_receipt["source_sha256"].items()
        ),
        "Shared static provenance and common causal inputs are fixed",
    )
    for path, digest in recorded["source_sha256"].items():
        prefix.checked({"path": path, "sha256": digest}, refs)
    cases = {}
    for ref in static["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(item["complete"] is item["passed"] is True and index not in cases, "Unique static case")
        cases[index] = item["data"]
    prefix.require(
        sorted(cases) == [2, 6, 10, 11]
        and all(cases[i]["model_prefix_witness"] is True for i in (10, 11))
        and cases[10]["independent_scope_audit"] == dict(status="passed", scope="guarded_two_step")
        and cases[11]["independent_scope_audit"] == dict(status="passed", scope="margin_two_step"),
        "Both preregistered task2 selection scopes passed independent static audit",
    )
    for phase, result in (("producer", producer), ("checker", checker)):
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            result["phase"] == phase
            and registration["inputs"] == prior_receipt
            and registration["source_refs"] == previous.source_refs(),
            "Prior margin feedback uses the same causal inputs and sources",
        )
        for identity, ref in zip(previous.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity
                and item["complete"] is item["passed"] is True
                and (phase != "checker" or item["candidate_modules_unloaded"] is True),
                "Both prior margin branches and independent replays are closed",
            )
    inputs.update(shared_feedback_config_ref=config_ref, shared_feedback_config=config, previous_receipt=prior_receipt)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=inputs["previous_receipt"],
        shared_feedback_config_ref=inputs["shared_feedback_config_ref"],
        static_ref=STATIC,
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered shared-budget feedback branch")
    return f"case_{index:02d}_{mode}"


def make_shared_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.margin_guarded_feedback import MarginGuardedFeedbackAudit

    return builder.build_refiner(env, case, receipt, MarginGuardedFeedbackAudit)


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
        full=SimpleNamespace(make_full_refiner=make_shared_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(item["case"]["case_index"], item["branch"]) for item in branches] != list(BRANCHES):
        raise ValueError("Shared-budget feedback branches must match registration")
    normalized = [{**item, "branch": "feasible_start_restoration"} for item in branches]
    counts = feasible.summarize_phase(normalized, phase, {**config, "max_solver_lp_per_phase": 0})
    if phase == "producer":
        selected = dict(margin_selected=0, guarded_selected=0, fallback_responses=0)
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("margin_guarded_traces", [])
            prefix.require(
                len(traces) == runtime["quality_refiner"]["quality_calls_completed"],
                "One shared-budget trace per completed decision",
            )
            for trace in traces:
                scope, status = trace["selected_scope"], trace["audit_status"]
                prefix.require(
                    trace["primary_responses"] + trace["fallback_responses"] == trace["response_evaluations"]
                    and trace["response_evaluations"] <= config["quality_response_budget"]
                    and trace["fallback_lp_attempts"] == 0
                    and trace["response_budget"] == config["quality_response_budget"]
                    and trace["feasibility_iteration_budget"] == config["max_feasibility_iterations"]
                    and trace["total_iteration_budget"] == config["max_optimizer_iterations"]
                    and scope in (None, "margin_two_step", "guarded_two_step")
                    and status in ("passed", "rejected", "not_attempted")
                    and (scope is None) == (status == "not_attempted")
                    and (scope != "margin_two_step" or trace["fallback_responses"] == 0)
                    and (scope != "guarded_two_step" or trace["fallback_initializer_reused"] is True),
                    "Shared budgets and independently audited selection must remain explicit",
                )
                selected["margin_selected"] += int(scope == "margin_two_step" and status == "passed")
                selected["guarded_selected"] += int(scope == "guarded_two_step" and status == "passed")
                selected["fallback_responses"] += trace["fallback_responses"]
            prefix.require(
                sum(t["audit_status"] == "passed" for t in traces)
                == runtime["quality_refiner"]["witness_audits_passed"]
                and sum(t["response_evaluations"] for t in traces)
                == runtime["quality_refiner"]["quality_response_evaluations"],
                "Each independently audited selection and model response is charged",
            )
        counts.update(selected)
    return counts


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_task2_margin_guarded_feedback",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="shared_feedback_config",
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
