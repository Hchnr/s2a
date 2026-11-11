"""Compare independently audited model-margin feedback at the same task-2 starts."""

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
from scripts.simulation.runtime import run_task2_causal_shift_fallback as fallback
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_guarded_two_step_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_margin_two_step_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_margin_two_step_feedback_20261111.md"
ROOT = REPO / "outputs/task2_margin_two_step_feedback_20261111"
DRY = REPO / "outputs/task2_margin_two_step_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_margin_two_step_feedback_20261111/result.json"
STATIC = {
    "path": str(REPO / "outputs/margin_two_step_search_20261111/result.json"),
    "sha256": "117800ab09af1c44bc44670a30e730f8b45e53331a6c3652d483d93dc7bf6bba",
}
PRIOR_PRODUCER = fallback.PRIOR_PRODUCER
PRIOR_CHECKER = fallback.PRIOR_CHECKER
FALLBACK_PRODUCER = {
    "path": str(REPO / "outputs/task2_causal_shift_fallback_20261111/producer/result.json"),
    "sha256": "54e42a23f0d150d11ee3a896df8e28a04a7786edd538784dd91b8a76d74f4ad2",
}
FALLBACK_CHECKER = {
    "path": str(REPO / "outputs/task2_causal_shift_fallback_20261111/checker/result.json"),
    "sha256": "4848e256de842e96cb66c9d07991a5da223e6015b1a07a0d1bd73b4e1b8c66c7",
}
BRANCHES = ((10, "margin_two_step"), (11, "margin_two_step"))
CHECKER_FORBIDDEN = fallback.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.margin_two_step_feedback",
    "s2a.research.trajectory_search.margin_two_step_search",
    "scripts.tracking.trajectory_search.compare_margin_two_step_search",
)


def static_source_refs():
    """Reconstruct static provenance without importing the candidate solver."""
    return previous.static_source_refs() + [
        prefix.reference(REPO / path)
        for path in (
            "scripts/tracking/trajectory_search/compare_margin_two_step_search.py",
            "configs/tracking/joint_control/margin_two_step_search_20261111.json",
            "plans/v0.0.1/tracking/joint_control/margin_two_step_search_20261111.md",
            "src/s2a/research/trajectory_search/margin_two_step_search.py",
            "tests/research/trajectory_search/test_margin_two_step_search.py",
            "tests/research/native_runtime/test_margin_two_step_search.py",
        )
    ]


def source_refs():
    return (
        fallback.source_refs()
        + static_source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                CONFIG,
                PLAN,
                REPO / "src/s2a/research/joint_control/preview/margin_two_step_feedback.py",
                REPO / "tests/research/joint_control/preview/test_margin_two_step_feedback.py",
                REPO / "tests/research/native_runtime/test_task2_margin_two_step_feedback.py",
            )
        ]
    )


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-margin-two-step-feedback.v1",
            case_indices=[10, 11],
            branch="margin_two_step",
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
            feasibility_objective_scale=10000.0,
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
        "Exact two-point model margins and original feedback budgets",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs, static_inputs = previous.load_inputs(), static_binding.load_inputs()
    feedback_receipt = copy.deepcopy(previous.input_receipt(inputs))
    static_receipt = copy.deepcopy(static_binding.input_receipt(static_inputs))
    prefix.require(inputs["asset_refs"] == static_inputs["asset_refs"], "The same frozen runtime assets are required")
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    static, producer, checker, fallback_producer, fallback_checker = (
        prefix.read(ref, refs) for ref in (STATIC, PRIOR_PRODUCER, PRIOR_CHECKER, FALLBACK_PRODUCER, FALLBACK_CHECKER)
    )
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            and item["control_qualified"] is False
            for item in (static, producer, checker, fallback_producer, fallback_checker)
        ),
        "Closed static and independently replayed prior feedback evidence",
    )
    registration = prefix.read(static["registration_ref"], refs)
    recorded = registration["inputs"]
    prior_static = recorded["guarded"]["previous"]
    prefix.require(
        registration["source_refs"] == static_source_refs()
        and recorded["margin_config_ref"]
        == prefix.reference(REPO / "configs/tracking/joint_control/margin_two_step_search_20261111.json")
        and recorded["prior_ref"] == previous.STATIC
        and all(prior_static[key] == value for key, value in static_receipt.items() if key != "source_sha256")
        and all(
            prior_static["source_sha256"].get(path) == digest
            for path, digest in static_receipt["source_sha256"].items()
        ),
        "Margin static provenance and common causal inputs are fixed",
    )
    for path, digest in recorded["source_sha256"].items():
        prefix.checked({"path": path, "sha256": digest}, refs)
    static_cases = {}
    for ref in static["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(item["complete"] is item["passed"] is True and index not in static_cases, "Unique static case")
        static_cases[index] = item["data"]
    prefix.require(
        sorted(static_cases) == [2, 6, 10, 11]
        and static_cases[10]["model_prefix_witness"] is False
        and static_cases[11]["model_prefix_witness"] is True
        and static_cases[11]["independent_margin_audit"]["status"] == "passed",
        "The preregistered raw and terminal margin outcomes are retained",
    )
    for phase, result in (("producer", producer), ("checker", checker)):
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            result["phase"] == phase
            and registration["inputs"] == feedback_receipt
            and registration["source_refs"] == previous.source_refs(),
            "Original guarded feedback is bound to these inputs and sources",
        )
        for identity, ref in zip(previous.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity
                and item["complete"] is item["passed"] is True
                and (phase != "checker" or item["candidate_modules_unloaded"] is True),
                "Both guarded branches and independent replays are closed",
            )
    for phase, result in (("producer", fallback_producer), ("checker", fallback_checker)):
        registration = prefix.read(result["registration_ref"], refs)
        recorded_fallback = registration["inputs"]
        prefix.require(
            result["phase"] == phase
            and recorded_fallback["previous"] == feedback_receipt
            and recorded_fallback["prior_producer_ref"] == PRIOR_PRODUCER
            and recorded_fallback["prior_checker_ref"] == PRIOR_CHECKER
            and registration["source_refs"] == fallback.source_refs(),
            "Shifted-plan fallback uses the same guarded-feedback inputs",
        )
        for identity, ref in zip(fallback.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity
                and item["complete"] is item["passed"] is True
                and (phase != "checker" or item["candidate_modules_unloaded"] is True),
                "Both shifted-plan branches and independent replays are closed",
            )
    inputs.update(margin_feedback_config_ref=config_ref, margin_feedback_config=config, prior_receipt=feedback_receipt)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=inputs["prior_receipt"],
        margin_feedback_config_ref=inputs["margin_feedback_config_ref"],
        static_ref=STATIC,
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        fallback_producer_ref=FALLBACK_PRODUCER,
        fallback_checker_ref=FALLBACK_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered margin feedback branch")
    return f"case_{index:02d}_{mode}"


def make_margin_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.margin_two_step_feedback import MarginTwoStepFeedbackSolverAudit

    return builder.build_refiner(env, case, receipt, MarginTwoStepFeedbackSolverAudit)


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
        full=SimpleNamespace(make_full_refiner=make_margin_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(item["case"]["case_index"], item["branch"]) for item in branches] != list(BRANCHES):
        raise ValueError("Margin feedback branches must match registration")
    normalized = [{**item, "branch": "two_step_viability"} for item in branches]
    counts = previous.previous.summarize_phase(normalized, phase, config)
    if phase == "producer":
        total_margin_candidates = 0
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("margin_two_step_traces", [])
            prefix.require(
                len(traces) == runtime["quality_refiner"]["quality_calls_completed"],
                "One margin trace per completed solve",
            )
            for trace in traces:
                prefix.require(
                    trace.get("model_position_margin_m") == config["model_position_margin_m"]
                    and trace.get("model_rotation_margin_rad") == config["model_rotation_margin_rad"]
                    and trace.get("selection_tolerance") == config["selection_tolerance"]
                    and trace.get("objective_scale") == config["feasibility_objective_scale"]
                    and trace.get("original_acceptance_caps_changed") is False
                    and trace.get("extra_model_responses") == 0
                    and trace.get("response_evaluations", 321) <= config["quality_response_budget"]
                    and trace.get("audit_status") in ("passed", "rejected", "not_attempted")
                    and (trace["margin_prefix_evaluations"] > 0) == (trace["audit_status"] != "not_attempted"),
                    "Fixed margins, independently audited selection and response accounting",
                )
                total_margin_candidates += int(trace["audit_status"] == "passed")
            prefix.require(
                sum(t["audit_status"] == "passed" for t in traces)
                == runtime["quality_refiner"]["witness_audits_passed"]
                and sum(t["response_evaluations"] for t in traces)
                == runtime["quality_refiner"]["quality_response_evaluations"],
                "Every selected margin candidate and response is charged",
            )
        counts["margin_candidates_audited"] = total_margin_candidates
    return counts


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_task2_margin_two_step_feedback",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="margin_feedback_config",
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
