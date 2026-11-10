"""Compare two-step audited feedback at the same task-6 development starts."""

import argparse
import copy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import build_causal_quality_feedback as builder
from scripts.simulation.runtime import prepare_multitask_feasibility_first as multitask
from scripts.simulation.runtime import prepare_task2_earlier_preview as original_binding
from scripts.simulation.runtime import prepare_two_step_viability as static_binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_feasible_start_restoration as feasible
from scripts.simulation.runtime import run_task2_staged_first_feasibility as staged

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_two_step_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_two_step_feedback_20261111.md"
ROOT = REPO / "outputs/task2_two_step_feedback_20261111"
DRY = REPO / "outputs/task2_two_step_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_two_step_feedback_20261111/result.json"
STATIC = {
    "path": str(REPO / "outputs/two_step_viability_20261111/result.json"),
    "sha256": "d9e4a6aabe8e5fb29a0c5e9feeaaf5ac020be5461324dd7e36a22f0db600825f",
}
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/task2_staged_first_feasibility_20261111/producer/result.json"),
    "sha256": "6010b4acbac2f489cf5720318408e8729085697a6e333d77b1d2a05cb0cec3d6",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/task2_staged_first_feasibility_20261111/checker/result.json"),
    "sha256": "427010338d1f617b4cb8d6a479aefd0d314c91b42ba900f5a97428e219d4302d",
}
BRANCHES = ((10, "two_step_viability"), (11, "two_step_viability"))
CHECKER_FORBIDDEN = staged.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.two_step_feedback",
    "s2a.research.joint_control.tracking.two_step_quality_audit",
    "s2a.research.trajectory_search.two_step_viability",
)


def source_refs():
    return (
        staged.source_refs()
        + static_binding.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                CONFIG,
                PLAN,
                REPO / "src/s2a/research/joint_control/preview/two_step_feedback.py",
                REPO / "tests/research/joint_control/preview/test_two_step_feedback.py",
                REPO / "tests/research/native_runtime/test_task2_two_step_feedback.py",
            )
        ]
    )


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-two-step-feedback.v1",
            case_indices=[10, 11],
            branch="two_step_viability",
            seed=43,
            continuation_steps=8,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
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
        "Exact two-step feedback scope and original costs",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs, static_inputs = staged.load_inputs(), static_binding.load_inputs()
    old_staged = copy.deepcopy(staged.input_receipt(inputs))
    old_static = copy.deepcopy(static_binding.input_receipt(static_inputs))
    prefix.require(inputs["asset_refs"] == static_inputs["asset_refs"], "Same frozen runtime assets")
    refs = inputs["source_sha256"]
    multitask.merge_refs(refs, static_inputs["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    static, producer, checker = (prefix.read(ref, refs) for ref in (STATIC, PRIOR_PRODUCER, PRIOR_CHECKER))
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            for item in (static, producer, checker)
        )
        and all(item["control_qualified"] is False for item in (static, producer, checker)),
        "Closed prior static and staged-feedback evidence",
    )
    registration = prefix.read(static["registration_ref"], refs)
    prefix.require(
        registration["inputs"] == old_static and registration["source_refs"] == static_binding.source_refs(),
        "Prior static inputs and sources remain fixed",
    )
    static_cases = {}
    for ref in static["case_refs"]:
        item = prefix.read(ref, refs)
        prefix.require(item["complete"] is item["passed"] is True, "Each prior static case closed")
        static_cases[item["case"]["case_index"]] = item["data"]
    prefix.require(
        sorted(static_cases) == [2, 6, 10, 11]
        and static_cases[10]["model_prefix_witness"] is False
        and static_cases[11]["model_prefix_witness"] is True
        and static_cases[11]["independent_audit"]["status"] == "passed",
        "The preregistered raw and terminal static outcomes are retained",
    )
    for phase, result in (("producer", producer), ("checker", checker)):
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            result["phase"] == phase
            and registration["inputs"] == old_staged
            and registration["source_refs"] == staged.source_refs(),
            "Same prior causal staged-feedback inputs and sources",
        )
        for identity, ref in zip(staged.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity and item["complete"] is item["passed"] is True,
                "Both prior staged-feedback branches closed",
            )
    inputs.update(two_step_config_ref=config_ref, two_step_config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=staged.input_receipt(inputs),
        two_step_config_ref=inputs["two_step_config_ref"],
        static_ref=STATIC,
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered two-step feedback branch")
    return f"case_{index:02d}_{mode}"


def make_two_step_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.two_step_feedback import TwoStepFeedbackSolverAudit

    return builder.build_refiner(env, case, receipt, TwoStepFeedbackSolverAudit)


def replay_action(expected):
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot import two-step candidates"
    )
    return staged.replay_action(expected)


def isolated_worker():
    old = staged.isolated_worker()
    if old.__code__ is not original.worker.__code__ or old.__globals__ is original.worker.__globals__:
        raise ValueError("The original worker must remain unmodified")
    namespace = old.__globals__.copy()
    namespace.update(
        ROOT=ROOT,
        binding=SimpleNamespace(
            load_inputs=load_inputs, input_receipt=input_receipt, policy_case=original_binding.policy_case
        ),
        branch_key=branch_key,
        source_refs=source_refs,
        full=SimpleNamespace(make_full_refiner=make_two_step_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(item["case"]["case_index"], item["branch"]) for item in branches] != list(BRANCHES):
        raise ValueError("Two-step feedback branches must match registration")
    normalized = [{**item, "branch": "feasible_start_restoration"} for item in branches]
    counts = feasible.summarize_phase(normalized, phase, {**config, "max_solver_lp_per_phase": 0})
    if phase == "producer":
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("two_step_feedback_traces", [])
            completed = runtime.get("quality_refiner", {}).get("quality_calls_completed", 0)
            if len(traces) != completed or any(
                trace.get("audit_status") == "pending"
                or trace.get("response_evaluations", 321) > config["quality_response_budget"]
                for trace in traces
            ):
                raise ValueError("Each two-step solve must retain its final audit and response costs")
    return counts


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_task2_two_step_feedback",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="two_step_config",
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
