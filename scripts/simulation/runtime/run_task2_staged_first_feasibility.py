"""Compare staged first-pose feasibility at the same task-6 development starts."""

import argparse
import copy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import build_causal_quality_feedback as builder
from scripts.simulation.runtime import prepare_task2_earlier_preview as original_binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_feasible_start_restoration as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_staged_first_feasibility_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_staged_first_feasibility_20261111.md"
ROOT = REPO / "outputs/task2_staged_first_feasibility_20261111"
DRY = REPO / "outputs/task2_staged_first_feasibility_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_staged_first_feasibility_20261111/result.json"
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/task2_feasible_start_restoration_20261111/producer/result.json"),
    "sha256": "6cacd8be2eb60dd50635613a02a4fc66500b4fe375fa24642382c48e9a52d142",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/task2_feasible_start_restoration_20261111/checker/result.json"),
    "sha256": "b4e4cdd8e86d5dd1c21048792640c0fe7105ff43762827a442e7cdbbb1e42337",
}
BRANCHES = ((10, "staged_first_feasibility"), (11, "staged_first_feasibility"))
CHECKER_FORBIDDEN = previous.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.staged_first_feasibility_feedback",
    "s2a.research.trajectory_search.staged_first_feasibility",
)


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "scripts/simulation/runtime/build_causal_quality_feedback.py",
            REPO / "scripts/simulation/runtime/registered_feedback_phase.py",
            REPO / "src/s2a/research/trajectory_search/staged_first_feasibility.py",
            REPO / "src/s2a/research/joint_control/preview/staged_first_feasibility_feedback.py",
            REPO / "tests/research/trajectory_search/test_staged_first_feasibility.py",
            REPO / "tests/research/joint_control/preview/test_staged_first_feasibility_feedback.py",
            REPO / "tests/research/native_runtime/test_registered_feedback_phase.py",
            REPO / "tests/research/native_runtime/test_task2_staged_first_feasibility.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-staged-first-feasibility.v1",
            case_indices=[10, 11],
            branch="staged_first_feasibility",
            seed=43,
            continuation_steps=8,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            quality_response_budget=320,
            max_optimizer_iterations=120,
            max_first_stage_iterations=40,
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
        "Exact staged feasibility scope and budgets",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs = previous.load_inputs()
    old_receipt = copy.deepcopy(previous.input_receipt(inputs))
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    producer, checker = (prefix.read(ref, refs) for ref in (PRIOR_PRODUCER, PRIOR_CHECKER))
    for phase, result in (("producer", producer), ("checker", checker)):
        prefix.require(
            result["phase"] == phase
            and result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["control_qualified"] is False
            and result["world_model_calls"] == result["fresh_cases"] == 0
            and len(result["branch_refs"]) == 2,
            "Both prior feasible-start branches must be closed",
        )
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            registration["inputs"] == old_receipt and registration["source_refs"] == previous.source_refs(),
            "Same prior causal inputs and source identity",
        )
        for identity, ref in zip(previous.BRANCHES, result["branch_refs"], strict=True):
            branch = prefix.read(ref, refs)
            prefix.require(
                (branch["case"]["case_index"], branch["branch"]) == identity
                and branch["complete"] is branch["passed"] is True,
                "Both prior branches are retained",
            )
    inputs.update(staged_config_ref=config_ref, staged_config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=previous.input_receipt(inputs),
        staged_config_ref=inputs["staged_config_ref"],
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered staged feasibility branch")
    return f"case_{index:02d}_{mode}"


def make_staged_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.staged_first_feasibility_feedback import StagedFirstFeasibilitySolverAudit

    return builder.build_refiner(env, case, receipt, StagedFirstFeasibilitySolverAudit)


def replay_action(expected):
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot import staged or earlier candidates"
    )
    return previous.replay_action(expected)


def isolated_worker():
    old = previous.isolated_worker()
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
        full=SimpleNamespace(make_full_refiner=make_staged_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(x["case"]["case_index"], x["branch"]) for x in branches] != list(BRANCHES):
        raise ValueError("Staged feasibility branches must match registration")
    normalized = [{**item, "branch": "feasible_start_restoration"} for item in branches]
    counts = previous.summarize_phase(normalized, phase, {**config, "max_solver_lp_per_phase": 0})
    if phase == "producer":
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("staged_first_feasibility_traces", [])
            completed = runtime.get("quality_refiner", {}).get("quality_calls_completed", 0)
            if len(traces) != completed or any(
                trace.get("response_evaluations", 321) > config["quality_response_budget"] for trace in traces
            ):
                raise ValueError("Every staged solve must retain its phase and response costs")
    return counts


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_task2_staged_first_feasibility",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="staged_config",
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
