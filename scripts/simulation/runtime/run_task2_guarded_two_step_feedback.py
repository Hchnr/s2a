"""Compare guarded two-step feedback at matched task-6 development starts."""

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
from scripts.simulation.runtime import run_task2_two_step_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_guarded_two_step_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_guarded_two_step_feedback_20261111.md"
ROOT = REPO / "outputs/task2_guarded_two_step_feedback_20261111"
DRY = REPO / "outputs/task2_guarded_two_step_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_guarded_two_step_feedback_20261111/result.json"
STATIC = {
    "path": str(REPO / "outputs/guarded_two_step_search_20261111/result.json"),
    "sha256": "0f6ed1926a5091eda36e6bf3794e588aa62d7b3ffadfba82afa1b289750a404b",
}
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/task2_two_step_feedback_20261111/producer/result.json"),
    "sha256": "cfb36bb2c2227749d585f5267914e2a8f8edd1987fb7a90f2b55afba9115e7cf",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/task2_two_step_feedback_20261111/checker/result.json"),
    "sha256": "e714f71e11282a4a718565956b12e33a3c2fe934acb3bb642495a3d5608cd75e",
}
GUARD_CONFIG = REPO / "configs/tracking/joint_control/guarded_two_step_search_20261111.json"
GUARD_PLAN = REPO / "plans/v0.0.1/tracking/joint_control/guarded_two_step_search_20261111.md"
BRANCHES = ((10, "guarded_two_step"), (11, "guarded_two_step"))
CHECKER_FORBIDDEN = previous.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.guarded_two_step_feedback",
    "s2a.research.trajectory_search.guarded_two_step_search",
    "scripts.tracking.trajectory_search.compare_guarded_two_step_search",
)


def static_source_refs():
    """Reconstruct static provenance without importing its candidate code."""
    return static_binding.source_refs() + [
        prefix.reference(path)
        for path in (
            REPO / "scripts/tracking/trajectory_search/compare_guarded_two_step_search.py",
            GUARD_CONFIG,
            GUARD_PLAN,
            REPO / "src/s2a/research/trajectory_search/guarded_two_step_search.py",
            REPO / "tests/research/trajectory_search/test_guarded_two_step_search.py",
            REPO / "tests/research/native_runtime/test_guarded_two_step_search.py",
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
                REPO / "src/s2a/research/joint_control/preview/guarded_two_step_feedback.py",
                REPO / "tests/research/joint_control/preview/test_guarded_two_step_feedback.py",
                REPO / "tests/research/native_runtime/test_task2_guarded_two_step_feedback.py",
            )
        ]
    )


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-guarded-two-step-feedback.v1",
            case_indices=[10, 11],
            branch="guarded_two_step",
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
            normalized_search_guard=1e-4,
            feasibility_objective_scale=1e4,
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
        "Exact guarded two-step feedback scope and original budgets",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs, static_inputs = previous.load_inputs(), static_binding.load_inputs()
    old_feedback = copy.deepcopy(previous.input_receipt(inputs))
    expected_static = copy.deepcopy(static_binding.input_receipt(static_inputs))
    prefix.require(inputs["asset_refs"] == static_inputs["asset_refs"], "The same frozen runtime assets are required")
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    static, producer, checker = (prefix.read(ref, refs) for ref in (STATIC, PRIOR_PRODUCER, PRIOR_CHECKER))
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            for item in (static, producer, checker)
        )
        and all(item["control_qualified"] is False for item in (static, producer, checker)),
        "Closed prior guarded static and two-step feedback evidence",
    )
    registration = prefix.read(static["registration_ref"], refs)
    recorded = registration["inputs"]
    prior_static = recorded["previous"]
    prefix.require(
        registration["source_refs"] == static_source_refs()
        and recorded["guard_config_ref"] == prefix.reference(GUARD_CONFIG)
        and recorded["prior_ref"] == previous.STATIC
        and all(prior_static[key] == value for key, value in expected_static.items() if key != "source_sha256")
        and all(
            prior_static["source_sha256"].get(path) == digest
            for path, digest in expected_static["source_sha256"].items()
        ),
        "Guarded static provenance and common inputs are fixed",
    )
    for path, digest in recorded["source_sha256"].items():
        prefix.checked({"path": path, "sha256": digest}, refs)
    static_cases = {}
    for ref in static["case_refs"]:
        item = prefix.read(ref, refs)
        prefix.require(item["complete"] is item["passed"] is True, "Each guarded static case closed")
        static_cases[item["case"]["case_index"]] = item["data"]
    prefix.require(
        sorted(static_cases) == [2, 6, 10, 11]
        and all(static_cases[index]["model_prefix_witness"] is True for index in (10, 11))
        and all(static_cases[index]["independent_audit"]["status"] == "passed" for index in (10, 11)),
        "Both matched task2 static candidates passed independent audit",
    )
    for phase, result in (("producer", producer), ("checker", checker)):
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            result["phase"] == phase
            and registration["inputs"] == old_feedback
            and registration["source_refs"] == previous.source_refs(),
            "Same prior two-step feedback inputs and sources",
        )
        for identity, ref in zip(previous.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity and item["complete"] is item["passed"] is True,
                "Both prior feedback branches closed",
            )
    inputs.update(guarded_config_ref=config_ref, guarded_config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=previous.input_receipt(inputs),
        guarded_config_ref=inputs["guarded_config_ref"],
        static_ref=STATIC,
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered guarded two-step branch")
    return f"case_{index:02d}_{mode}"


def make_guarded_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.guarded_two_step_feedback import GuardedTwoStepFeedbackSolverAudit

    return builder.build_refiner(env, case, receipt, GuardedTwoStepFeedbackSolverAudit)


def replay_action(expected):
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot import guarded candidates"
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
        full=SimpleNamespace(make_full_refiner=make_guarded_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(item["case"]["case_index"], item["branch"]) for item in branches] != list(BRANCHES):
        raise ValueError("Guarded feedback branches must match registration")
    normalized = [{**item, "branch": "two_step_viability"} for item in branches]
    counts = previous.summarize_phase(normalized, phase, config)
    if phase == "producer":
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("guarded_two_step_traces", [])
            completed = runtime.get("quality_refiner", {}).get("quality_calls_completed", 0)
            if len(traces) != completed or any(
                trace.get("normalized_guard") != config["normalized_search_guard"]
                or trace.get("scale") != config["feasibility_objective_scale"]
                or trace.get("acceptance_caps_changed") is not False
                or trace.get("future_stage_changed") is not False
                or trace.get("response_evaluations", 321) > config["quality_response_budget"]
                for trace in traces
            ):
                raise ValueError("Every guarded solve must retain unchanged acceptance and response costs")
    return counts


def execute(phase):
    registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_task2_guarded_two_step_feedback",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="guarded_config",
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
