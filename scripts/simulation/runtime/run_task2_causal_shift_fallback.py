"""Compare causal shifted-plan fallback at the guarded task-2 starts."""

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
from scripts.simulation.runtime import run_task2_guarded_two_step_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_causal_shift_fallback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_causal_shift_fallback_20261111.md"
ROOT = REPO / "outputs/task2_causal_shift_fallback_20261111"
DRY = REPO / "outputs/task2_causal_shift_fallback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_causal_shift_fallback_20261111/result.json"
STATIC = {
    "path": str(REPO / "outputs/task2_shifted_seed_qualification_20261111/result.json"),
    "sha256": "f5af0996b2efced760615c19dcbe143e26b1a2b18a020ff4fea7731e3fcd1279",
}
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/task2_guarded_two_step_feedback_20261111/producer/result.json"),
    "sha256": "f83d77c74a4230419cb41eb624d8c8fdd07b8f27ca8646e8838121dbaad3a4b7",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/task2_guarded_two_step_feedback_20261111/checker/result.json"),
    "sha256": "2e4ce651b8d7a0d62cf88a1f0097555bbe21c8d65f1b615b42cc919c89718f5c",
}
DRIFT = {
    "path": str(REPO / "outputs/task2_guarded_model_drift_20261111/result.json"),
    "sha256": "4e9bcddb23ce4f7cf7e39afece182e6d312046d515ba142b9be027a0f15ff3ff",
}
BRANCHES = ((10, "causal_shift_fallback"), (11, "causal_shift_fallback"))
CHECKER_FORBIDDEN = previous.CHECKER_FORBIDDEN + ("s2a.research.joint_control.preview.causal_shift_fallback",)


def static_source_refs():
    """Rebuild qualification provenance without importing candidate policies."""
    return previous.source_refs() + [
        prefix.reference(REPO / path)
        for path in (
            "scripts/simulation/runtime/prepare_task2_guarded_model_drift.py",
            "configs/tracking/joint_control/task2_guarded_model_drift_20261111.json",
            "plans/v0.0.1/tracking/joint_control/task2_guarded_model_drift_20261111.md",
            "src/s2a/research/joint_control/tracking/plan_model_drift.py",
            "tests/research/joint_control/tracking/test_plan_model_drift.py",
            "tests/research/native_runtime/test_task2_guarded_model_drift_inputs.py",
            "scripts/tracking/trajectory_search/audit_task2_guarded_model_drift.py",
            "tests/research/native_runtime/test_task2_guarded_model_drift_runtime.py",
            "scripts/tracking/trajectory_search/audit_task2_shifted_seed.py",
            "configs/tracking/joint_control/task2_shifted_seed_qualification_20261111.json",
            "plans/v0.0.1/tracking/joint_control/task2_shifted_seed_qualification_20261111.md",
            "tests/research/native_runtime/test_task2_shifted_seed.py",
        )
    ]


def source_refs():
    return static_source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/joint_control/preview/causal_shift_fallback.py",
            REPO / "tests/research/joint_control/preview/test_causal_shift_fallback.py",
            REPO / "tests/research/native_runtime/test_task2_causal_shift_fallback.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-causal-shift-fallback.v1",
            case_indices=[10, 11],
            branch="causal_shift_fallback",
            seed=43,
            continuation_steps=8,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            primary_hard_pose_prefix=2,
            fallback_hard_pose_prefix=1,
            quality_response_budget=320,
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
            max_fallback_rollouts_per_phase=16,
            max_audit_endpoint_queries_per_phase=3200,
            max_audit_transition_steps_per_phase=41600,
            max_workers=2,
            worker_timeout_seconds=1800,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact primary/fallback feedback scope and original per-decision budget",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs = previous.load_inputs()
    prior_receipt = copy.deepcopy(previous.input_receipt(inputs))
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    static, producer, checker = (prefix.read(ref, refs) for ref in (STATIC, PRIOR_PRODUCER, PRIOR_CHECKER))
    prefix.require(
        all(
            item["complete"] is item["passed"] is item["sources_unchanged"] is True
            for item in (static, producer, checker)
        )
        and all(item["control_qualified"] is False for item in (static, producer, checker)),
        "Closed prior static qualification and independently replayed guarded feedback",
    )
    registration = prefix.read(static["registration_ref"], refs)
    recorded = registration["inputs"]
    prefix.require(
        registration["source_refs"] == static_source_refs()
        and recorded["previous"]["previous"] == prior_receipt
        and recorded["previous"]["producer_ref"] == PRIOR_PRODUCER
        and recorded["previous"]["checker_ref"] == PRIOR_CHECKER
        and recorded["prior_ref"] == DRIFT
        and recorded["asset_refs"] == inputs["asset_refs"],
        "Static qualification is bound to the exact guarded-feedback inputs",
    )
    for path, digest in recorded["source_sha256"].items():
        prefix.checked({"path": path, "sha256": digest}, refs)
    static_cases = {}
    for ref in static["case_refs"]:
        item = prefix.read(ref, refs)
        index = item["case"]["case_index"]
        prefix.require(
            item["complete"] is item["passed"] is True and index not in static_cases, "Unique closed static case"
        )
        static_cases[index] = item["data"]
    prefix.require(
        sorted(static_cases) == [10, 11]
        and all(static_cases[i]["independent_audit"]["status"] == "passed" for i in (10, 11))
        and all(static_cases[i]["linear_safety"]["accepted"] is True for i in (10, 11))
        and all(static_cases[i]["quality"]["model_prefix_candidate"] is False for i in (10, 11))
        and static_cases[10]["quality"]["point_pose_caps_satisfied"][0] is False
        and static_cases[11]["quality"]["point_pose_caps_satisfied"][0] is True,
        "Prior raw and terminal first-pose static findings",
    )
    for phase, result in (("producer", producer), ("checker", checker)):
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            result["phase"] == phase
            and registration["inputs"] == prior_receipt
            and registration["source_refs"] == previous.source_refs(),
            "Same guarded causal feedback inputs and frozen sources",
        )
        for identity, ref in zip(previous.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity
                and item["complete"] is item["passed"] is True
                and (phase != "checker" or item["candidate_modules_unloaded"] is True),
                "Both earlier guarded branches closed and replay checker remained isolated",
            )
    inputs.update(fallback_config_ref=config_ref, fallback_config=config, fallback_previous_receipt=prior_receipt)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=inputs["fallback_previous_receipt"],
        fallback_config_ref=inputs["fallback_config_ref"],
        static_ref=STATIC,
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered causal fallback branch")
    return f"case_{index:02d}_{mode}"


def make_fallback_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.causal_shift_fallback import CausalShiftFallbackAudit

    adapter = CausalShiftFallbackAudit(receipt)
    policy, history = builder.build_refiner(env, case, receipt, lambda _receipt: adapter)
    adapter.attach_history(policy.tracking.history)
    return policy, history


def replay_action(expected):
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker cannot import candidate policies"
    )
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
        full=SimpleNamespace(make_full_refiner=make_fallback_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(item["case"]["case_index"], item["branch"]) for item in branches] != list(BRANCHES):
        raise ValueError("Fallback feedback branches must match registration")
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
        extra = dict(start_lp_calls=0, fallback_rollouts=0, primary_selected=0, fallback_selected=0)
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("causal_shift_fallback_traces", [])
            prefix.require(
                len(traces) == runtime["quality_refiner"]["quality_calls_completed"],
                "One fallback trace per completed primary search",
            )
            for trace in traces:
                primary, fallback_count = trace["primary_response_evaluations"], trace["fallback_rollouts"]
                scope = trace["selected_scope"]
                prefix.require(
                    fallback_count in (0, 1)
                    and trace["response_evaluations"] == primary + fallback_count <= 320
                    and 0 <= trace["primary_initialization"]["start_lp_attempts"] <= 2
                    and scope in (None, "two_step", "first_only")
                    and (scope != "two_step" or fallback_count == 0)
                    and (
                        scope != "first_only"
                        or (fallback_count == 1 and trace["primary_status"] == "two_step_no_witness")
                    )
                    and (scope is not None or trace["audit_status"] == "not_attempted"),
                    "Original response and fallback dispatch accounting",
                )
                extra["start_lp_calls"] += trace["primary_initialization"]["start_lp_attempts"]
                extra["fallback_rollouts"] += fallback_count
                extra["primary_selected"] += int(scope == "two_step" and trace["audit_status"] == "passed")
                extra["fallback_selected"] += int(scope == "first_only" and trace["audit_status"] == "passed")
            prefix.require(
                sum(t["audit_status"] == "passed" for t in traces)
                == runtime["quality_refiner"]["witness_audits_passed"]
                and sum(t["response_evaluations"] for t in traces)
                == runtime["quality_refiner"]["quality_response_evaluations"],
                "Each independently audited selection and actual response is charged",
            )
        prefix.require(
            extra["start_lp_calls"] <= config["max_start_lp_per_phase"]
            and extra["fallback_rollouts"] <= config["max_fallback_rollouts_per_phase"],
            "Registered per-phase LP and fallback budgets",
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
        worker_module="scripts.simulation.runtime.run_task2_causal_shift_fallback",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="fallback_config",
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
