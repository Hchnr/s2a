"""Recheck one WAM-reference margin trigger from the actual completed state."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import FunctionType

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import bounded_reference_choice as runtime
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_bounded_reference_choice as baseline
from scripts.simulation.runtime import run_refresh_margin_physical as previous
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/deferred_margin_recheck_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/deferred_margin_recheck_20261116.md"
OUTPUT = ROOT / "outputs/deferred_margin_recheck_20261116"
QA = ROOT / "outputs/qa/deferred_margin_recheck_final_20261116/result.json"
DRY = OUTPUT / "dry.json"
MODULE = "scripts.simulation.runtime.run_deferred_margin_recheck"
CANDIDATE_MODULE = "s2a.research.native_runtime.deferred_margin_recheck"


def validate_config(config):
    fixed = dict(
        schema="s2a.deferred-margin-recheck.v1",
        output="outputs/deferred_margin_recheck_20261116",
        case_index=2,
        sampling_seed=43,
        reference_step=16,
        deferred_trigger_step=17,
        allowed_trigger_reason="quality_margin_consumed",
        max_deferred_triggers=1,
        max_control_steps_per_phase=48,
        max_native_steps_per_phase=1200,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
        max_quality_responses_per_period=320,
        max_baseline_responses_per_period=480,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if (
        not isinstance(config, dict)
        or set(config) != set(fixed)
        or any(config[key] != value or type(config[key]) is not type(value) for key, value in fixed.items())
    ):
        raise ValueError("The single case, trigger and physical/model budgets must remain fixed")


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/deferred_margin_recheck.py",
            ROOT / "tests/research/native_runtime/test_deferred_margin_recheck.py",
            ROOT / "tests/research/native_runtime/test_deferred_margin_recheck_runtime.py",
        )
    ]


def prepare():
    data = previous.prepare()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    old = data["baseline_producer"]["runtime"]["continuation"]["tail"]
    trigger = old["stop"]["trigger"]
    prefix.require(
        old["stop"]["reason"] == "quality_refresh_required"
        and old["stop"]["at_step"] == config["deferred_trigger_step"]
        and trigger["replan"] is True
        and trigger["reason"] == config["allowed_trigger_reason"]
        and trigger["current_step"] == config["deferred_trigger_step"]
        and len(baseline.tail_periods(data["baseline_producer"])) == 1,
        "The original independently replayed baseline must stop on the registered trigger",
    )
    data["identity"] = dict(
        **data["identity"], deferred_config_ref=config_ref, deferred_plan_ref=prefix.reference(PLAN)
    )
    data["config"] = config
    return data


def checker_isolation():
    isolation.checker_isolation()
    prefix.require(CANDIDATE_MODULE not in sys.modules, "The checker may not load the recheck diagnostic")


def dry_run():
    prefix.environment()
    original.require_no_simulator_import()
    data = prepare()
    checker_isolation()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                identity=data["identity"],
                source_refs=source_refs(),
                native_models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def comparison_with_gate(gate):
    """Bind an opt-in trigger locally without mutating the recorded runtime."""
    old = runtime.run_comparison
    namespace = old.__globals__.copy()
    namespace["quality_stop"] = gate
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def worker(phase):
    prefix.require(phase in ("producer", "checker"), "One registered physical phase is required")
    print(json.dumps(dict(event="worker_started", pid=os.getpid(), phase=phase)), flush=True)
    directory = OUTPUT / phase / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        data = prepare()
        prefix.require(
            registration["phase"] == phase
            and registration["identity"] == data["identity"]
            and registration["source_refs"] == source_refs(),
            "The registered physical inputs must remain fixed",
        )
        saved = None
        if phase == "checker":
            checker_isolation()
            produced = prefix.read(registration["producer_branch_ref"])
            prefix.require(produced["complete"] is produced["passed"] is True, "A closed producer is required")
            saved = produced["runtime"]["continuation"]
        cv2 = data["inputs"]["base"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime

        inputs, case = data["inputs"], data["inputs"]["case"]
        adapter = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        gate = None
        run_comparison = runtime.run_comparison
        if phase == "producer":
            from s2a.research.native_runtime.deferred_margin_recheck import DeferredMarginRecheck

            gate = DeferredMarginRecheck(target_step=data["config"]["deferred_trigger_step"])
            run_comparison = comparison_with_gate(gate)

        def continuation(env, terminal, observers, receipt, *, progress=None):
            return run_comparison(
                env,
                terminal,
                observers,
                receipt,
                phase=phase,
                mode="new",
                case=case,
                history=data["history"],
                original_captures=data["captures"],
                exchanges=data["exchanges"],
                responses=data["responses"],
                runtime_output=directory / "runtime",
                producer=saved,
                progress=progress,
            )

        def progress(receipt):
            print(json.dumps(baseline.progress_event(phase, "new", receipt)), flush=True)

        actual = adapter.run_feedback(
            case["config"],
            case["record"],
            case["record_xml"],
            directory / "runtime",
            inputs["base"]["asset_refs"],
            registered_seed=data["config"]["sampling_seed"],
            continuation=continuation,
            max_control_steps=data["config"]["max_control_steps_per_phase"],
            progress=progress,
        )
        if phase == "checker":
            checker_isolation()
            prefix.require(actual["newly_selected_commands"] == 0, "The checker may only replay recorded actions")
        else:
            actual["deferred_margin_events"] = gate.events
        return dict(
            complete=True, passed=True, runtime=actual, case=case["case"], cv2_admission=cv2_receipt, fresh_cases=0
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def validate_candidate(branch, data):
    counts = baseline.validate_branch(branch, "producer")
    actual = branch["runtime"]
    tail = baseline.tail_periods(branch)
    original_tail = baseline.tail_periods(data["baseline_producer"])
    events = actual.get("deferred_margin_events")
    prefix.require(
        len(tail) >= 1
        and len(original_tail) == 1
        and isinstance(events, list)
        and len(events) == 1
        and events[0]["current_step"] == data["config"]["deferred_trigger_step"]
        and events[0]["original_stop"]["stop"]["trigger"]["reason"] == "quality_margin_consumed",
        "The matched baseline action and exactly one registered margin deferral are required",
    )
    comparison.compare_periods(original_tail, tail[:1])
    for row in tail[1:]:
        verification = row["decision"]["quality"]["verification"]
        prefix.require(
            verification["passed"] is verification["first_pose_caps_satisfied"] is True,
            "Every additional selected command must retain its original independent quality audit",
        )
    counts.update(
        deferred_events=copy.deepcopy(events),
        post_trigger_periods=len(tail) - 1,
        post_trigger_pose_caps_satisfied=sum(row["pose_caps_satisfied"] is True for row in tail[1:]),
        post_trigger_task_success=sum(row["task_success"] is True for row in tail[1:]),
        post_trigger_grasp=sum(row["target_grasp"] is True for row in tail[1:]),
        stop=copy.deepcopy(actual["continuation"]["tail"]["stop"]),
    )
    return counts


def configured_phase():
    """Reuse the existing owner with isolated experimental bindings."""
    old = previous.phase_run
    namespace = old.__globals__.copy()
    namespace.update(
        OUTPUT=OUTPUT,
        QA=QA,
        DRY=DRY,
        MODULE=MODULE,
        prepare=prepare,
        source_refs=source_refs,
        validate_candidate=validate_candidate,
    )
    function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("producer", "checker"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry:
        dry_run()
    elif args.phase:
        configured_phase()(args.phase)
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
