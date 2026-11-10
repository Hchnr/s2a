"""Run restoration on the same earlier task-6 states and replay its actions."""

import argparse
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_task2_earlier_preview as original_binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_replan_quality_feedback as full
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_earlier_restoration_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_earlier_restoration_feedback_20261111.md"
ROOT = REPO / "outputs/task2_earlier_restoration_feedback_20261111"
DRY = REPO / "outputs/task2_earlier_restoration_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_earlier_restoration_feedback_20261111/result.json"
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/task2_earlier_preview_feedback_20261111/producer/result.json"),
    "sha256": "b48288924dc24d2f6dee61b4a27d8491dced865dac1c311c07cb4336d9440206",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/task2_earlier_preview_feedback_20261111/checker/result.json"),
    "sha256": "f61a9360e7d16f5599b1be49dbe5f068fed74e87f1b3a71c31893dac766e8f38",
}
BRANCHES = ((10, "restoration"), (11, "restoration"))
CHECKER_FORBIDDEN = original.CHECKER_FORBIDDEN + (full.RESTORE_MODULE,)
_ORIGINAL_WORKER = original.worker


def source_refs():
    return (
        original.source_refs()
        + full.source_refs(restore=True)
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                CONFIG,
                PLAN,
                REPO / "tests/research/native_runtime/test_task2_earlier_restoration_feedback.py",
            )
        ]
    )


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-earlier-restoration-feedback.v1",
            case_indices=[10, 11],
            branch="restoration",
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
        "Exact earlier restoration scope and cost bounds",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs = original_binding.load_inputs()
    old_receipt = original_binding.input_receipt(inputs)
    old_receipt["source_sha256"] = dict(old_receipt["source_sha256"])
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    producer, checker = (prefix.read(ref, refs) for ref in (PRIOR_PRODUCER, PRIOR_CHECKER))
    for phase, result in (("producer", producer), ("checker", checker)):
        prefix.require(
            result["phase"] == phase
            and result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["control_qualified"] is False
            and result["world_model_calls"] == result["fresh_cases"] == 0
            and len(result["branch_refs"]) == 6,
            "All original comparison branches must be closed",
        )
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            registration["inputs"] == old_receipt and registration["source_refs"] == original.source_refs(),
            "Same earlier causal inputs and original comparison sources",
        )
        for expected, ref in zip(original.BRANCHES, result["branch_refs"], strict=True):
            branch = prefix.read(ref, refs)
            prefix.require(
                (branch["case"]["case_index"], branch["branch"]) == expected
                and branch["complete"] is branch["passed"] is True,
                "Every original branch is retained in its registered order",
            )
    inputs.update(restoration_config_ref=config_ref, restoration_config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        original=original_binding.input_receipt(inputs),
        restoration_config_ref=inputs["restoration_config_ref"],
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered restoration branch")
    return f"case_{index:02d}_{mode}"


def make_restoration_refiner(env, case, receipt, *, factory=full.make_full_refiner):
    return factory(env, case, receipt, restore=True)


def replay_action(expected):
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN),
        "Checker cannot import restoration or other candidates",
    )
    return original.replay_action(expected)


def isolated_worker():
    """Reuse the previous worker without changing its module globals or code."""
    if original.worker is not _ORIGINAL_WORKER or _ORIGINAL_WORKER.__globals__ is not vars(original):
        raise ValueError("The original comparison worker must remain unchanged")
    namespace = _ORIGINAL_WORKER.__globals__.copy()
    namespace.update(
        ROOT=ROOT,
        binding=SimpleNamespace(
            load_inputs=load_inputs, input_receipt=input_receipt, policy_case=original_binding.policy_case
        ),
        branch_key=branch_key,
        source_refs=source_refs,
        full=SimpleNamespace(make_full_refiner=make_restoration_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(
        _ORIGINAL_WORKER.__code__,
        namespace,
        _ORIGINAL_WORKER.__name__,
        _ORIGINAL_WORKER.__defaults__,
        _ORIGINAL_WORKER.__closure__,
    )
    worker.__kwdefaults__ = dict(_ORIGINAL_WORKER.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if phase not in ("producer", "checker") or len(branches) != 2:
        raise ValueError("Exactly two restoration branches in one phase required")
    if [(x["case"]["case_index"], x["branch"]) for x in branches] != list(BRANCHES):
        raise ValueError("Restoration branch identities must match registration")
    if not all(x["complete"] is x["passed"] is True and x["runtime"]["cleanup_complete"] is True for x in branches):
        raise ValueError("Every restoration runtime and cleanup must close")
    runs = [x["runtime"] for x in branches]
    control = sum(x["control_steps_completed"] for x in runs)
    native = sum(x["counters"]["mj_step"]["completed"] for x in runs)
    models = sum(x["counters"]["model_construct"]["completed"] for x in runs)
    contexts = sum(x["counters"]["context"]["completed"] for x in runs)
    selected = sum(x["newly_selected_commands"] for x in runs)
    if (
        control > config["max_control_steps_per_phase"]
        or native != 25 * control
        or native > config["max_native_steps_per_phase"]
        or models != config["max_models_per_phase"]
        or contexts != config["max_contexts_per_phase"]
        or (phase == "checker" and selected != 0)
    ):
        raise ValueError("Restoration physical budget or replay-only checker exceeded")
    counts = dict(
        control_steps=control,
        native_steps=native,
        model_initializations=models,
        contexts=contexts,
        newly_selected_commands=selected,
    )
    if phase == "producer":
        values = dict(
            refiner_calls=sum(x.get("refiner_calls", {}).get("attempted", 0) for x in runs),
            refiner_response_evaluations=sum(
                x.get("quality_refiner", {}).get("baseline_response_evaluations", 0) for x in runs
            ),
            quality_calls=sum(x.get("quality_refiner", {}).get("quality_calls_attempted", 0) for x in runs),
            quality_response_evaluations=sum(
                x.get("quality_refiner", {}).get("quality_response_evaluations", 0) for x in runs
            ),
            audit_endpoint_queries=sum(x.get("quality_refiner", {}).get("endpoint_queries_charged", 0) for x in runs),
            audit_transition_steps=sum(x.get("quality_refiner", {}).get("transition_steps_charged", 0) for x in runs),
        )
        budgets = {
            "refiner_calls": "max_refiner_calls_per_phase",
            "refiner_response_evaluations": "max_refiner_response_evaluations_per_phase",
            "quality_calls": "max_quality_calls_per_phase",
            "quality_response_evaluations": "max_quality_response_evaluations_per_phase",
            "audit_endpoint_queries": "max_audit_endpoint_queries_per_phase",
            "audit_transition_steps": "max_audit_transition_steps_per_phase",
        }
        if any(value > config[budgets[key]] for key, value in values.items()):
            raise ValueError("Restoration solver or independent audit budget exceeded")
        counts.update(values)
    return counts


def execute(phase):
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU QA",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry identity",
    )
    producer_refs = None
    if phase == "checker":
        producer = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 2,
            "Both restoration branches closed",
        )
        producer_refs = {
            branch_key(*branch): ref for branch, ref in zip(BRANCHES, producer["branch_refs"], strict=True)
        }
    phase_dir = ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    registration = publish(
        phase_dir / "registration.json",
        dict(phase=phase, inputs=input_receipt(inputs), source_refs=sources, producer_branch_refs=producer_refs),
    )
    specs = [
        dict(
            key=branch_key(index, mode),
            argv=[
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_task2_earlier_restoration_feedback",
                "--worker",
                phase,
                str(index),
                mode,
            ],
            log=phase_dir / branch_key(index, mode) / "worker.log",
        )
        for index, mode in BRANCHES
    ]
    lifecycle = run_owned_workers(
        specs, cwd=REPO, timeout_seconds=inputs["restoration_config"]["worker_timeout_seconds"]
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(phase_dir / "workers.json", lifecycle)
    available = [
        prefix.reference(phase_dir / branch_key(*branch) / "result.json")
        for branch in BRANCHES
        if (phase_dir / branch_key(*branch) / "result.json").is_file()
    ]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        registration_ref=registration,
        workers_ref=workers,
        available_branch_refs=available,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 2
            and all(
                x["returncode"] == 0 and x["actual_wait"] and x["reaped"] and not x["signal_sent"]
                for x in lifecycle["workers"]
            ),
            "Both restoration workers exited and were reaped",
        )
        branch_refs = [prefix.reference(phase_dir / branch_key(*branch) / "result.json") for branch in BRANCHES]
        branches = [prefix.read(ref) for ref in branch_refs]
        counts = summarize_phase(branches, phase, inputs["restoration_config"])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Restoration sources unchanged"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_refs=branch_refs, counts=counts)
    finally:
        print(publish(phase_dir / "result.json", result), flush=True)


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
