"""Compare first-hard restoration on the same task-6 development prefixes."""

import argparse
import copy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.geometry import inverse
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, current_robot_dynamics
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_task2_earlier_preview as original_binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_earlier_restoration_feedback as previous

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_first_hard_restoration_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_first_hard_restoration_20261111.md"
ROOT = REPO / "outputs/task2_first_hard_restoration_20261111"
DRY = REPO / "outputs/task2_first_hard_restoration_dry_20261111/result.json"
QA = REPO / "outputs/qa/task2_first_hard_restoration_20261111/result.json"
PRIOR_PRODUCER = {
    "path": str(REPO / "outputs/task2_earlier_restoration_feedback_20261111/producer/result.json"),
    "sha256": "109ea10f9dc8697398eef0bd72fc999cc007b152ca0137d83034aa5084b03da0",
}
PRIOR_CHECKER = {
    "path": str(REPO / "outputs/task2_earlier_restoration_feedback_20261111/checker/result.json"),
    "sha256": "b8df0bfdab1b9d40ea274c870e6891ea0b8af3c89083f5066cc1be2c39c7ed1e",
}
BRANCHES = ((10, "first_hard_restoration"), (11, "first_hard_restoration"))
CHECKER_FORBIDDEN = previous.CHECKER_FORBIDDEN + (
    "s2a.research.joint_control.preview.first_hard_restoration_feedback",
    "s2a.research.trajectory_search.first_hard_restoration",
)


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/trajectory_search/first_hard_restoration.py",
            REPO / "src/s2a/research/joint_control/preview/first_hard_restoration_feedback.py",
            REPO / "tests/research/trajectory_search/test_first_hard_restoration.py",
            REPO / "tests/research/joint_control/preview/test_first_hard_restoration_feedback.py",
            REPO / "tests/research/native_runtime/test_task2_first_hard_restoration.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-first-hard-restoration.v1",
            case_indices=[10, 11],
            branch="first_hard_restoration",
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
        "Exact first-hard restoration scope and budget",
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
            "Both earlier restoration branches must be closed",
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
                "Every prior restoration branch is retained",
            )
    inputs.update(first_hard_config_ref=config_ref, first_hard_config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        previous=previous.input_receipt(inputs),
        first_hard_config_ref=inputs["first_hard_config_ref"],
        prior_producer_ref=PRIOR_PRODUCER,
        prior_checker_ref=PRIOR_CHECKER,
        source_sha256=inputs["source_sha256"],
    )


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered first-hard restoration branch")
    return f"case_{index:02d}_{mode}"


def make_first_hard_refiner(env, case, receipt):
    """Bind the new candidate to the existing causal preview transaction."""
    from s2a.research.joint_control.preview.completed_history import restore_recomputed_history
    from s2a.research.joint_control.preview.first_hard_restoration_feedback import FirstHardRestorationSolverAudit
    from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
    from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
    from s2a.research.response_models.causal.causal_inner_error import CausalInnerError

    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, velocity, site = current_robot_dynamics(env, jacobian)
    adapter, history = FirstHardRestorationSolverAudit(receipt), CausalInnerError()
    refiner = QualityFeedbackRefiner(
        history,
        receipt,
        start_step=case["spec"]["prefix_steps"],
        start_cursor=case["intervention_spec"]["cursor"],
        solver=adapter.solve,
        auditor=adapter.audit,
        max_quality_response_evaluations=320,
    )
    evidence, completed = case["previous_evidence"], case["completed_history"]
    coupled, tracking, difference = restore_recomputed_history(
        completed,
        evidence["site"],
        evidence["twist"],
        evidence["coupled_prediction"],
        evidence["tracking_prediction"],
        site,
        jacobian @ velocity,
        refiner=refiner,
    )
    site_from_tcp = inverse(case["controller"]["source_from_site"]) @ case["tool"]
    policy = RefinedPreviewFeedback(
        env,
        coupled,
        tracking,
        case["targets"] @ inverse(site_from_tcp),
        case["anchor"] @ inverse(site_from_tcp),
        site_from_tcp,
        case["intervention_spec"]["cursor"],
        completed.previous_command,
        case["controller"]["contract"],
        receipt,
    )
    receipt["initial_innovation_recompute_max_difference"] = difference
    return policy, history


def replay_action(expected):
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN),
        "Checker cannot import first-hard or earlier candidates",
    )
    return previous.replay_action(expected)


def isolated_worker():
    """Bind the new policy to a private copy of the already-qualified worker."""
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
        full=SimpleNamespace(make_full_refiner=make_first_hard_refiner),
        CHECKER_FORBIDDEN=CHECKER_FORBIDDEN,
        replay_action=replay_action,
    )
    worker = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
    worker.__kwdefaults__ = dict(old.__kwdefaults__ or {})
    return worker


def summarize_phase(branches, phase, config):
    if [(x["case"]["case_index"], x["branch"]) for x in branches] != list(BRANCHES):
        raise ValueError("First-hard branches must match the registered matrix")
    normalized = [{**item, "branch": "restoration"} for item in branches]
    counts = previous.summarize_phase(normalized, phase, config)
    if phase == "producer":
        for item in branches:
            runtime = item["runtime"]
            traces = runtime.get("first_hard_restoration_traces", [])
            completed = runtime.get("quality_refiner", {}).get("quality_calls_completed", 0)
            if len(traces) != completed or any(trace.get("audit_status") == "pending" for trace in traces):
                raise ValueError("Every completed first-hard search must retain its final audit status")
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
            "Both first-hard producer branches closed",
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
                "scripts.simulation.runtime.run_task2_first_hard_restoration",
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
        specs, cwd=REPO, timeout_seconds=inputs["first_hard_config"]["worker_timeout_seconds"]
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
            "Both first-hard workers exited and were reaped",
        )
        branch_refs = [prefix.reference(phase_dir / branch_key(*branch) / "result.json") for branch in BRANCHES]
        branches = [prefix.read(ref) for ref in branch_refs]
        counts = summarize_phase(branches, phase, inputs["first_hard_config"])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "First-hard source identity unchanged"
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
