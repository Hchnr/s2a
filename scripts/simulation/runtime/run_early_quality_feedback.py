"""Execute and independently replay past-only quality-constrained feedback."""

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_early_refiner_feedback as baseline

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/early_quality_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/early_quality_feedback_20261110.md"
QA = REPO / "outputs/qa/early_quality_feedback_20261110/result.json"
DRY = REPO / "outputs/early_quality_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/early_quality_feedback_20261110"
DIAGNOSTIC = {
    "path": str(REPO / "outputs/early_quality_feasibility_v2_20261110/result.json"),
    "sha256": "a035a84aaf1b98de08a6dab3d500ed177ec280b8f3f82cd5dd93178959760821",
}
QUALITY_MODULE = "s2a.research.joint_control.preview.quality_feedback_refiner"
BRANCHES = ((2, "original"), (2, "quality"), (6, "original"), (6, "quality"))


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/joint_control/preview/quality_feedback_refiner.py",
        REPO / "src/s2a/research/joint_control/tracking/early_quality_diagnostic.py",
        REPO / "src/s2a/research/response_models/causal/causal_inner_error.py",
        REPO / "src/s2a/research/trajectory_search/quality_capped_inner_joint_plan.py",
        REPO / "tests/research/joint_control/preview/test_quality_feedback_refiner.py",
        REPO / "tests/research/native_runtime/test_early_quality_feedback_inputs.py",
    )
    return baseline.source_refs() + [prefix.reference(path) for path in paths]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "Fixed registered quality feedback branch")
    return f"case_{index:02d}_{mode}"


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.early-quality-feedback.v1",
            "cases": [2, 6],
            "branches": ["original", "quality"],
            "continuation_steps": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "braking_time_s": 0.1,
            "absolute_comparison_tolerance": 1e-10,
            "relative_comparison_tolerance": 0.0,
            "max_control_steps_per_phase": 222,
            "max_native_steps_per_phase": 5550,
            "max_models_per_phase": 12,
            "max_contexts_per_phase": 12,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "max_quality_calls_per_phase": 16,
            "max_quality_response_evaluations_per_phase": 2560,
            "max_audit_endpoint_queries_per_phase": 3200,
            "max_audit_transition_steps_per_phase": 41600,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered quality feedback scope and budgets",
    )
    diagnostic = prefix.read(DIAGNOSTIC)
    prefix.require(
        diagnostic["complete"] is diagnostic["passed"] is diagnostic["sources_unchanged"] is True
        and diagnostic["physics_steps"] == diagnostic["world_model_calls"] == diagnostic["fresh_cases"] == 0,
        "Prior quality diagnostic closed without physics or model calls",
    )
    inputs = baseline.load_inputs()
    inputs["source_sha256"][str(CONFIG)] = config_ref["sha256"]
    inputs["source_sha256"][DIAGNOSTIC["path"]] = DIAGNOSTIC["sha256"]
    return inputs


def checker_action(expected):
    prefix.require(QUALITY_MODULE not in sys.modules, "Checker must not import the quality candidate")
    return baseline.checker_action(expected)


def make_quality_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.completed_history import restore_recomputed_history
    from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
    from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
    from s2a.research.response_models.causal.causal_inner_error import CausalInnerError

    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, velocity, site = current_robot_dynamics(env, jacobian)
    twist = jacobian @ velocity
    evidence, completed = case["previous_evidence"], case["completed_history"]
    history = CausalInnerError()
    refiner = QualityFeedbackRefiner(
        history,
        receipt,
        start_step=case["spec"]["prefix_steps"],
        start_cursor=case["intervention_spec"]["cursor"],
    )
    coupled, tracking, difference = restore_recomputed_history(
        completed,
        evidence["site"],
        evidence["twist"],
        evidence["coupled_prediction"],
        evidence["tracking_prediction"],
        site,
        twist,
        refiner=refiner,
    )
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    targets = case["targets"] @ inverse(site_from_tcp)
    anchor = case["anchor"] @ inverse(site_from_tcp)
    policy = RefinedPreviewFeedback(
        env,
        coupled,
        tracking,
        targets,
        anchor,
        site_from_tcp,
        case["intervention_spec"]["cursor"],
        completed.previous_command,
        case["controller"]["contract"],
        receipt,
    )
    receipt["initial_innovation_recompute_max_difference"] = difference
    return policy, history


def quality_callbacks(policy, history, receipt, start_step):
    from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped

    def action(q, limits, jacobian, previous, offset):
        try:
            return policy.select(q, limits, jacobian, previous, offset)
        except QualitySelectionStopped as error:
            return error.decision()

    def completed(env, command, offset):
        observed = policy.observe(env, command, offset)
        period = receipt["continuation_partial"]["periods"][-1]
        prefix.require(period["step"] == start_step + offset, "Just-completed period required")
        residual = history.complete(start_step + offset, [item["q"] for item in period["actual_inner"]])
        observed["completed_inner_error_rad"] = residual
        observed["past_error_after_completion"] = history.snapshot()
        return observed

    return action, completed


def reconcile_stop(continuation, expected_stop, *, phase, mode, start_step):
    """Replay physical stops; record decision stops without re-solving them."""
    if phase != "checker":
        return continuation
    if expected_stop is not None and expected_stop.get("reason") in ("quality_no_witness", "quality_witness_rejected"):
        prefix.require(
            mode == "quality"
            and continuation["stop"] is None
            and len(continuation["periods"]) < 8
            and expected_stop.get("at_step") == start_step + len(continuation["periods"])
            and expected_stop.get("next_action_dispatched") is False,
            "Decision stop position must follow exactly the replayed actions",
        )
        continuation["stop"] = expected_stop.copy()
        continuation["decision_stop_recorded_not_recomputed"] = True
    else:
        prefix.require(continuation["stop"] == expected_stop, "Replayed physical stop differs from producer")
        continuation["decision_stop_recorded_not_recomputed"] = False
    return continuation


def worker(phase, index, mode):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / phase / branch_key(index, mode)
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen branch inputs and sources",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    if phase == "producer":
        case = baseline.completed_case(index, inputs)
    expected, expected_stop = None, None
    if phase == "checker":
        original = prefix.read(prefix.reference(ROOT / "producer" / branch_key(index, mode) / "result.json"))
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch required")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "case_index": index,
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            receipt["preview_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_response_evaluations"] = 0
            on_completed = None
            if phase == "checker":
                action = checker_action(expected)
            elif mode == "original":
                policy = baseline.make_refiner(env, case, receipt)
                action, on_completed = policy.select, policy.observe
            else:
                policy, history = make_quality_refiner(env, case, receipt)
                action, on_completed = quality_callbacks(policy, history, receipt, case["spec"]["prefix_steps"])
            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=case["spec"]["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=case["fixed_targets"],
                tool=case["tool"],
                target_object=case["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=on_completed,
                progress=progress,
            )
            return reconcile_stop(
                outcome, expected_stop, phase=phase, mode=mode, start_step=case["spec"]["prefix_steps"]
            )

        def progress(receipt):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "case": index,
                        "branch": mode,
                        "checked_prefix_endpoints": receipt.get("checked_endpoints", 0),
                        "control_completed": receipt.get("control_steps_completed", 0),
                        "new_commands": receipt.get("newly_selected_commands", 0),
                        "preview": receipt.get("preview_calls"),
                        "refiner": receipt.get("refiner_calls"),
                        "quality": receipt.get("quality_refiner"),
                        "native_steps": receipt["counters"]["mj_step"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        try:
            receipt = run_prefix_runtime(
                case["config"],
                case["record"],
                case["record_xml"],
                case["commands"],
                case["saved"],
                case["receipts"],
                directory / "runtime",
                inputs["asset_refs"],
                case["controller"],
                continuation=continuation,
                continuation_budget=8,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            prefix.require(
                QUALITY_MODULE not in sys.modules
                and baseline.POLICY_MODULE not in sys.modules
                and baseline.PREVIEW_MODULE not in sys.modules
                and receipt["newly_selected_commands"] == 0
                and receipt["preview_calls"]["attempted"] == receipt["refiner_calls"]["attempted"] == 0,
                "Checker replays saved actions without candidate decisions",
            )
        result.update(
            complete=True,
            passed=True,
            runtime=receipt,
            candidate_policy_imported=baseline.POLICY_MODULE in sys.modules,
            preview_module_imported=baseline.PREVIEW_MODULE in sys.modules,
            quality_module_imported=QUALITY_MODULE in sys.modules,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--produce", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
        inputs = load_inputs()
        completed = [baseline.completed_case(index, inputs) for index in (2, 6)]
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "cases": [x["intervention_spec"] for x in inputs["cases"]],
                    "first_warnings": [x["first_warning"] for x in inputs["cases"]],
                    "completed_history_periods": [x["completed_history"].completed_periods for x in completed],
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.produce:
        run_phase("producer")
    elif args.check:
        run_phase("checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "Registered phase")
        worker(phase, int(index), mode)


def run_phase(phase):
    directory = ROOT / phase
    prefix.require(not directory.exists(), "New phase output only; no automatic retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Dry closed")
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources, inputs = source_refs(), load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source/input identity",
    )
    producer_ref = None
    if phase == "checker":
        producer_ref = prefix.reference(ROOT / "producer/result.json")
        producer = prefix.read(producer_ref)
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 4,
            "Producer complete before checker",
        )
        for ref in producer["branch_refs"]:
            prefix.checked(ref)
    registration = publish(
        directory / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "diagnostic_ref": DIAGNOSTIC,
            "producer_ref": producer_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "branches": [list(b) for b in BRANCHES],
            "max_control_steps": 222,
            "max_native_steps": 5550,
            "max_refiner_calls": 32,
            "max_refiner_response_evaluations": 15360,
            "max_quality_calls": 16,
            "max_quality_response_evaluations": 2560,
            "max_audit_endpoint_queries": 3200,
            "max_audit_transition_steps": 41600,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    specs = [
        {
            "key": branch_key(index, mode),
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_early_quality_feedback",
                "--worker",
                phase,
                str(index),
                mode,
            ],
            "log": directory / branch_key(index, mode) / "worker.log",
        }
        for index, mode in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(directory / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 4
            and all(
                w["returncode"] == 0 and w["actual_wait"] and w["reaped"] and not w["signal_sent"]
                for w in lifecycle["workers"]
            ),
            "All four workers closed and reaped",
        )
        refs = [prefix.reference(directory / branch_key(i, mode) / "result.json") for i, mode in BRANCHES]
        branches = [prefix.read(ref) for ref in refs]
        prefix.require(all(b["complete"] is b["passed"] is True for b in branches), "Complete branch results")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        controls = sum(b["runtime"]["control_steps_completed"] for b in branches)
        native = sum(b["runtime"]["counters"]["mj_step"]["completed"] for b in branches)
        models = sum(b["runtime"]["counters"]["model_construct"]["completed"] for b in branches)
        contexts = sum(b["runtime"]["counters"]["context"]["completed"] for b in branches)
        refiners = sum(b["runtime"]["refiner_calls"]["attempted"] for b in branches)
        evaluations = sum(
            b["runtime"].get("quality_refiner", {}).get("baseline_response_evaluations", 0)
            if b["branch"] == "quality"
            else b["runtime"]["refiner_response_evaluations"]
            for b in branches
        )
        quality_calls = sum(b["runtime"].get("quality_refiner", {}).get("quality_calls_attempted", 0) for b in branches)
        quality_responses = sum(
            b["runtime"].get("quality_refiner", {}).get("quality_response_evaluations", 0) for b in branches
        )
        endpoint_queries = sum(
            b["runtime"].get("quality_refiner", {}).get("endpoint_queries_charged", 0) for b in branches
        )
        transition_steps = sum(
            b["runtime"].get("quality_refiner", {}).get("transition_steps_charged", 0) for b in branches
        )
        prefix.require(
            controls <= 222
            and native == controls * 25
            and native <= 5550
            and models <= 12
            and contexts <= 12
            and refiners <= 32
            and evaluations <= 15360
            and quality_calls <= 16
            and quality_responses <= 2560
            and endpoint_queries <= 3200
            and transition_steps <= 41600,
            "Registered phase budgets",
        )
        if phase == "checker":
            prefix.require(
                refiners == quality_calls == quality_responses == endpoint_queries == transition_steps == 0
                and all(
                    not b["candidate_policy_imported"]
                    and not b["preview_module_imported"]
                    and not b["quality_module_imported"]
                    and b["runtime"]["newly_selected_commands"] == 0
                    for b in branches
                ),
                "No independent checker decisions",
            )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_refs=refs,
            control_steps=controls,
            native_steps=native,
            model_initializations=models,
            software_contexts=contexts,
            refiner_calls=refiners,
            refiner_response_evaluations=evaluations,
            quality_calls=quality_calls,
            quality_response_evaluations=quality_responses,
            audit_endpoint_queries_charged=endpoint_queries,
            audit_transition_steps_charged=transition_steps,
            newly_selected_commands=sum(b["runtime"]["newly_selected_commands"] for b in branches),
        )
    finally:
        print(publish(directory / "result.json", result), flush=True)


if __name__ == "__main__":
    main()
