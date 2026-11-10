"""Apply past-only quality feedback to the newly replanned reference."""

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
from scripts.simulation.runtime import run_combined_warning_feedback as base
from scripts.simulation.runtime import run_replan_reference_switch as switch

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/replan_quality_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/replan_quality_feedback_20261110.md"
QA = REPO / "outputs/qa/replan_quality_feedback_20261110/result.json"
DRY = REPO / "outputs/replan_quality_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/replan_quality_feedback_20261110"
BASE_PRODUCER = {
    "path": str(switch.ROOT / "producer/result.json"),
    "sha256": "58bf371b022e9daded4929df1904cc6f15ea4866f7e34791cb359f4a5f65044e",
}
BASE_CHECKER = {
    "path": str(switch.ROOT / "checker/result.json"),
    "sha256": "ba09b7ca0a4ca39de3edeb545e09db580ff5a8a7cad4af77e71f3c368c87102b",
}
CASES = (2, 6)


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "tests/research/native_runtime/test_replan_quality_feedback_inputs.py",
    )
    return switch.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.replan-quality-feedback.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "prefix_steps": 16,
            "reference_chunk_index": 2,
            "reference_cursor": 0,
            "branch": "quality_new",
            "continuation_steps": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "braking_time_s": 0.1,
            "max_control_steps_per_phase": 48,
            "max_native_steps_per_phase": 1200,
            "max_models_per_phase": 6,
            "max_contexts_per_phase": 6,
            "max_refiner_calls_per_phase": 16,
            "max_refiner_response_evaluations_per_phase": 7680,
            "max_quality_calls_per_phase": 16,
            "max_quality_response_evaluations_per_phase": 2560,
            "max_audit_endpoint_queries_per_phase": 3200,
            "max_audit_transition_steps_per_phase": 41600,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered new-reference quality comparison",
    )
    prior = []
    for ref in (BASE_PRODUCER, BASE_CHECKER):
        result = prefix.read(ref)
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and len(result["branch_refs"]) == 4,
            "The matched old/new producer and checker are closed",
        )
        prior.append(result)
    inputs = switch.load_inputs()
    refs = inputs["source_sha256"]
    refs.update(
        {
            str(CONFIG): config_ref["sha256"],
            BASE_PRODUCER["path"]: BASE_PRODUCER["sha256"],
            BASE_CHECKER["path"]: BASE_CHECKER["sha256"],
        }
    )
    for result in prior:
        for position in (0, 2):
            ref = result["branch_refs"][position]
            prefix.checked(ref)
            refs[ref["path"]] = ref["sha256"]
    return inputs


def read_actual_tcp(env, case):
    """Read the completed robot endpoint without stepping or forwarding physics."""
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    return site @ inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]


def after_quality_completed(quality_done, receipt, case, expected=None, *, tcp_reader=read_actual_tcp):
    """Update completed control history before scoring this same actual endpoint."""

    def completed(env, command, offset):
        history = quality_done(env, command, offset)
        period = receipt["continuation_partial"]["periods"][-1]
        prefix.require(period["step"] == 16 + offset, "Only the just-completed endpoint is scored")
        actual_tcp = tcp_reader(env, case)
        expected_score = None if expected is None else expected[offset]["completed_feedback"]
        score = switch.completed_reference_scores(
            actual_tcp, case["scoring_targets"], "new", offset, period["tcp_error"], expected=expected_score
        )
        score["policy_history"] = history
        return score

    return completed


def branch_key(index):
    prefix.require(index in CASES, "Fixed registered quality branch")
    return f"case_{index:02d}_quality_new"


def worker(phase, index):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / phase / branch_key(index)
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen branch inputs and sources",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    selected = switch.selected_case(case, "new")
    expected, expected_stop = None, None
    if phase == "checker":
        original = prefix.read(prefix.reference(ROOT / "producer" / branch_key(index) / "result.json"))
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch required")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "case_index": index,
        "branch": "quality_new",
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
            if phase == "checker":
                action = base.quality.checker_action(expected)

                def quality_done(_env, _command, _offset):
                    return None
            else:
                policy, history = base.quality.make_quality_refiner(env, selected, receipt)
                select, quality_done = base.quality.quality_callbacks(policy, history, receipt, 16)
                action = base.guarded_action(select, receipt)
            on_completed = after_quality_completed(quality_done, receipt, case, expected)
            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=16,
                previous_command=case["commands"][-1],
                targets=selected["fixed_targets"],
                tool=case["tool"],
                target_object=selected["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=on_completed,
                progress=progress,
            )
            if phase == "checker":
                return base.reconcile_checker_stop(outcome, expected_stop, mode="quality", start_step=16)
            return outcome

        def progress(receipt):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "case": index,
                        "checked_prefix_endpoints": receipt.get("checked_endpoints", 0),
                        "control_completed": receipt.get("control_steps_completed", 0),
                        "new_commands": receipt.get("newly_selected_commands", 0),
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
                base.quality.QUALITY_MODULE not in sys.modules
                and base.quality.baseline.POLICY_MODULE not in sys.modules
                and base.quality.baseline.PREVIEW_MODULE not in sys.modules
                and receipt["newly_selected_commands"] == 0
                and receipt["preview_calls"]["attempted"] == receipt["refiner_calls"]["attempted"] == 0,
                "Checker only replays saved actions and scores actual states",
            )
        result.update(
            complete=True,
            passed=True,
            runtime=receipt,
            candidate_policy_imported=base.quality.baseline.POLICY_MODULE in sys.modules,
            preview_module_imported=base.quality.baseline.PREVIEW_MODULE in sys.modules,
            quality_module_imported=base.quality.QUALITY_MODULE in sys.modules,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def run_phase(phase):
    directory = ROOT / phase
    prefix.require(not directory.exists(), "New phase output only; no automatic retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Zero-model dry closed")
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
        "Dry source and input identity",
    )
    producer_ref = None
    if phase == "checker":
        producer_ref = prefix.reference(ROOT / "producer/result.json")
        producer = prefix.read(producer_ref)
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 2,
            "Quality producer must complete before replay",
        )
        for ref in producer["branch_refs"]:
            prefix.checked(ref)
    registration = publish(
        directory / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "baseline_producer_ref": BASE_PRODUCER,
            "baseline_checker_ref": BASE_CHECKER,
            "producer_ref": producer_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": list(CASES),
            "max_control_steps": 48,
            "max_native_steps": 1200,
            "max_refiner_calls": 16,
            "max_refiner_response_evaluations": 7680,
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
            "key": branch_key(index),
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_replan_quality_feedback",
                "--worker",
                phase,
                str(index),
            ],
            "log": directory / branch_key(index) / "worker.log",
        }
        for index in CASES
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
            and len(lifecycle["workers"]) == 2
            and all(
                worker["returncode"] == 0 and worker["actual_wait"] and worker["reaped"] and not worker["signal_sent"]
                for worker in lifecycle["workers"]
            ),
            "Both workers closed and reaped",
        )
        refs = [prefix.reference(directory / branch_key(index) / "result.json") for index in CASES]
        branches = [prefix.read(ref) for ref in refs]
        prefix.require(all(branch["complete"] is branch["passed"] is True for branch in branches), "Complete branches")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        controls = sum(branch["runtime"]["control_steps_completed"] for branch in branches)
        native = sum(branch["runtime"]["counters"]["mj_step"]["completed"] for branch in branches)
        models = sum(branch["runtime"]["counters"]["model_construct"]["completed"] for branch in branches)
        contexts = sum(branch["runtime"]["counters"]["context"]["completed"] for branch in branches)
        refiners = sum(branch["runtime"]["refiner_calls"]["attempted"] for branch in branches)
        evaluations = sum(
            branch["runtime"].get("quality_refiner", {}).get("baseline_response_evaluations", 0) for branch in branches
        )
        quality_calls = sum(
            branch["runtime"].get("quality_refiner", {}).get("quality_calls_attempted", 0) for branch in branches
        )
        quality_responses = sum(
            branch["runtime"].get("quality_refiner", {}).get("quality_response_evaluations", 0) for branch in branches
        )
        endpoint_queries = sum(
            branch["runtime"].get("quality_refiner", {}).get("endpoint_queries_charged", 0) for branch in branches
        )
        transition_steps = sum(
            branch["runtime"].get("quality_refiner", {}).get("transition_steps_charged", 0) for branch in branches
        )
        prefix.require(
            controls <= 48
            and native == controls * 25
            and native <= 1200
            and models <= 6
            and contexts <= 6
            and refiners <= 16
            and evaluations <= 7680
            and quality_calls <= 16
            and quality_responses <= 2560
            and endpoint_queries <= 3200
            and transition_steps <= 41600,
            "Registered phase budgets",
        )
        if phase == "checker":
            prefix.require(
                refiners
                == evaluations
                == quality_calls
                == quality_responses
                == endpoint_queries
                == transition_steps
                == 0
                and all(
                    not branch["candidate_policy_imported"]
                    and not branch["preview_module_imported"]
                    and not branch["quality_module_imported"]
                    and branch["runtime"]["newly_selected_commands"] == 0
                    for branch in branches
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
            newly_selected_commands=sum(branch["runtime"]["newly_selected_commands"] for branch in branches),
        )
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--produce", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
        inputs = load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "prefix_steps": [case["spec"]["prefix_steps"] for case in inputs["cases"]],
                    "completed_history_periods": [
                        case["completed_history"].completed_periods for case in inputs["cases"]
                    ],
                    "models": 0,
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
        phase, index = args.worker
        prefix.require(phase in ("producer", "checker"), "Registered phase")
        worker(phase, int(index))


if __name__ == "__main__":
    main()
