"""Execute and replay the original causal refiner at first-warning endpoints."""

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
from scripts.simulation.runtime import compare_early_warning_tracking as early
from scripts.simulation.runtime import compare_multicase_emergency_retreat as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/early_refiner_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/early_refiner_feedback_20261110.md"
POLICY = REPO / "src/s2a/research/joint_control/preview/refined_feedback.py"
POLICY_TEST = REPO / "tests/research/joint_control/preview/test_refined_feedback.py"
INPUT_TEST = REPO / "tests/research/native_runtime/test_early_refiner_feedback_inputs.py"
QA = REPO / "outputs/qa/early_refiner_feedback_20261110/result.json"
DRY = REPO / "outputs/early_refiner_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/early_refiner_feedback_20261110"
RECON = {
    "path": str(REPO / "outputs/early_refiner_reconstruction_20261110/result.json"),
    "sha256": "650c09893dcd43c0148cd5cdc178ca1c5f67b390be6802b05f645d3f50e82fc0",
}
POLICY_MODULE = "s2a.research.joint_control.preview.refined_feedback"
PREVIEW_MODULE = "s2a.research.joint_control.preview.dependency_safe_preview_runtime"
BRANCHES = ((2, "history"), (2, "refiner"), (6, "history"), (6, "refiner"))


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        POLICY,
        POLICY_TEST,
        INPUT_TEST,
        Path(early.__file__),
        Path(prefix.__file__),
        prefix.CORE,
        prefix.RUNTIME,
        prefix.TEST,
        previous.CORE,
        previous.CORE_TEST,
        REPO / "scripts/simulation/runtime/reconstruct_early_refiner.py",
        REPO / "src/s2a/research/joint_control/preview/completed_history.py",
        REPO / "src/s2a/research/joint_control/preview/tracking_preview_transaction.py",
        REPO / "src/s2a/infrastructure/owned_processes.py",
        REPO / "tests/unit/infrastructure/test_owned_processes.py",
    )
    return [prefix.reference(path) for path in paths]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "Fixed registered branch")
    return f"case_{index:02d}_{mode}"


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config["schema"] == "s2a.early-refiner-feedback.v1"
        and config["cases"] == [2, 6]
        and config["branches"] == ["history", "refiner"]
        and config["continuation_steps"] == 8
        and config["control_dt_s"] == 0.05
        and config["joint_buffer_rad"] == 0.05
        and config["position_cap_m"] == 0.01
        and config["rotation_cap_rad"] == 0.05
        and config["arm_rate_limit"] == 0.25
        and config["absolute_comparison_tolerance"] == 1e-10
        and config["relative_comparison_tolerance"] == 0
        and config["max_control_steps_per_phase"] == 222
        and config["max_native_steps_per_phase"] == 5550
        and config["max_models_per_phase"] == config["max_contexts_per_phase"] == 12
        and config["max_refiner_calls_per_phase"] == 16
        and config["max_refiner_response_evaluations_per_phase"] == 7680
        and config["worker_timeout_seconds"] == 1800
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "Original feedback scope and budgets",
    )
    recon = prefix.read(RECON)
    prefix.require(
        recon["complete"]
        is recon["passed"]
        is recon["sources_unchanged"]
        is recon["original_decisions_reproduced"]
        is True,
        "Both original decisions reconstructed",
    )
    inputs = early.load_inputs()
    inputs["config_feedback"] = config
    inputs["source_sha256"][str(CONFIG)] = config_ref["sha256"]
    inputs["source_sha256"][RECON["path"]] = RECON["sha256"]
    return inputs


def checker_action(expected):
    prefix.require(expected is not None, "Saved actions required for independent replay")
    prefix.require(
        POLICY_MODULE not in sys.modules and PREVIEW_MODULE not in sys.modules,
        "Checker must not import feedback or preview",
    )

    def action(_q, _limits, _jacobian, _previous, offset):
        return {"command": np.asarray(expected[offset]["command"], dtype=float), "newly_selected": False}

    return action


def completed_case(index, inputs):
    """Load only the previously completed history for a producer candidate."""
    from scripts.simulation.runtime.reconstruct_early_refiner import load_inputs as reconstruction_inputs

    restored = reconstruction_inputs()
    prefix.require(
        restored["asset_refs"] == inputs["asset_refs"]
        and all(inputs["source_sha256"].get(path) == digest for path, digest in restored["source_sha256"].items()),
        "Completed-history input identity",
    )
    return next(item for item in restored["cases"] if item["spec"]["case_index"] == index)


def make_refiner(env, case, receipt):
    from s2a.research.joint_control.preview.completed_history import restore_recomputed_history
    from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback

    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, velocity, site = current_robot_dynamics(env, jacobian)
    twist = jacobian @ velocity
    evidence, historical = case["previous_evidence"], case["completed_history"]
    coupled, tracking, difference = restore_recomputed_history(
        historical,
        evidence["site"],
        evidence["twist"],
        evidence["coupled_prediction"],
        evidence["tracking_prediction"],
        site,
        twist,
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
        historical.previous_command,
        case["controller"]["contract"],
        receipt,
    )
    receipt["initial_innovation_recompute_max_difference"] = difference
    return policy


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
    if phase == "producer" and mode == "refiner":
        case = completed_case(index, inputs)
    expected, expected_stop = None, None
    if phase == "checker":
        original = prefix.read(prefix.reference(ROOT / "producer" / branch_key(index, mode) / "result.json"))
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch")
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
            historical, on_completed = None, None
            if phase == "checker":
                action = checker_action(expected)
            elif mode == "history":
                action, historical = previous.make_action("history", case)
            else:
                policy = make_refiner(env, case, receipt)
                action, on_completed = policy.select, policy.observe
            return run_continuation(
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
                stop_on_outside=mode != "history",
                expected_periods=expected,
                historical_states=historical,
                on_completed=on_completed,
                progress=progress,
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
                POLICY_MODULE not in sys.modules
                and PREVIEW_MODULE not in sys.modules
                and receipt["continuation"]["stop"] == expected_stop
                and receipt["newly_selected_commands"] == 0
                and receipt["preview_calls"]["attempted"] == receipt["refiner_calls"]["attempted"] == 0,
                "Checker replays actions without decision calls",
            )
        result.update(
            complete=True,
            passed=True,
            runtime=receipt,
            candidate_policy_imported=POLICY_MODULE in sys.modules,
            preview_module_imported=PREVIEW_MODULE in sys.modules,
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
            "reconstruction_ref": RECON,
            "producer_ref": producer_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "branches": [list(b) for b in BRANCHES],
            "max_control_steps": 222,
            "max_native_steps": 5550,
            "max_refiner_calls": 16,
            "max_refiner_response_evaluations": 7680,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    specs = []
    for index, mode in BRANCHES:
        key = branch_key(index, mode)
        specs.append(
            {
                "key": key,
                "argv": [
                    sys.executable,
                    "-m",
                    "scripts.simulation.runtime.run_early_refiner_feedback",
                    "--worker",
                    phase,
                    str(index),
                    mode,
                ],
                "log": directory / key / "worker.log",
            }
        )
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
            and all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle["workers"]),
            "All four workers closed",
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
        evaluations = sum(b["runtime"]["refiner_response_evaluations"] for b in branches)
        prefix.require(
            controls <= 222
            and native == controls * 25
            and native <= 5550
            and models <= 12
            and contexts <= 12
            and refiners <= 16
            and evaluations <= 7680,
            "Original phase budgets",
        )
        if phase == "checker":
            prefix.require(
                refiners == 0
                and all(
                    not b["candidate_policy_imported"]
                    and not b["preview_module_imported"]
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
            newly_selected_commands=sum(b["runtime"]["newly_selected_commands"] for b in branches),
        )
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


if __name__ == "__main__":
    main()
