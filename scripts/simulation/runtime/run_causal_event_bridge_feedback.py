"""Execute and independently replay one fixed causal reference transition."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import evaluate_causal_event_bridge as scoring
from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as base
from scripts.simulation.runtime import run_causal_event_long_feedback as long
from scripts.simulation.runtime import run_causal_event_switch_feedback as switch

REPO = prefix.REPO
QA = REPO / "outputs/qa/causal_event_bridge_feedback_20261110/result.json"
DRY = REPO / "outputs/causal_event_bridge_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/causal_event_bridge_feedback_20261110"
PRODUCER_MODULE = "s2a.research.wam.tracking.preview.causal_reference_bridge"


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            Path(scoring.__file__),
            REPO / "tests/research/native_runtime/test_causal_event_bridge_scoring.py",
            REPO / "tests/research/native_runtime/test_causal_event_bridge_feedback.py",
        )
    ]


def reference_document(inputs):
    """Producer creates one immutable reference from the registered source chunks."""
    reference = binding.prepare_reference(inputs["case"], inputs["config"])
    audit = binding.verify_reference(inputs["case"], inputs["config"], reference)
    return {
        "schema": "s2a.causal-event-bridge-reference.v1",
        "case_id": inputs["config"]["case_id"],
        "config_ref": inputs["config_ref"],
        "source_sha256": inputs["source_sha256"],
        "reference": reference,
        "independent_construction_check": audit,
    }


def checked_reference(inputs, document):
    """Reconstruct independently without importing the bridge producer."""
    prefix.require(
        document["schema"] == "s2a.causal-event-bridge-reference.v1"
        and document["case_id"] == inputs["config"]["case_id"]
        and document["config_ref"] == inputs["config_ref"]
        and document["source_sha256"] == inputs["source_sha256"],
        "Transition provenance and source identity",
    )
    audit = binding.verify_reference(inputs["case"], inputs["config"], document["reference"])
    prefix.require(audit == document["independent_construction_check"], "Independently reconstructed reference")
    return document["reference"], audit


def require_checker_independence():
    base.checker_action([])
    prefix.require(PRODUCER_MODULE not in sys.modules, "Checker must not import the bridge producer")


def worker(phase):
    prefix.environment()
    directory = ROOT / phase / "bridge"
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen bridge sources and inputs",
    )
    if phase == "checker":
        require_checker_independence()
    reference, reference_audit = checked_reference(inputs, prefix.read(registration["reference_ref"]))
    case, config = inputs["case"], inputs["config"]
    old, latest = binding.selected_references(case)
    targets = np.asarray(reference["poses"], dtype=float)[:16].copy()
    selected = {"targets": reference["poses"], "openings": reference["openings"], "anchor": latest["anchor"]}
    preceding = [
        prefix.read(inputs["prior_branches"]["remembered"]["producer_ref"]),
        prefix.read(inputs["direct_branches"]["producer"]),
    ]
    previous_periods = [item["runtime"]["continuation"]["periods"] for item in preceding]
    expected, expected_stop = None, None
    if phase == "checker":
        producer = prefix.read(registration["producer_branch_ref"])
        prefix.require(producer["complete"] is producer["passed"] is True, "Closed bridge producer")
        expected, expected_stop = (
            producer["runtime"]["continuation"]["periods"],
            producer["runtime"]["continuation"]["stop"],
        )
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "case": case["spec"],
        "reference_ref": registration["reference_ref"],
        "independent_construction_check": reference_audit,
        "automatic_retry": False,
    }
    try:
        gripper = None if phase == "checker" else base.reconstruct_gripper(case, result)
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            for item in preceding:
                base.check_prefix_inner(receipt["final_prefix_inner"], item["runtime"]["final_prefix_inner"])
            start_position = long.target_position(env, case["spec"]["target"])
            receipt["target_start_position"], receipt["previous_verified_steps"] = start_position, 0
            policy = None
            if phase == "checker":
                np.testing.assert_allclose(
                    start_position, producer["runtime"]["target_start_position"], atol=1e-10, rtol=0
                )
                action = base.checker_action(expected)
            else:
                policy = base.make_feedback(env, case, old, gripper, receipt, max_steps=16)

                def action(q, limits, jacobian, previous, offset):
                    if offset == config["bridge_start_offset"] and not policy.last_inner_loss:
                        switch.switch_reference(policy, selected, receipt)
                    return policy.select(q, limits, jacobian, previous, offset)

            def observed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == config["prefix_steps"] + offset, "Just-completed bridge endpoint")
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [item["q"] for item in period["actual_inner"]]
                    )
                expected_score = None if expected is None else expected[offset]["completed_feedback"]["scores"]
                scores = scoring.score_completed(
                    actual_env, case, reference, offset, period["tcp_error"], expected=expected_score
                )
                position = long.target_position(actual_env, case["spec"]["target"])
                height = float(position[2] - start_position[2])
                if offset < config["bridge_start_offset"]:
                    for earlier in previous_periods:
                        prior = earlier[offset]
                        long.check_previous_period(period, prior)
                        prior_feedback = prior["completed_feedback"]
                        scoring.compare_scores(scores, prior_feedback["scores"], names=("latest", "remembered"))
                        np.testing.assert_allclose(position, prior_feedback["object_position"], atol=1e-10, rtol=0)
                        np.testing.assert_allclose(height, prior_feedback["object_height_change_m"], atol=1e-10, rtol=0)
                    receipt["previous_verified_steps"] += 1
                mode = "remembered" if offset < config["bridge_start_offset"] else "bridge"
                if expected is not None:
                    recorded = expected[offset]["completed_feedback"]
                    prefix.require(recorded["reference_used"] == mode, "Replayed bridge switch time")
                    np.testing.assert_allclose(position, recorded["object_position"], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(height, recorded["object_height_change_m"], atol=1e-10, rtol=0)
                return {
                    "scores": scores,
                    "reference_used": mode,
                    "object_position": position,
                    "object_height_change_m": height,
                    "policy_history": history,
                    "policy_history_recomputed": policy is not None,
                }

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=config["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=targets,
                tool=case["tool"],
                target_object=case["spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=observed,
                progress=progress,
                continuation_limit=16,
            )
            prefix.require(receipt["previous_verified_steps"] == 8, "Both earlier eight-period outcomes reproduce")
            if phase == "checker":
                q, limits, _jac = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                outcome = base.reconcile_stop(
                    outcome,
                    expected_stop,
                    start_step=config["prefix_steps"],
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                    horizon=16,
                )
                outcome["switch_decision_not_recomputed"] = True
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 64 == 0 or completed >= config["prefix_steps"]:
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "control_completed": completed,
                            "new_commands": receipt.get("newly_selected_commands", 0),
                            "native_steps": receipt["counters"]["mj_step"],
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )

        try:
            runtime = run_prefix_runtime(
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
                continuation_budget=16,
                continuation_limit=16,
                registered_seed=42,
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(complete=True, passed=True, runtime=runtime)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute(phase):
    prefix.environment()
    if phase == "checker":
        require_checker_independence()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen bridge CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == binding.input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves bridge inputs",
    )
    producer_ref, reference_ref = None, None
    if phase == "checker":
        producer = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True,
            "Bridge producer closed",
        )
        producer_ref, reference_ref = producer["branch_ref"], producer["reference_ref"]
        checked_reference(inputs, prefix.read(reference_ref))
    phase_dir = ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    if phase == "producer":
        reference_ref = publish(phase_dir / "reference.json", reference_document(inputs))
    registration_ref = publish(
        phase_dir / "registration.json",
        {
            "phase": phase,
            "inputs": binding.input_receipt(inputs),
            "source_refs": sources,
            "reference_ref": reference_ref,
            "producer_branch_ref": producer_ref,
        },
    )
    specs = [
        {
            "key": "bridge",
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_causal_event_bridge_feedback",
                "--worker",
                phase,
            ],
            "log": phase_dir / "bridge/worker.log",
        }
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(phase_dir / "workers.json", lifecycle)
    branch_path = phase_dir / "bridge/result.json"
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "registration_ref": registration_ref,
        "reference_ref": reference_ref,
        "workers_ref": workers_ref,
        "available_branch_ref": prefix.reference(branch_path) if branch_path.is_file() else None,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 1
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "Registered bridge worker closed and was reaped",
        )
        branch_ref = prefix.reference(branch_path)
        branch = prefix.read(branch_ref)
        runtime, config = branch["runtime"], inputs["config"]
        prefix.require(
            branch["complete"] is branch["passed"] is runtime["cleanup_complete"] is True
            and runtime["previous_verified_steps"] == 8,
            "Both earlier eight-period outcomes and cleanup verified",
        )
        control, native = runtime["control_steps_completed"], runtime["counters"]["mj_step"]["completed"]
        models, contexts = (
            runtime["counters"]["model_construct"]["completed"],
            runtime["counters"]["context"]["completed"],
        )
        selected = runtime["newly_selected_commands"]
        prefix.require(
            control <= config["max_control_steps_per_phase"]
            and native == 25 * control
            and native <= config["max_native_steps_per_phase"]
            and models == config["max_models_per_phase"]
            and contexts == config["max_contexts_per_phase"],
            "Registered bridge physical budget",
        )
        counts = {
            "control_steps": control,
            "native_steps": native,
            "model_initializations": models,
            "contexts": contexts,
            "newly_selected_commands": selected,
        }
        if phase == "producer":
            prefix.require(runtime["reference_switch"]["completed"] is True, "Transition at completed boundary")
            stop = runtime["continuation"]["stop"]
            if stop and stop["reason"] in ("quality_no_witness", "quality_witness_rejected"):
                prefix.require(
                    stop.get("gripper_proposal_dispatched") is False and "gripper_proposal" in stop,
                    "Rejected gripper proposal is recorded and not dispatched",
                )
            history, proposal = branch["gripper_history"], runtime.get("gripper_proposals", {})
            calls = history["planner_calls"] + proposal.get("planner_calls", 0)
            sequences = history["candidate_sequences"] + proposal.get("candidate_sequences", 0)
            refiner_calls = runtime.get("refiner_calls", {}).get("attempted", 0)
            refiner_evals = runtime.get("quality_refiner", {}).get("baseline_response_evaluations", 0)
            quality_calls = runtime.get("quality_refiner", {}).get("quality_calls_attempted", 0)
            quality_evals = runtime.get("quality_refiner", {}).get("quality_response_evaluations", 0)
            prefix.require(
                calls <= config["max_gripper_planner_calls_per_phase"]
                and sequences <= config["max_gripper_candidate_sequences_per_phase"]
                and refiner_calls <= config["max_refiner_calls_per_phase"]
                and refiner_evals <= config["max_refiner_response_evaluations_per_phase"]
                and quality_calls <= config["max_quality_calls_per_phase"]
                and quality_evals <= config["max_quality_response_evaluations_per_phase"],
                "Registered bridge solver budgets",
            )
            counts.update(
                gripper_planner_calls=calls,
                gripper_candidate_sequences=sequences,
                refiner_calls=refiner_calls,
                refiner_response_evaluations=refiner_evals,
                quality_calls=quality_calls,
                quality_response_evaluations=quality_evals,
            )
        else:
            prefix.require(selected == 0, "Bridge checker only replays actual actions")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged bridge sources and inputs"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    finally:
        print(publish(phase_dir / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", choices=("producer", "checker"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        inputs = binding.load_inputs()
        document = reference_document(inputs)
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "inputs": binding.input_receipt(inputs),
                    "independent_construction_check": document["independent_construction_check"],
                    "model_initializations": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.producer or args.checker:
        execute("producer" if args.producer else "checker")
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
