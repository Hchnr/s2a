"""Run and independently replay two windows of correct-reference feedback."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.simulation.runtime import prepare_ordered_correct_stream as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.verification.reference_ports.check_defer_close_release import independent_proposal, verify_proposal

REPO = prefix.REPO
ROOT = REPO / "outputs/ordered_correct_stream_feedback_20261111"
DRY = REPO / "outputs/ordered_correct_stream_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/ordered_correct_stream_feedback_20261111/result.json"
BRANCHES = ((8, "stream"),)
FORBIDDEN = common.FORBIDDEN_CHECKER_MODULES + (
    "s2a.research.gripper.defer_close_release",
    "s2a.research.gripper.defer_close_transaction",
    "s2a.research.gripper.ordered_correct_history",
    "s2a.research.joint_control.preview.ordered_correct_arm_history",
    "s2a.research.joint_control.preview.streaming_feedback_boundary",
    "s2a.research.response_models.causal.segmented_inner_error",
)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "src/s2a/research/native_runtime/development_continuation.py",
            REPO / "src/s2a/research/native_runtime/guarded_correct_prefix.py",
            REPO / "src/s2a/research/joint_control/preview/loaded_gripper_feedback.py",
            REPO / "src/s2a/research/joint_control/preview/refined_feedback.py",
            REPO / "src/s2a/research/joint_control/preview/quality_feedback_refiner.py",
            REPO / "src/s2a/research/joint_control/preview/completed_history.py",
            REPO / "src/s2a/research/gripper/loaded_gripper_transaction.py",
            REPO / "tests/research/native_runtime/test_ordered_correct_stream_inputs.py",
        )
    ]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered correct-reference branch required")
    return f"case_{index:02d}_{mode}"


def checker_isolation():
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Checker must not import candidate policies")


def reconstruct_gripper(inputs, result):
    from s2a.research.gripper.defer_close_transaction import DeferringLoadedGripperTransaction
    from s2a.research.gripper.ordered_correct_history import reconstruct_ordered_gripper

    old = inputs["previous"]
    controller, history = reconstruct_ordered_gripper(
        inputs["rows"],
        inputs["prefix_chunks"],
        old["tool"],
        inputs["parameters"],
        velocity_envelope=inputs["velocity_envelope"],
        gripper_config=old["controller"]["gripper_config"],
        progress=lambda value: result.update(gripper_reconstruction_progress=value),
    )
    audited = prefix.read(inputs["refs"]["gripper_rebuild"])["history"]
    prefix.require(
        history == audited and history["last_completed_command"] == float(old["commands"][-1, 6]),
        "Exact completed gripper history required",
    )
    result["gripper_history"] = history
    return DeferringLoadedGripperTransaction(controller)


def make_initial_policy(env, inputs, gripper, runtime_receipt):
    from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
    from s2a.research.joint_control.preview.ordered_correct_arm_history import restore_ordered_arm_history
    from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
    from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
    from s2a.research.response_models.causal.segmented_inner_error import SegmentedInnerHistory

    old = inputs["previous"]
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, velocity, site = current_robot_dynamics(env, jacobian)
    twist = jacobian @ velocity
    np.testing.assert_allclose(site, inputs["preaction_site"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(twist, inputs["preaction_twist"], atol=1e-10, rtol=0)
    owner, segment_receipt = SegmentedInnerHistory(), {}
    inner = owner.begin_segment(80)
    quality = QualityFeedbackRefiner(inner, segment_receipt, start_step=80, start_cursor=0)
    coupled, tracking, difference = restore_ordered_arm_history(
        inputs["rows"][-1], site, twist, old["controller"]["contract"], inputs["tracking_before"], refiner=quality
    )
    runtime_receipt["initial_innovation_recompute_max_difference"] = difference
    runtime_receipt["segment_receipts"] = [segment_receipt]
    runtime_receipt["initial_inner_error_history_available"] = inner.snapshot()["available"]
    site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
    anchor, targets = np.asarray(inputs["first_anchor"]), inputs["targets"][0]
    np.testing.assert_allclose(anchor, site @ site_from_tcp, atol=1e-9, rtol=0)
    arm = RefinedPreviewFeedback(
        env,
        coupled,
        tracking,
        targets @ inverse(site_from_tcp),
        anchor @ inverse(site_from_tcp),
        site_from_tcp,
        0,
        old["commands"][-1],
        old["controller"]["contract"],
        segment_receipt,
    )
    policy = LoadedGripperFeedback(
        env,
        arm,
        gripper,
        inner,
        targets,
        inputs["openings"][0],
        0,
        site_from_tcp,
        80,
        segment_receipt,
        initial_inner_loss=runtime_receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
        max_steps=8,
    )
    return policy, owner


def check_proposal(env, inputs, period, offset):
    checker_isolation()
    old = inputs["previous"]
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
    grip = common.measured_gripper(env)
    detail = period["decision"]["gripper_decision"]
    baseline, candidate = detail["baseline"], detail["deferral"]
    window, cursor = divmod(offset, 8)
    checked = independent_proposal(
        inputs["targets"][window],
        inputs["openings"][window],
        cursor,
        site @ site_from_tcp,
        grip["opening"],
        grip["motor_target"],
        baseline["command"],
        baseline["latch"]["closing_episode"],
    )
    difference = verify_proposal(candidate, checked)
    prefix.require(
        checked["command"] == detail["command"] == float(period["command"][6]),
        "Actual dispatched deferral command required",
    )
    return difference


def score_opening(env, inputs, offset):
    measured = common.measured_gripper(env)
    window, cursor = divmod(offset, 8)
    target = float(inputs["openings"][window][cursor])
    return {"gripper": measured, "opening_target": target, "opening_error": measured["opening"] - target}


def compare_score(actual, expected):
    for key in ("opening_target", "opening_error"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=0)
    for key in ("opening", "qpos", "qvel", "motor_target"):
        np.testing.assert_allclose(actual["gripper"][key], expected["gripper"][key], atol=1e-10, rtol=0)


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        registration["inputs"] == binding.input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Frozen correct-reference inputs and source identity required",
    )
    old, expected, original = inputs["previous"], None, None
    if phase == "checker":
        checker_isolation()
        original = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch required")
        expected = original["runtime"]["continuation"]["periods"]
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        gripper = None if phase == "checker" else reconstruct_gripper(inputs, result)
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.guarded_correct_prefix import GuardedCorrectTask2Seed42PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            policy = owner = None
            if phase == "checker":
                checker_isolation()
                actual_prefix, recorded_prefix = (
                    receipt["prefix_inner_periods"],
                    original["runtime"]["prefix_inner_periods"],
                )
                prefix.require(len(actual_prefix) == len(recorded_prefix) == 80, "Complete checked prefix required")
                for actual, recorded in zip(actual_prefix, recorded_prefix, strict=True):
                    common.check_prefix_inner(actual, recorded)
                receipt["independent_deferral_max_difference"] = 0.0
            else:
                policy, owner = make_initial_policy(env, inputs, gripper, receipt)

            def choose(q, limits, jacobian, previous, offset):
                nonlocal policy
                if phase == "checker":
                    delta = check_proposal(env, inputs, expected[offset], offset)
                    receipt["independent_deferral_max_difference"] = max(
                        receipt["independent_deferral_max_difference"], delta
                    )
                    return {"command": np.asarray(expected[offset]["command"]), "newly_selected": False}
                if offset == 8:
                    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                    if policy.last_inner_loss or margin < 0.05:
                        return {"stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin}}
                    from s2a.research.joint_control.preview.streaming_feedback_boundary import advance_feedback

                    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
                    anchor = site @ policy.site_from_tcp
                    segment_receipt = {}
                    policy = advance_feedback(
                        policy,
                        owner,
                        inputs["targets"][1],
                        inputs["openings"][1],
                        anchor,
                        generated_step=88,
                        receipt=segment_receipt,
                    )
                    receipt["segment_receipts"].append(segment_receipt)
                return policy.select(q, limits, jacobian, previous, offset % 8)

            def completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                history = (
                    None
                    if policy is None
                    else policy.complete(
                        actual_env, command, offset % 8, [item["q"] for item in period["actual_inner"]]
                    )
                )
                score = score_opening(actual_env, inputs, offset)
                if expected is not None:
                    compare_score(score, expected[offset]["completed_feedback"]["score"])
                return {"score": score, "policy_history": history, "policy_history_recomputed": policy is not None}

            fixed_targets = np.concatenate((inputs["targets"][0][:8], inputs["targets"][1][:8]))
            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=80,
                previous_command=old["commands"][-1],
                targets=fixed_targets,
                tool=old["tool"],
                target_object=inputs["config"]["target_object"],
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
                continuation_limit=16,
            )
            if phase == "checker":
                q, limits, _jac = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                return common.reconcile_stop(
                    outcome,
                    original["runtime"]["continuation"]["stop"],
                    start_step=80,
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                    horizon=16,
                )
            return outcome

        try:
            runtime = GuardedCorrectTask2Seed42PrefixRuntime().run(
                old["runtime_config"],
                old["record"],
                old["record_xml"],
                old["commands"],
                old["saved"],
                old["receipts"],
                directory / "runtime",
                old["asset_refs"],
                old["controller"],
                continuation=continuation,
                continuation_budget=16,
                continuation_limit=16,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            checker_isolation()
        result.update(
            complete=True,
            passed=True,
            runtime=runtime,
            candidate_modules_unloaded=all(name not in sys.modules for name in FORBIDDEN),
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_phase(branches, phase, config):
    prefix.require(
        len(branches) == 1 and branches[0]["complete"] is branches[0]["passed"] is True, "One closed branch required"
    )
    branch = branches[0]
    runtime, periods = branch["runtime"], branch["runtime"]["continuation"]["periods"]
    segments = runtime.get("segment_receipts", [])
    counts = {
        "control_steps": runtime["control_steps_completed"],
        "native_steps": runtime["counters"]["mj_step"]["completed"],
        "newly_selected_commands": runtime["newly_selected_commands"],
        "continuation_steps": len(periods),
        "prefix_inner_samples": sum(len(p["samples"]) for p in runtime["prefix_inner_periods"]),
        "quality_calls": sum(s["quality_refiner"]["quality_calls_attempted"] for s in segments),
        "quality_responses": sum(s["quality_refiner"]["quality_response_evaluations"] for s in segments),
        "baseline_responses": sum(s["quality_refiner"]["baseline_response_evaluations"] for s in segments),
        "gripper_sequences": sum(s["gripper_proposals"]["candidate_sequences"] for s in segments),
        "segment_count": len(segments),
    }
    prefix.require(
        counts["control_steps"] <= config["max_control_steps_per_phase"]
        and counts["native_steps"] <= config["max_native_steps_per_phase"]
        and counts["prefix_inner_samples"] == 80 * 25
        and runtime["checked_endpoints"] == 81
        and runtime["cleanup_complete"] is True
        and counts["continuation_steps"] <= 16
        and counts["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and counts["quality_responses"] <= 16 * config["max_quality_responses_per_period"]
        and counts["baseline_responses"] <= 16 * config["max_baseline_responses_per_period"]
        and counts["gripper_sequences"] <= 16 * config["max_gripper_sequences_per_period"]
        and (phase != "checker" or branch["candidate_modules_unloaded"]),
        "Bounded producer or isolated actual-command replay required",
    )
    return counts


def dry_run():
    inputs = binding.load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": binding.input_receipt(inputs),
                "source_refs": source_refs(),
                "world_model_calls": 0,
                "physics_steps": 0,
                "fresh_cases": 0,
            },
        )
    )


def execute(phase):
    return registered_feedback_phase.execute_phase(
        phase,
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.run_ordered_correct_stream",
        branch_key=branch_key,
        load_inputs=binding.load_inputs,
        input_receipt=binding.input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="config",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.producer:
        execute("producer")
    elif args.checker:
        execute("checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "Known feedback phase required")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
