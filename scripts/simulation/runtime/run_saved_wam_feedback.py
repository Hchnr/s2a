"""Execute and independently replay one retained WAM reference on a verified prefix."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.simulation.runtime import capture_ordered_feedback_observations as capture_runner
from scripts.simulation.runtime import prepare_ordered_correct_stream as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_ordered_correct_stream as prior
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/saved_wam_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/saved_wam_feedback_20261111.md"
ROOT = REPO / "outputs/saved_wam_feedback_20261111"
DRY = REPO / "outputs/saved_wam_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/saved_wam_feedback_20261111/result.json"
BRANCHES = ((8, "saved_wam"),)


def source_refs():
    return prior.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            Path(capture_runner.__file__),
            Path(repro.__file__),
            REPO / "src/s2a/references/action_reference.py",
            REPO / "src/s2a/research/wam/training/native_observations.py",
            REPO / "tests/research/native_runtime/test_saved_wam_feedback.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = {
        "schema": "s2a.saved-wam-feedback.v1",
        "output": "outputs/saved_wam_feedback_20261111",
        "case_index": 8,
        "seed": 42,
        "prefix_steps": 80,
        "continuation_steps": 8,
        "max_control_steps_per_phase": 88,
        "max_native_steps_per_phase": 2200,
        "max_quality_responses_per_period": 160,
        "max_baseline_responses_per_period": 480,
        "max_gripper_sequences_per_period": 6561,
        "worker_timeout_seconds": 1800,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "fits": 0,
        "automatic_retry": False,
    }
    prefix.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items()),
        "Fixed one-window saved-WAM feedback budget required",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    base = binding.load_inputs()
    refs = {
        key: binding.configured_ref(config, key)
        for key in ("response", "partial_audit", "capture_branch", "correct_producer")
    }
    response, audit, capture_branch, correct = (
        prefix.read(refs[key]) for key in ("response", "partial_audit", "capture_branch", "correct_producer")
    )
    partial = audit["rows"][0]
    source_capture = capture_branch["runtime"]["captures"][0]
    prefix.require(
        audit["complete"] is True
        and audit["original_worker_success"] is False
        and audit["mask_initial_noise_ledger_available"] is False
        and audit["new_model_requests"] == audit["new_physics_steps"] == 0
        and partial["step"] == 80
        and partial["response_sha256"] == refs["response"]["sha256"]
        and capture_branch["complete"] is capture_branch["passed"] is correct["complete"] is correct["passed"] is True
        and source_capture["frame_index"] == 80
        and source_capture["render_before"] == source_capture["render_after"]
        and response["args"]["vision_path"] == source_capture["canvas_png"]["path"]
        and repro.sha256(source_capture["canvas_png"]["path"]) == source_capture["canvas_png"]["sha256"]
        and base["previous"]["record"]["task"] == response["args"]["prompt"] == "put the cream cheese in the bowl",
        "Audited partial WAM response and its exact current image required",
    )
    action = repro.response_arrays(response)["action"]
    anchor = np.asarray(source_capture["current_source_pose"], dtype=float) @ base["previous"]["tool"]
    np.testing.assert_allclose(anchor, base["first_anchor"], atol=1e-10, rtol=0)
    targets, openings = decode_action_targets(anchor, action, horizon=40)
    consumed = dict(base["source_sha256"])
    for ref in (config_ref, *refs.values(), source_capture["canvas_png"]):
        prefix.checked(ref, consumed)
    return {
        "config_ref": config_ref,
        "config": config,
        "refs": refs,
        "base": base,
        "source_capture": source_capture,
        "correct": correct,
        "anchor": anchor,
        "targets": targets,
        "openings": openings,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "base": binding.input_receipt(inputs["base"]),
        "source_sha256": inputs["source_sha256"],
    }


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One saved-WAM branch required")
    return f"case_{index:02d}_{mode}"


def validate_live_capture(expected, actual):
    prefix.require(
        expected["frame_index"] == actual["frame_index"] == 80
        and expected["current_endpoint"]["sim_state_sha256"] == actual["current_endpoint"]["sim_state_sha256"]
        and expected["current_endpoint"]["sim_time"] == actual["current_endpoint"]["sim_time"]
        and expected["canvas_rgb_sha256"] == actual["canvas_rgb_sha256"]
        and expected["canvas_png"]["sha256"] == actual["canvas_png"]["sha256"]
        and actual["render_before"] == actual["render_after"]
        and actual["future_rgb_frames"] == 0,
        "Current state, time and rendered WAM input must exactly match the saved request",
    )
    np.testing.assert_allclose(actual["current_source_pose"], expected["current_source_pose"], atol=1e-10, rtol=0)


def model_policy_inputs(inputs, anchor):
    base = dict(inputs["base"])
    base.update(
        first_anchor=anchor,
        targets=[inputs["targets"], inputs["targets"]],
        openings=[inputs["openings"], inputs["openings"]],
    )
    return base


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Frozen saved-WAM inputs and sources required",
    )
    base, old = inputs["base"], inputs["base"]["previous"]
    original = expected = None
    if phase == "checker":
        prior.checker_isolation()
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
        gripper = None if phase == "checker" else prior.reconstruct_gripper(base, result)
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.guarded_correct_prefix import GuardedCorrectTask2Seed42PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            actual_prefix, correct_prefix = (
                receipt["prefix_inner_periods"],
                inputs["correct"]["runtime"]["prefix_inner_periods"],
            )
            prefix.require(len(actual_prefix) == len(correct_prefix) == 80, "All original prefix periods required")
            for actual, earlier in zip(actual_prefix, correct_prefix, strict=True):
                common.check_prefix_inner(actual, earlier)
            live = capture_runner.render_registered_endpoint(
                observers, env, 80, terminal["sample"], terminal["endpoint_receipt"], directory / "runtime"
            )
            validate_live_capture(inputs["source_capture"], live)
            receipt["current_wam_image"] = live
            anchor = np.asarray(terminal["sample"]["source_pose"]) @ old["tool"]
            np.testing.assert_allclose(anchor, inputs["anchor"], atol=1e-10, rtol=0)
            policy_inputs = model_policy_inputs(inputs, anchor)
            policy = None
            if phase == "checker":
                prior.checker_isolation()
                receipt["independent_deferral_max_difference"] = 0.0
            else:
                policy, _owner = prior.make_initial_policy(env, policy_inputs, gripper, receipt)

            def choose(q, limits, jacobian, previous, offset):
                if phase == "checker":
                    delta = prior.check_proposal(env, policy_inputs, expected[offset], offset)
                    receipt["independent_deferral_max_difference"] = max(
                        receipt["independent_deferral_max_difference"], delta
                    )
                    return {"command": np.asarray(expected[offset]["command"]), "newly_selected": False}
                return policy.select(q, limits, jacobian, previous, offset)

            def completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                history = (
                    None
                    if policy is None
                    else policy.complete(actual_env, command, offset, [item["q"] for item in period["actual_inner"]])
                )
                score = prior.score_opening(actual_env, policy_inputs, offset)
                if expected is not None:
                    prior.compare_score(score, expected[offset]["completed_feedback"]["score"])
                return {"score": score, "policy_history": history, "policy_history_recomputed": policy is not None}

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=80,
                previous_command=old["commands"][-1],
                targets=inputs["targets"][:8],
                tool=old["tool"],
                target_object=base["config"]["target_object"],
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
                continuation_limit=8,
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
                    horizon=8,
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
                continuation_budget=8,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            prior.checker_isolation()
        result.update(
            complete=True,
            passed=True,
            runtime=runtime,
            candidate_modules_unloaded=all(name not in sys.modules for name in prior.FORBIDDEN),
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_phase(branches, phase, config):
    prefix.require(
        len(branches) == 1 and branches[0]["complete"] is branches[0]["passed"] is True,
        "One closed feedback branch required",
    )
    branch, runtime = branches[0], branches[0]["runtime"]
    periods = runtime["continuation"]["periods"]
    segments = runtime.get("segment_receipts", [])
    counts = {
        "control_steps": runtime["control_steps_completed"],
        "native_steps": runtime["counters"]["mj_step"]["completed"],
        "continuation_steps": len(periods),
        "newly_selected_commands": runtime["newly_selected_commands"],
        "prefix_inner_samples": sum(len(period["samples"]) for period in runtime["prefix_inner_periods"]),
        "quality_responses": sum(segment["quality_refiner"]["quality_response_evaluations"] for segment in segments),
        "baseline_responses": sum(segment["quality_refiner"]["baseline_response_evaluations"] for segment in segments),
        "gripper_sequences": sum(segment["gripper_proposals"]["candidate_sequences"] for segment in segments),
    }
    prefix.require(
        counts["control_steps"] == 80 + len(periods)
        and counts["native_steps"] == 2000 + 25 * len(periods)
        and counts["control_steps"] <= config["max_control_steps_per_phase"]
        and counts["native_steps"] <= config["max_native_steps_per_phase"]
        and counts["prefix_inner_samples"] == 2000
        and runtime["checked_endpoints"] == 81
        and runtime["cleanup_complete"] is True
        and len(periods) <= 8
        and counts["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and counts["quality_responses"] <= 8 * config["max_quality_responses_per_period"]
        and counts["baseline_responses"] <= 8 * config["max_baseline_responses_per_period"]
        and counts["gripper_sequences"] <= 8 * config["max_gripper_sequences_per_period"]
        and (phase != "checker" or branch["candidate_modules_unloaded"]),
        "Bounded actual feedback or isolated replay required",
    )
    return counts


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": input_receipt(inputs),
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
        worker_module="scripts.simulation.runtime.run_saved_wam_feedback",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
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
