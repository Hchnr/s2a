"""Execute two causally linked WAM windows and independently replay actual actions."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    run_continuation,
)
from scripts.simulation.runtime import capture_ordered_feedback_observations as capture
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_ordered_correct_stream as prior
from scripts.simulation.runtime import run_saved_wam_feedback as saved
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_wam_two_window_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_wam_two_window_feedback_20261111.md"
ROOT = REPO / "outputs/causal_wam_two_window_feedback_20261111"
DRY = REPO / "outputs/causal_wam_two_window_feedback_dry_20261111/result.json"
QA = REPO / "outputs/qa/causal_wam_two_window_feedback_20261111/result.json"
BRANCHES = ((8, "causal_wam"),)


def source_refs():
    return saved.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "src/s2a/research/joint_control/preview/streaming_feedback_boundary.py",
            REPO / "tests/research/native_runtime/test_causal_wam_two_window_feedback.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = {
        "schema": "s2a.causal-wam-two-window-feedback.v1",
        "output": str(ROOT.relative_to(REPO)),
        "case_index": 8,
        "seed": 42,
        "prefix_steps": 80,
        "continuation_steps": 16,
        "window_steps": 8,
        "max_control_steps_per_phase": 96,
        "max_native_steps_per_phase": 2400,
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
        "Fixed two-window causal WAM feedback budget required",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    old = saved.load_inputs()
    refs = {
        key: saved.binding.configured_ref(config, key)
        for key in (
            "model_audit",
            "model_result",
            "model_owner",
            "response80",
            "response88",
            "capture_branch",
            "previous_feedback",
        )
    }
    audit, model, owner, response80, response88, capture_branch, feedback = (
        prefix.read(refs[key])
        for key in (
            "model_audit",
            "model_result",
            "model_owner",
            "response80",
            "response88",
            "capture_branch",
            "previous_feedback",
        )
    )
    images = capture_branch["runtime"]["captures"]
    periods = feedback["runtime"]["continuation"]["periods"]
    prefix.require(
        audit["complete"] is audit["causal_first_reference_exact"] is True
        and [row["response_sha256"] for row in audit["rows"]]
        == [refs["response80"]["sha256"], refs["response88"]["sha256"]]
        and model["complete"] is True
        and model["model_initializations"] == 1
        and model["model_requests"] == 2
        and model["denoise_calls"] == 70
        and owner["complete"] is owner["actual_wait"] is owner["reaped"] is owner["released"] is True
        and owner["worker_exit_code"] == 0
        and owner["signal_sent"] is False
        and capture_branch["complete"] is capture_branch["passed"] is feedback["complete"] is feedback["passed"] is True
        and feedback["runtime"]["continuation"]["stop"] is None
        and len(periods) == 8
        and [row["step"] for row in periods] == list(range(80, 88))
        and len(images) == 2
        and [item["frame_index"] for item in images] == [80, 88]
        and images[1]["current_endpoint"]["sim_state_sha256"] == periods[-1]["endpoint_receipt"]["sim_state_sha256"]
        and images[1]["current_endpoint"]["sim_time"] == periods[-1]["endpoint_receipt"]["sim_time"],
        "Independently audited causal references and their actual feedback endpoints required",
    )
    original = prefix.read(old["refs"]["response"])
    comparison = repro.compare_responses(response80, original)
    prefix.require(
        all(row["elementwise_equal"] for row in comparison.values()),
        "First reference must equal the reference that actually drove the previous feedback",
    )
    tool = old["base"]["previous"]["tool"]
    anchors = [np.asarray(image["current_source_pose"], dtype=float) @ tool for image in images]
    np.testing.assert_allclose(anchors[0], old["anchor"], atol=1e-10, rtol=0)
    targets, openings = [], []
    for anchor, response in zip(anchors, (response80, response88), strict=True):
        pose, opening = decode_action_targets(anchor, repro.response_arrays(response)["action"], horizon=40)
        targets.append(pose)
        openings.append(opening)
    np.testing.assert_allclose(targets[0], old["targets"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(openings[0], old["openings"])
    consumed = dict(old["source_sha256"])
    for ref in (config_ref, *refs.values(), *(image["canvas_png"] for image in images)):
        prefix.checked(ref, consumed)
    return {
        "config_ref": config_ref,
        "config": config,
        "refs": refs,
        "old": old,
        "images": images,
        "periods": periods,
        "anchors": anchors,
        "targets": targets,
        "openings": openings,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "old": saved.input_receipt(inputs["old"]),
        "source_sha256": inputs["source_sha256"],
    }


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One causal WAM feedback branch required")
    return f"case_{index:02d}_{mode}"


def validate_live_capture(expected, actual):
    prefix.require(
        expected["frame_index"] == actual["frame_index"]
        and expected["current_endpoint"]["sim_state_sha256"] == actual["current_endpoint"]["sim_state_sha256"]
        and expected["current_endpoint"]["sim_time"] == actual["current_endpoint"]["sim_time"]
        and expected["canvas_rgb_sha256"] == actual["canvas_rgb_sha256"]
        and expected["canvas_png"]["sha256"] == actual["canvas_png"]["sha256"]
        and actual["render_before"] == actual["render_after"]
        and actual["future_rgb_frames"] == 0,
        "The live state, time and image must match the corresponding WAM request",
    )
    np.testing.assert_allclose(actual["current_source_pose"], expected["current_source_pose"], atol=1e-10, rtol=0)


def render_and_verify(observers, env, step, sample, endpoint, runtime_output, expected):
    live = capture.render_registered_endpoint(observers, env, step, sample, endpoint, runtime_output)
    validate_live_capture(expected, live)
    return live


def require_previous_action(decision, expected):
    prefix.require(
        "command" in decision and "stop" not in decision and decision.get("newly_selected") is True,
        "The first window must reproduce its previously selected action",
    )
    np.testing.assert_allclose(decision["command"], expected["command"], atol=1e-10, rtol=0)


def compare_first_period(actual, expected):
    for key in ("command", "actual_state", "tcp_error"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=0)
    prefix.require(len(actual["actual_inner"]) == len(expected["actual_inner"]) == 25, "All inner samples required")
    for key in ("q", "qvel", "time"):
        np.testing.assert_allclose(
            [item[key] for item in actual["actual_inner"]],
            [item[key] for item in expected["actual_inner"]],
            atol=1e-10,
            rtol=0,
        )
    for key in (
        "inner_outside_samples",
        "inner_below_buffer_samples",
        "pose_caps_satisfied",
        "task_success",
        "target_grasp",
    ):
        prefix.require(actual[key] == expected[key], f"Previously executed WAM feedback differs: {key}")


def policy_inputs(inputs, anchor):
    base = dict(inputs["old"]["base"])
    base.update(first_anchor=anchor, targets=inputs["targets"], openings=inputs["openings"])
    return base


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Frozen causal WAM inputs and sources required",
    )
    base, old = inputs["old"]["base"], inputs["old"]["base"]["previous"]
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
            previous_prefix = inputs["old"]["correct"]["runtime"]["prefix_inner_periods"]
            prefix.require(
                len(receipt["prefix_inner_periods"]) == len(previous_prefix) == 80,
                "Every historical prefix period required",
            )
            for actual, earlier in zip(receipt["prefix_inner_periods"], previous_prefix, strict=True):
                common.check_prefix_inner(actual, earlier)
            images = receipt["current_wam_images"] = []
            first = render_and_verify(
                observers,
                env,
                80,
                terminal["sample"],
                terminal["endpoint_receipt"],
                directory / "runtime",
                inputs["images"][0],
            )
            images.append(first)
            anchor = np.asarray(terminal["sample"]["source_pose"]) @ old["tool"]
            np.testing.assert_allclose(anchor, inputs["anchors"][0], atol=1e-10, rtol=0)
            policy_data = policy_inputs(inputs, anchor)
            policy = owner = None
            if phase == "checker":
                prior.checker_isolation()
                receipt["independent_deferral_max_difference"] = 0.0
            else:
                policy, owner = prior.make_initial_policy(env, policy_data, gripper, receipt)

            def capture_second():
                sample, endpoint = observers.endpoint(env, 88)
                live = render_and_verify(
                    observers, env, 88, sample, endpoint, directory / "runtime", inputs["images"][1]
                )
                current_anchor = np.asarray(sample["source_pose"]) @ old["tool"]
                np.testing.assert_allclose(current_anchor, inputs["anchors"][1], atol=1e-10, rtol=0)
                images.append(live)
                return current_anchor

            def choose(q, limits, jacobian, previous, offset):
                nonlocal policy
                if offset == 8:
                    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                    if phase != "checker" and (policy.last_inner_loss or margin < 0.05):
                        return {"stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin}}
                    current_anchor = capture_second()
                    if phase != "checker":
                        from s2a.research.joint_control.preview.streaming_feedback_boundary import advance_feedback

                        segment_receipt = {}
                        policy = advance_feedback(
                            policy,
                            owner,
                            inputs["targets"][1],
                            inputs["openings"][1],
                            current_anchor,
                            generated_step=88,
                            receipt=segment_receipt,
                        )
                        receipt["segment_receipts"].append(segment_receipt)
                if phase == "checker":
                    difference = prior.check_proposal(env, policy_data, expected[offset], offset)
                    receipt["independent_deferral_max_difference"] = max(
                        receipt["independent_deferral_max_difference"], difference
                    )
                    return {"command": np.asarray(expected[offset]["command"]), "newly_selected": False}
                decision = policy.select(q, limits, jacobian, previous, offset % 8)
                if offset < 8:
                    require_previous_action(decision, inputs["periods"][offset])
                return decision

            def completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                if offset < 8:
                    compare_first_period(period, inputs["periods"][offset])
                history = (
                    None
                    if policy is None
                    else policy.complete(
                        actual_env, command, offset % 8, [item["q"] for item in period["actual_inner"]]
                    )
                )
                score = prior.score_opening(actual_env, policy_data, offset)
                if expected is not None:
                    prior.compare_score(score, expected[offset]["completed_feedback"]["score"])
                if offset < 8:
                    prior.compare_score(score, inputs["periods"][offset]["completed_feedback"]["score"])
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
                target_object=base["config"]["target_object"],
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
                continuation_limit=16,
            )
            if phase == "checker":
                if len(expected) == 8 and len(original["runtime"]["current_wam_images"]) == 2:
                    capture_second()
                q, limits, _jacobian = current_joint_site_state(env)
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
        "One closed causal WAM feedback branch required",
    )
    branch, runtime = branches[0], branches[0]["runtime"]
    periods, segments = runtime["continuation"]["periods"], runtime.get("segment_receipts", [])
    counts = {
        "control_steps": runtime["control_steps_completed"],
        "native_steps": runtime["counters"]["mj_step"]["completed"],
        "continuation_steps": len(periods),
        "newly_selected_commands": runtime["newly_selected_commands"],
        "prefix_inner_samples": sum(len(period["samples"]) for period in runtime["prefix_inner_periods"]),
        "quality_responses": sum(segment["quality_refiner"]["quality_response_evaluations"] for segment in segments),
        "baseline_responses": sum(segment["quality_refiner"]["baseline_response_evaluations"] for segment in segments),
        "gripper_sequences": sum(segment["gripper_proposals"]["candidate_sequences"] for segment in segments),
        "captured_steps": [item["frame_index"] for item in runtime["current_wam_images"]],
    }
    prefix.require(
        counts["control_steps"] == 80 + len(periods)
        and counts["native_steps"] == 2000 + 25 * len(periods)
        and counts["control_steps"] <= config["max_control_steps_per_phase"]
        and counts["native_steps"] <= config["max_native_steps_per_phase"]
        and counts["prefix_inner_samples"] == 2000
        and runtime["checked_endpoints"] == 81
        and runtime["cleanup_complete"] is True
        and len(periods) <= 16
        and counts["newly_selected_commands"] == (len(periods) if phase == "producer" else 0)
        and counts["quality_responses"] <= 16 * config["max_quality_responses_per_period"]
        and counts["baseline_responses"] <= 16 * config["max_baseline_responses_per_period"]
        and counts["gripper_sequences"] <= 16 * config["max_gripper_sequences_per_period"]
        and counts["captured_steps"] in ([80], [80, 88])
        and (len(periods) <= 8 or counts["captured_steps"] == [80, 88])
        and (phase != "checker" or branch["candidate_modules_unloaded"]),
        "Bounded causal WAM feedback or isolated replay required",
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
        worker_module="scripts.simulation.runtime.run_causal_wam_two_window_feedback",
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
