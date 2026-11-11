"""Execute and independently replay the registered bounded release hold."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.simulation.runtime import prepare_bounded_release_hold as binding
from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as prior_binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as base
from scripts.simulation.runtime import run_causal_event_long_feedback as long
from scripts.simulation.runtime import run_causal_event_switch_feedback as switch
from scripts.wam.tracking.preview import bind_bounded_release_shadow as shadow

REPO = prefix.REPO
ROOT = REPO / "outputs/bounded_release_hold_feedback_20261111"
QA = REPO / "outputs/qa/bounded_release_hold_feedback_20261111/result.json"
DRY = REPO / "outputs/bounded_release_hold_feedback_dry_20261111/result.json"
BRANCHES = ((5, "bounded_release_hold"),)
CHECKER_FORBIDDEN = base.FORBIDDEN_CHECKER_MODULES + (
    "s2a.research.wam.tracking.preview.bounded_release_revision",
    "scripts.wam.tracking.preview.replay_bounded_release_revision",
)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "tests/research/native_runtime/test_bounded_release_hold_feedback.py",
        )
    ]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "The single registered physical branch")
    return f"case_{index:02d}_{mode}"


def require_checker_independence():
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker must not import candidate policies"
    )


def start_hold(policy, selector, old, latest, config, receipt, *, measured_tcp, measured_opening, limits):
    """Use just-completed state to prepare a causal reference selection."""
    from s2a.research.wam.tracking.preview.bounded_release_revision import ReferenceChunk

    periods = receipt["continuation_partial"]["periods"]
    prefix.require(
        len(periods) == config["activation_offset"] and periods[-1]["step"] == 367, "Completed activation boundary"
    )
    samples, limits = periods[-1]["actual_inner"], np.asarray(limits, dtype=float)
    prefix.require(len(samples) == 25 and limits.shape == (7, 2), "Actual preceding inner period and limits")
    q = np.asarray([item["q"] for item in samples], dtype=float)
    prefix.require(q.shape == (25, 7) and np.isfinite(q).all(), "Finite completed inner samples")
    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    np.testing.assert_allclose(margin, periods[-1]["minimum_inner_margin_rad"], atol=1e-10, rtol=0)
    prefix.require(
        not policy.last_inner_loss and margin >= config["joint_buffer_rad"],
        "Observed buffer loss stops before selection",
    )
    latch = policy.gripper.state()
    prefix.require(
        not latch["staged"] and not latch["faulted"] and type(latch["latch"]["closing"]) is bool,
        "Completed gripper state",
    )
    decision = selector.prepare(
        ReferenceChunk(config["latest_step"], latest["targets"], latest["openings"]),
        committed=ReferenceChunk(config["old_step"], old["targets"], old["openings"]),
        step=config["prefix_steps"] + config["activation_offset"],
        measured_tcp=measured_tcp,
        measured_opening=measured_opening,
        closing_episode=latch["latch"]["closing"],
        previous_inner_margin_rad=margin,
    )
    receipt["release_selection"] = {
        "decision": decision,
        "previous_inner_margin_rad": margin,
        "measured_opening": float(measured_opening),
        "closing_episode": latch["latch"]["closing"],
        "completed_hold_steps": 0,
        "bridge_executed": False,
    }
    if decision["source"] != "committed" or decision["reason"] != "bounded_hold":
        selector.abort(decision)
        receipt["release_selection"]["state_after_abort"] = selector.state()
        return {"stop": {"reason": "release_guard_not_eligible", "selection_reason": decision["reason"]}}
    prefix.require(
        decision["poses"].shape == (32, 4, 4) and decision["openings"].shape == (32,), "Complete selected preview"
    )
    np.testing.assert_array_equal(decision["poses"][:8], old["fixed_targets"][8:16])
    targets, openings = latest["targets"].copy(), latest["openings"].copy()
    targets[:8], openings[:8] = old["fixed_targets"][:8], old["fixed_openings"][:8]
    targets[8:], openings[8:] = decision["poses"], decision["openings"]
    try:
        switch.switch_reference(policy, {"targets": targets, "openings": openings, "anchor": latest["anchor"]}, receipt)
    except BaseException:
        selector.abort(decision)
        receipt["release_selection"]["state_after_abort"] = selector.state()
        raise
    return None


def finish_selection(selector, receipt):
    """Commit only a full hold; isolate incomplete or uncertain execution."""
    selection = receipt.get("release_selection")
    if selection is None or not selector.state()["staged"]:
        return
    periods = receipt.get("continuation_partial", {}).get("periods", [])
    completed = max(0, len(periods) - 8)
    prefix.require(completed <= 8, "Only the registered hold may complete")
    selection["completed_hold_steps"] = completed
    uncertain = receipt["control_steps_attempted"] != receipt["control_steps_completed"]
    selection["partial_physical_execution_uncertain"] = uncertain
    if uncertain:
        # Preserve the staged token: reusing this instance requires reconstruction.
        selection["state_after_execution"] = selector.state()
        selection["reconstruction_required"] = True
    elif completed == 8:
        selector.commit(selection["decision"], completed_steps=8)
        selection["state_after_execution"] = selector.state()
    else:
        selector.abort(selection["decision"], completed_steps=completed)
        selection["state_after_execution"] = selector.state()


def reconcile_checker(outcome, expected_stop, *, current_margin, prefix_inner_loss):
    if expected_stop is not None and expected_stop.get("reason") == "release_guard_not_eligible":
        prefix.require(
            outcome["stop"] is None
            and len(outcome["periods"]) == 8
            and expected_stop.get("at_step") == 368
            and expected_stop.get("next_action_dispatched") is False,
            "Recorded guard stop follows the eight replayed commands",
        )
        outcome["stop"] = expected_stop.copy()
        outcome["decision_stop_recorded_not_recomputed"] = True
        return outcome
    return base.reconcile_stop(
        outcome,
        expected_stop,
        start_step=360,
        current_margin=current_margin,
        prefix_inner_loss=prefix_inner_loss,
        horizon=16,
    )


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen pilot sources and inputs",
    )
    if phase == "checker":
        require_checker_independence()
    case, config = inputs["case"], inputs["config"]
    old, latest = prior_binding.selected_references(case)
    prior = prefix.read(inputs["prior_long_refs"]["producer"])
    prior_periods = prior["runtime"]["continuation"]["periods"]
    expected, expected_stop, producer = None, None, None
    if phase == "checker":
        producer = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(producer["complete"] is producer["passed"] is True, "Closed pilot producer")
        expected = producer["runtime"]["continuation"]["periods"]
        expected_stop = producer["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "case": case["spec"],
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        gripper = None if phase == "checker" else base.reconstruct_gripper(case, result)
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            base.check_prefix_inner(receipt["final_prefix_inner"], prior["runtime"]["final_prefix_inner"])
            start_position = long.target_position(env, case["spec"]["target"])
            np.testing.assert_allclose(start_position, prior["runtime"]["target_start_position"], atol=1e-10, rtol=0)
            receipt["target_start_position"], receipt["previous_verified_steps"] = start_position, 0
            policy, selector = None, None
            if phase == "checker":
                require_checker_independence()
                action = base.checker_action(expected)
            else:
                from s2a.research.wam.tracking.preview.bounded_release_revision import (
                    BoundedReleaseRevision,
                    ReleaseRevisionConfig,
                )

                revision = prefix.read(prefix.reference(shadow.CONFIG))
                selector = BoundedReleaseRevision(
                    ReleaseRevisionConfig(
                        **{
                            name: revision[name]
                            for name in (
                                "stride",
                                "bridge_steps",
                                "release_clearance",
                                "opening_revision_gap",
                                "current_position_gate_m",
                                "current_rotation_gate_rad",
                                "joint_buffer_rad",
                            )
                        }
                    )
                )
                policy = base.make_feedback(env, case, old, gripper, receipt, max_steps=16)

                def action(q, limits, jacobian, previous, offset):
                    if offset == 8 and not policy.last_inner_loss:
                        _mass, _velocity, site = current_robot_dynamics(env, jacobian)
                        measured = base.measured_gripper(env)
                        stop = start_hold(
                            policy,
                            selector,
                            old,
                            latest,
                            config,
                            receipt,
                            measured_tcp=site @ policy.site_from_tcp,
                            measured_opening=measured["opening"],
                            limits=limits,
                        )
                        if stop is not None:
                            return stop
                    return policy.select(q, limits, jacobian, previous, offset)

            def observed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == 360 + offset, "Just-completed physical period")
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [sample["q"] for sample in period["actual_inner"]]
                    )
                expected_score = None
                if expected is not None:
                    expected_score = expected[offset]["completed_feedback"]["scores"]
                elif offset < 8:
                    expected_score = prior_periods[offset]["completed_feedback"]["scores"]
                scores = base.score_completed(
                    actual_env, case, old, "remembered", offset, period["tcp_error"], expected=expected_score
                )
                if offset < 8:
                    long.check_previous_period(period, prior_periods[offset])
                    receipt["previous_verified_steps"] += 1
                position = long.target_position(actual_env, case["spec"]["target"])
                height = float(position[2] - start_position[2])
                if offset < 8:
                    np.testing.assert_allclose(
                        position, prior_periods[offset]["completed_feedback"]["object_position"], atol=1e-10, rtol=0
                    )
                if expected is not None:
                    checked = expected[offset]["completed_feedback"]
                    np.testing.assert_allclose(position, checked["object_position"], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(height, checked["object_height_change_m"], atol=1e-10, rtol=0)
                return {
                    "scores": scores,
                    "object_position": position,
                    "object_height_change_m": height,
                    "policy_history": history,
                    "policy_history_recomputed": policy is not None,
                }

            try:
                outcome = run_continuation(
                    env,
                    terminal,
                    observers,
                    receipt,
                    start_step=360,
                    previous_command=case["commands"][-1],
                    targets=old["fixed_targets"],
                    tool=case["tool"],
                    target_object=case["spec"]["target"],
                    choose_action=action,
                    stop_on_outside=True,
                    expected_periods=expected,
                    on_completed=observed,
                    progress=progress,
                    continuation_limit=16,
                )
            finally:
                if selector is not None:
                    finish_selection(selector, receipt)
            prefix.require(
                receipt["previous_verified_steps"] == 8, "The complete earlier eight-period outcome must reproduce"
            )
            if phase == "checker":
                q, limits, _ = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                outcome = reconcile_checker(
                    outcome,
                    expected_stop,
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
                outcome["release_selection_not_recomputed"] = True
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 64 == 0 or completed >= 360:
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "control_completed": completed,
                            "new_commands": receipt.get("newly_selected_commands", 0),
                            "native_steps": receipt["counters"]["mj_step"],
                        }
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
        unloaded = all(name not in sys.modules for name in CHECKER_FORBIDDEN)
        if phase == "checker":
            prefix.require(
                unloaded and runtime["newly_selected_commands"] == 0, "Checker only replays recorded actions"
            )
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=unloaded)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_phase(branches, phase, config):
    prefix.require(len(branches) == 1, "One registered physical branch")
    item, runtime = branches[0], branches[0]["runtime"]
    prefix.require(
        item["complete"] is item["passed"] is True and item["control_qualified"] is False, "Closed pilot branch"
    )
    counters = runtime["counters"]
    counts = {
        "control_steps": runtime["control_steps_completed"],
        "native_steps": counters["mj_step"]["completed"],
        "model_initializations": counters["model_construct"]["completed"],
        "contexts": counters["context"]["completed"],
        "newly_selected_commands": runtime["newly_selected_commands"],
    }
    if phase == "producer":
        quality, grip, history = runtime["quality_refiner"], runtime["gripper_proposals"], item["gripper_history"]
        counts.update(
            refiner_calls=runtime["refiner_calls"]["attempted"],
            refiner_response_evaluations=runtime["refiner_response_evaluations"],
            quality_calls=quality["quality_calls_attempted"],
            quality_response_evaluations=quality["quality_response_evaluations"],
            gripper_planner_calls=history["planner_calls"] + grip["planner_calls"],
            gripper_candidate_sequences=history["candidate_sequences"] + grip["candidate_sequences"],
        )
    else:
        prefix.require(
            item["candidate_modules_unloaded"] is True and counts["newly_selected_commands"] == 0,
            "Independent action replay",
        )
    budgets = {
        "control_steps": "max_control_steps_per_phase",
        "native_steps": "max_native_steps_per_phase",
        "model_initializations": "max_models_per_phase",
        "contexts": "max_contexts_per_phase",
        "refiner_calls": "max_refiner_calls_per_phase",
        "refiner_response_evaluations": "max_refiner_response_evaluations_per_phase",
        "quality_calls": "max_quality_calls_per_phase",
        "quality_response_evaluations": "max_quality_response_evaluations_per_phase",
        "gripper_planner_calls": "max_gripper_planner_calls_per_phase",
        "gripper_candidate_sequences": "max_gripper_candidate_sequences_per_phase",
    }
    for key, budget in budgets.items():
        if key in counts:
            prefix.require(counts[key] <= config[budget], f"Registered {key} budget")
    prefix.require(counts["native_steps"] == 25 * counts["control_steps"], "Complete physical periods")
    return counts


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
        inputs = binding.load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "inputs": binding.input_receipt(inputs),
                    "source_refs": source_refs(),
                    "model_initializations": 0,
                    "physics_steps": 0,
                },
            )
        )
    elif args.producer or args.checker:
        phase = "producer" if args.producer else "checker"
        if phase == "checker":
            require_checker_independence()
        registered_feedback_phase.execute_phase(
            phase,
            root=ROOT,
            qa_path=QA,
            dry_path=DRY,
            branches=BRANCHES,
            worker_module="scripts.simulation.runtime.run_bounded_release_hold_feedback",
            branch_key=branch_key,
            load_inputs=binding.load_inputs,
            input_receipt=binding.input_receipt,
            source_refs=source_refs,
            summarize_phase=summarize_phase,
            config_key="config",
        )
    else:
        phase, index, mode = args.worker
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
