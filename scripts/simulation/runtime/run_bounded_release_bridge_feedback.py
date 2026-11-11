"""Execute and independently replay the registered continuous release bridge."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.simulation.runtime import bounded_release_bridge_transition as transition
from scripts.simulation.runtime import prepare_bounded_release_bridge as binding
from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as prior_binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_bounded_release_hold_feedback as hold
from scripts.simulation.runtime import run_causal_event_feedback as base
from scripts.simulation.runtime import run_causal_event_long_feedback as long
from scripts.verification.reference_ports.check_bounded_release_bridge import verify_selected_reference
from scripts.wam.tracking.preview import bind_bounded_release_shadow as shadow

REPO = prefix.REPO
ROOT = REPO / "outputs/bounded_release_bridge_feedback_20261111"
QA = REPO / "outputs/qa/bounded_release_bridge_feedback_20261111/result.json"
DRY = REPO / "outputs/bounded_release_bridge_feedback_dry_20261111/result.json"
BRANCHES = ((5, "bounded_release_bridge"),)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "scripts/simulation/runtime/bounded_release_bridge_transition.py",
            REPO / "scripts/verification/reference_ports/check_bounded_release_bridge.py",
            REPO / "src/s2a/research/native_runtime/development_prefix_runtime_24.py",
            REPO / "tests/research/native_runtime/test_bounded_release_bridge_feedback.py",
            REPO / "tests/research/native_runtime/test_bounded_release_bridge_transition.py",
            REPO / "tests/research/native_runtime/test_bounded_release_bridge_reference.py",
            REPO / "tests/research/native_runtime/test_development_prefix_runtime_24.py",
        )
    ]


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "The single registered continuous bridge")
    return f"case_{index:02d}_{mode}"


def selector_from_registration():
    from s2a.research.wam.tracking.preview.bounded_release_revision import BoundedReleaseRevision, ReleaseRevisionConfig

    revision = prefix.read(prefix.reference(shadow.CONFIG))
    names = (
        "stride",
        "bridge_steps",
        "release_clearance",
        "opening_revision_gap",
        "current_position_gate_m",
        "current_rotation_gate_rad",
        "joint_buffer_rad",
    )
    return BoundedReleaseRevision(ReleaseRevisionConfig(**{name: revision[name] for name in names}))


def score_bridge(env, case, old, decision, period, offset, *, expected=None):
    """Score actual state against the selected and latest generated references."""
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    actual_tcp = site @ site_from_tcp
    global_offset = 16 + offset
    latest = case["references"]["latest"]
    latest_error = pose_error(latest["poses"][latest["cursor"] + global_offset], actual_tcp)
    scores = base.score_completed(env, case, old, "latest", global_offset, latest_error)
    selected_error = pose_error(decision["poses"][offset], actual_tcp)
    np.testing.assert_allclose(selected_error, period["tcp_error"], atol=1e-10, rtol=0)
    opening_target = float(decision["openings"][offset])
    scores.update(
        selected_reference_error=selected_error,
        selected_opening_target=opening_target,
        selected_opening_error=scores["gripper"]["opening"] - opening_target,
    )
    if expected is not None:
        transition.compare_history(scores, expected)
    return scores


def verify_checker_boundary(env, case, old, latest, selection, period):
    """Verify measured boundary and bridge geometry without loading the selector."""
    hold.require_checker_independence()
    q, limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    measured_tcp, opening = site @ site_from_tcp, base.measured_gripper(env)["opening"]
    samples = np.asarray([item["q"] for item in period["actual_inner"]], dtype=float)
    prefix.require(samples.shape == (25, 7), "The checker has all preceding inner samples")
    margin = float(np.min(np.minimum(samples - limits[:, 0], limits[:, 1] - samples)))
    current = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    np.testing.assert_allclose(margin, selection["previous_inner_margin_rad"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(current, selection["current_joint_margin_rad"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(measured_tcp, selection["measured_tcp"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(opening, selection["measured_opening"], atol=1e-10, rtol=0)
    prefix.require(min(margin, current) >= 0.05, "Observed bridge boundary remains inside the joint buffer")
    return verify_selected_reference(
        selection["decision"], old["targets"], old["openings"], latest["targets"], latest["openings"]
    )


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        binding.input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen continuous bridge inputs and sources",
    )
    if phase == "checker":
        hold.require_checker_independence()
    case = inputs["case"]
    old, latest = prior_binding.selected_references(case)
    prior = prefix.read(inputs["prior_hold_branch_refs"]["producer"])
    prior_periods = prior["runtime"]["continuation"]["periods"]
    producer, expected, expected_stop = None, None, None
    if phase == "checker":
        producer = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(producer["complete"] is producer["passed"] is True, "Closed bridge producer")
        expected = producer["runtime"]["continuation"]["periods"]
        expected_stop = producer["runtime"]["continuation"]["stop"]
        prefix.require(16 <= len(expected) <= 24, "The producer reproduced the complete earlier hold")
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
        from s2a.research.native_runtime.development_prefix_runtime_24 import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            base.check_prefix_inner(receipt["final_prefix_inner"], prior["runtime"]["final_prefix_inner"])
            start_position = long.target_position(env, case["spec"]["target"])
            np.testing.assert_allclose(start_position, prior["runtime"]["target_start_position"], atol=1e-10, rtol=0)
            receipt["target_start_position"], receipt["previous_verified_steps"] = start_position, 0
            policy, selector = None, None
            if phase == "checker":
                first_action = base.checker_action(expected[:16])
            else:
                selector = selector_from_registration()
                policy = base.make_feedback(env, case, old, gripper, receipt, max_steps=16)

                def first_action(q, limits, jacobian, previous, offset):
                    if offset == 8 and not policy.last_inner_loss:
                        _mass, _velocity, site = current_robot_dynamics(env, jacobian)
                        measured = base.measured_gripper(env)
                        stop = transition.start_extended_hold(
                            policy,
                            selector,
                            old,
                            latest,
                            inputs["hold_config"],
                            receipt,
                            measured_tcp=site @ policy.site_from_tcp,
                            measured_opening=measured["opening"],
                            limits=limits,
                        )
                        if stop is not None:
                            return stop
                    return policy.select(q, limits, jacobian, previous, offset)

            def first_completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == 360 + offset, "The matching first segment just completed")
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [item["q"] for item in period["actual_inner"]]
                    )
                    transition.compare_history(history, prior_periods[offset]["completed_feedback"]["policy_history"])
                long.check_previous_period(period, prior_periods[offset])
                prior_feedback = prior_periods[offset]["completed_feedback"]
                scores = base.score_completed(
                    actual_env, case, old, "remembered", offset, period["tcp_error"], expected=prior_feedback["scores"]
                )
                position = long.target_position(actual_env, case["spec"]["target"])
                height = float(position[2] - start_position[2])
                np.testing.assert_allclose(position, prior_feedback["object_position"], atol=1e-10, rtol=0)
                np.testing.assert_allclose(height, prior_feedback["object_height_change_m"], atol=1e-10, rtol=0)
                if expected is not None:
                    transition.compare_history(scores, expected[offset]["completed_feedback"]["scores"])
                receipt["previous_verified_steps"] += 1
                return {
                    "scores": scores,
                    "object_position": position,
                    "object_height_change_m": height,
                    "policy_history": history,
                    "policy_history_recomputed": policy is not None,
                }

            try:
                first = run_continuation(
                    env,
                    terminal,
                    observers,
                    receipt,
                    start_step=360,
                    previous_command=case["commands"][-1],
                    targets=old["fixed_targets"],
                    tool=case["tool"],
                    target_object=case["spec"]["target"],
                    choose_action=first_action,
                    stop_on_outside=True,
                    expected_periods=None if expected is None else expected[:16],
                    on_completed=first_completed,
                    progress=progress,
                    continuation_limit=16,
                )
            finally:
                if selector is not None:
                    hold.finish_selection(selector, receipt)
            prefix.require(
                first["steps_completed"] == receipt["previous_verified_steps"] == 16 and first["stop"] is None,
                "All 16 earlier physical periods must reproduce before bridging",
            )
            q, limits, jacobian = current_joint_site_state(env)
            _mass, _velocity, site = current_robot_dynamics(env, jacobian)
            measured = base.measured_gripper(env)
            site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
            if phase == "checker":
                selection = producer["runtime"].get("bridge_selection")
                if (
                    selection is None
                    or expected_stop
                    and expected_stop.get("reason") == "bridge_reference_not_eligible"
                ):
                    prefix.require(expected_stop is not None and len(expected) == 16, "Recorded boundary stop")
                    if expected_stop["reason"] == "observed_joint_buffer_loss":
                        samples = np.asarray([item["q"] for item in first["periods"][-1]["actual_inner"]])
                        margin = float(np.min(np.minimum(samples - limits[:, 0], limits[:, 1] - samples)))
                        current = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                        prefix.require(min(margin, current) < 0.05, "The replay must observe the buffer stop")
                    return dict(
                        periods=first["periods"],
                        steps_completed=16,
                        stop=expected_stop,
                        decision_stop_recorded_not_recomputed=True,
                    )
                receipt["bridge_reference_audit"] = verify_checker_boundary(
                    env, case, old, latest, selection, first["periods"][-1]
                )
                decision, boundary_stop = selection["decision"], None
            else:
                decision, boundary_stop = transition.prepare_bridge(
                    policy,
                    selector,
                    old,
                    latest,
                    receipt,
                    measured_tcp=site @ site_from_tcp,
                    measured_opening=measured["opening"],
                    limits=limits,
                    current_q=q,
                )
            if boundary_stop is not None:
                return dict(periods=first["periods"], steps_completed=16, stop=boundary_stop)

            def second_action(q, limits, jacobian, previous, offset):
                if phase == "checker":
                    return {"command": np.asarray(expected[16 + offset]["command"]), "newly_selected": False}
                return policy.select(q, limits, jacobian, previous, 16 + offset)

            def second_completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == 376 + offset, "The selected bridge period just completed")
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, 16 + offset, [item["q"] for item in period["actual_inner"]]
                    )
                checked = None if expected is None else expected[16 + offset]["completed_feedback"]
                scores = score_bridge(
                    actual_env,
                    case,
                    old,
                    decision,
                    period,
                    offset,
                    expected=None if checked is None else checked["scores"],
                )
                position = long.target_position(actual_env, case["spec"]["target"])
                height = float(position[2] - start_position[2])
                if checked is not None:
                    np.testing.assert_allclose(position, checked["object_position"], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(height, checked["object_height_change_m"], atol=1e-10, rtol=0)
                return {
                    "scores": scores,
                    "object_position": position,
                    "object_height_change_m": height,
                    "policy_history": history,
                    "policy_history_recomputed": policy is not None,
                }

            returned = False
            try:
                second = run_continuation(
                    env,
                    {"endpoint_receipt": first["periods"][-1]["endpoint_receipt"]},
                    observers,
                    receipt,
                    start_step=376,
                    previous_command=first["periods"][-1]["command"],
                    targets=np.asarray(decision["poses"])[:8],
                    tool=case["tool"],
                    target_object=case["spec"]["target"],
                    choose_action=second_action,
                    stop_on_outside=True,
                    expected_periods=None if expected is None else expected[16:],
                    on_completed=second_completed,
                    progress=progress,
                    continuation_limit=8,
                )
                returned = True
            finally:
                if selector is not None:
                    transition.finish_bridge(selector, receipt, segment_returned=returned)
            if phase == "checker":
                q, limits, _ = current_joint_site_state(env)
                current = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                second = base.reconcile_stop(
                    second,
                    expected_stop,
                    start_step=376,
                    current_margin=current,
                    prefix_inner_loss=first["periods"][-1]["inner_below_buffer_samples"] > 0,
                    horizon=8,
                )
            periods = first["periods"] + second["periods"]
            return {
                "periods": periods,
                "steps_completed": len(periods),
                "stop": second["stop"],
                "segments": [
                    {"start_step": 360, "steps_completed": 16},
                    {"start_step": 376, "steps_completed": len(second["periods"])},
                ],
                "decision_stop_recorded_not_recomputed": second.get("decision_stop_recorded_not_recomputed", False),
                "release_selection_not_recomputed": phase == "checker",
            }

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
                continuation_budget=24,
                continuation_limit=24,
                registered_seed=42,
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in hold.CHECKER_FORBIDDEN)
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
            hold.require_checker_independence()
        registered_feedback_phase.execute_phase(
            phase,
            root=ROOT,
            qa_path=QA,
            dry_path=DRY,
            branches=BRANCHES,
            worker_module="scripts.simulation.runtime.run_bounded_release_bridge_feedback",
            branch_key=branch_key,
            load_inputs=binding.load_inputs,
            input_receipt=binding.input_receipt,
            source_refs=source_refs,
            summarize_phase=hold.summarize_phase,
            config_key="config",
        )
    else:
        phase, index, mode = args.worker
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
