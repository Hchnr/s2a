"""Preserve completed control history across the registered hold and bridge."""

import numpy as np

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_bounded_release_hold_feedback as hold
from scripts.simulation.runtime import run_causal_event_switch_feedback as switch
from scripts.verification.reference_ports.check_bounded_release_bridge import verify_selected_reference


def compare_history(actual, expected):
    """Compare completed causal histories without comparing decision timing."""
    if isinstance(expected, dict):
        prefix.require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Completed history fields")
        for key in expected:
            compare_history(actual[key], expected[key])
    elif isinstance(expected, (list, tuple, np.ndarray)):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    elif isinstance(expected, (bool, str)) or expected is None:
        prefix.require(actual == expected, "Completed history value")
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def start_extended_hold(policy, selector, old, latest, config, receipt, **measured):
    """Extend the admitted budget only while switching to a covering reference."""
    prefix.require(policy.max_steps == 16, "The existing policy starts with its historical 16-period bound")
    prefix.require(len(receipt["continuation_partial"]["periods"]) == 8, "Eight actual periods precede extension")
    receipt["horizon_extension"] = {"before": 16, "requested": 24, "completed": False}
    policy.max_steps = 24
    try:
        stop = hold.start_hold(policy, selector, old, latest, config, receipt, **measured)
        if stop is not None:
            policy.max_steps = 16
            return stop
        prefix.require(
            policy.cursor == 0 and policy.poses.shape == (40, 4, 4) and policy.cursor + policy.max_steps + 7 <= 40,
            "The rebound generated reference covers the entire extended preview",
        )
        receipt["horizon_extension"].update(completed=True, cursor=policy.cursor, after=policy.max_steps)
        return None
    except BaseException:
        policy.max_steps = 16
        raise


def prepare_bridge(policy, selector, old, latest, receipt, *, measured_tcp, measured_opening, limits, current_q):
    """Select and bind the next reference using only the just-completed boundary."""
    from s2a.research.wam.tracking.preview.bounded_release_revision import ReferenceChunk

    periods = receipt["continuation_partial"]["periods"]
    prefix.require(len(periods) == 16 and periods[-1]["step"] == 375, "A complete actual hold precedes bridging")
    state = selector.state()
    prefix.require(
        state["mode"] == "held" and not state["staged"] and not state["faulted"], "The full hold must be committed"
    )
    limits, current_q = np.asarray(limits, dtype=float), np.asarray(current_q, dtype=float)
    samples = np.asarray([sample["q"] for sample in periods[-1]["actual_inner"]], dtype=float)
    prefix.require(
        limits.shape == (7, 2)
        and current_q.shape == (7,)
        and samples.shape == (25, 7)
        and np.isfinite(limits).all()
        and np.isfinite(current_q).all()
        and np.isfinite(samples).all(),
        "Current joints and 25 completed inner samples are required",
    )
    margin = float(np.min(np.minimum(samples - limits[:, 0], limits[:, 1] - samples)))
    current_margin = float(np.min(np.minimum(current_q - limits[:, 0], limits[:, 1] - current_q)))
    np.testing.assert_allclose(margin, periods[-1]["minimum_inner_margin_rad"], atol=1e-10, rtol=0)
    if policy.last_inner_loss or min(margin, current_margin) < 0.05:
        return None, {"reason": "observed_joint_buffer_loss", "at_step": 376, "next_action_dispatched": False}
    latch = policy.gripper.state()
    prefix.require(
        latch["staged"] is False and latch["faulted"] is False and type(latch["latch"]["closing"]) is bool,
        "The current gripper latch must be a completed state",
    )
    decision = selector.prepare(
        ReferenceChunk(360, latest["targets"], latest["openings"]),
        committed=ReferenceChunk(344, old["targets"], old["openings"]),
        step=376,
        measured_tcp=measured_tcp,
        measured_opening=measured_opening,
        closing_episode=latch["latch"]["closing"],
        previous_inner_margin_rad=margin,
    )
    selection = receipt["bridge_selection"] = {
        "decision": decision,
        "previous_inner_margin_rad": margin,
        "current_joint_margin_rad": current_margin,
        "measured_tcp": np.asarray(measured_tcp).copy(),
        "measured_opening": float(measured_opening),
        "closing_episode": latch["latch"]["closing"],
        "completed_bridge_steps": 0,
    }
    if (decision["source"], decision["reason"]) not in (("bridge", "bounded_bridge"), ("latest", "not_closing")):
        selector.abort(decision)
        selection["state_after_abort"] = selector.state()
        return None, {
            "reason": "bridge_reference_not_eligible",
            "selection_reason": decision["reason"],
            "at_step": 376,
            "next_action_dispatched": False,
        }
    try:
        selection["reference_audit"] = verify_selected_reference(
            decision, old["targets"], old["openings"], latest["targets"], latest["openings"]
        )
        before = switch.control_history(policy)
        targets, openings = policy.poses.copy(), policy.openings.copy()
        targets[16:], openings[16:] = decision["poses"], decision["openings"]
        detail = policy.rebind_reference(
            targets, openings, latest["anchor"], offset=16, next_cursor=16, generated_step=360
        )
        after = switch.control_history(policy)
        compare_history(after, before)
        selection["reference_switch"] = {"completed": True, "detail": detail, "unchanged_history": after}
    except BaseException:
        selector.abort(decision)
        selection["state_after_abort"] = selector.state()
        raise
    return decision, None


def finish_bridge(selector, receipt, *, segment_returned):
    """Commit only eight fully observed periods and isolate incomplete execution."""
    selection = receipt.get("bridge_selection")
    if selection is None or not selector.state()["staged"]:
        return
    completed = len(receipt.get("continuation_partial", {}).get("periods", []))
    prefix.require(completed <= 8, "Only the registered bridge stride may complete")
    selection["completed_bridge_steps"] = completed
    uncertain = receipt["control_steps_attempted"] != receipt["control_steps_completed"]
    selection["partial_physical_execution_uncertain"] = uncertain
    if uncertain or (completed == 8 and not segment_returned):
        selection["reconstruction_required"] = True
    elif completed == 8:
        selector.commit(selection["decision"], completed_steps=8)
    else:
        selector.abort(selection["decision"], completed_steps=completed)
    selection["state_after_execution"] = selector.state()
