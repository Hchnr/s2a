"""Rebuild a recorded reference prefix and probe one current-state action."""

import copy
from types import SimpleNamespace

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.streaming_continuation import measured_snapshot


def first_refresh_prefix(segments, *, expected_step, expected_reason):
    """Return only completed windows up to the first recorded refresh signal."""
    if (
        not isinstance(segments, list)
        or not segments
        or type(expected_step) is not int
        or expected_step <= 8
        or expected_reason not in ("quality_margin_consumed", "no_audited_second_pose")
    ):
        raise ValueError("Recorded segments and a fixed first-refresh identity are required")
    result, next_step = [], 8
    for segment in segments:
        periods = segment["continuation"]["periods"]
        signals = segment["signals"]
        if (
            segment["start_step"] != next_step
            or not isinstance(periods, list)
            or not 1 <= len(periods) <= 8
            or [row["step"] for row in periods] != list(range(next_step, next_step + len(periods)))
            or not isinstance(signals, list)
        ):
            raise ValueError("The recorded reference windows must be consecutive and bounded")
        triggered = [signal for signal in signals if signal.get("replan") is True]
        result.append(dict(start_step=next_step, periods=copy.deepcopy(periods)))
        next_step += len(periods)
        if triggered:
            if (
                len(triggered) != 1
                or triggered[0].get("current_step") != next_step
                or triggered[0].get("reason") != expected_reason
                or next_step != expected_step
                or segment.get("boundary") != "quality_trigger"
                or len(periods) >= 8
            ):
                raise ValueError("The first refresh must match the fixed completed-prefix boundary")
            return dict(windows=result, trigger=copy.deepcopy(triggered[0]))
        if len(periods) != 8 or segment.get("boundary") != "window_end":
            raise ValueError("Earlier windows must each finish eight periods before another reference")
    raise ValueError("The registered first refresh was not recorded")


def run_matched_reference_probe(
    env,
    terminal,
    observers,
    receipt,
    *,
    phase,
    case,
    initial_periods,
    windows,
    expected_trigger,
    load_window,
    score_gripper,
    compare_score,
    recorded_probe=None,
    progress=None,
    adapter_factory=None,
    activate_probe=None,
):
    """Recompute or replay one prefix, then dispatch at most one new action.

    ``load_window`` must independently bind each reference to the current
    measured state and already-recorded image. Historical actions are used
    only to reject a divergent prefix before dispatch and to check replay.
    """
    if (
        phase not in ("producer", "checker")
        or (phase == "checker") != (recorded_probe is not None)
        or not isinstance(initial_periods, list)
        or len(initial_periods) != 8
        or [row["step"] for row in initial_periods] != list(range(8))
        or not isinstance(windows, list)
        or not windows
        or not callable(load_window)
        or not callable(score_gripper)
        or not callable(compare_score)
        or (adapter_factory is None) != (activate_probe is None)
        or (adapter_factory is not None and (not callable(adapter_factory) or not callable(activate_probe)))
        or (phase == "checker" and adapter_factory is not None)
    ):
        raise ValueError("One fixed recorded prefix and one producer/checker mode are required")
    expected_step = expected_trigger.get("current_step")
    if type(expected_step) is not int or windows[-1]["start_step"] + len(windows[-1]["periods"]) != expected_step:
        raise ValueError("The original signal must follow the last completed reference period")
    if phase == "producer":
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import (
            advance_segmented_reset_policy,
            build_segmented_reset_policy,
        )
        from s2a.research.native_runtime.deferred_margin_recheck import DeferredMarginRecheck

        if adapter_factory is None:
            policy, owner, previous = build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)
        else:
            policy, owner, previous = build_segmented_reset_policy(
                env, policy_inputs(case, terminal), receipt, adapter_factory=adapter_factory
            )
        gate = DeferredMarginRecheck(target_step=expected_step)
    else:
        policy = owner = gate = None
        previous = np.zeros(7)

    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    wrapped = SimpleNamespace(
        endpoint=cache,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )
    completed_windows = []

    def run_recorded_window(start, current, targets, openings, expected):
        nonlocal previous

        def choose(q, limits, jacobian, prior, offset):
            before = measured_snapshot(env, start + offset)
            if phase == "checker":
                return dict(command=np.asarray(expected[offset]["command"], dtype=float), newly_selected=False)
            decision = policy.select(q, limits, jacobian, prior, offset)
            if measured_snapshot(env, start + offset) != before:
                raise ValueError("Prefix selection cannot advance or change the physical state")
            if decision.get("newly_selected") is not True or "stop" in decision:
                raise ValueError("The recomputed prefix must select the original action")
            np.testing.assert_array_equal(decision["command"], expected[offset]["command"])
            return decision

        def completed(actual_env, command, offset):
            before = measured_snapshot(actual_env, start + offset + 1)
            period = receipt["continuation_partial"]["periods"][-1]
            history = None
            if phase == "producer":
                history = policy.complete(actual_env, command, offset, [item["q"] for item in period["actual_inner"]])
            score = score_gripper(actual_env, openings[offset])
            compare_score(score, expected[offset]["completed_feedback"]["score"])
            if measured_snapshot(actual_env, start + offset + 1) != before:
                raise ValueError("Prefix feedback cannot advance or change the physical state")
            return dict(score=score, policy_history=history, policy_history_recomputed=phase == "producer")

        outcome = run_continuation(
            env,
            current,
            wrapped,
            receipt,
            start_step=start,
            previous_command=previous,
            targets=targets,
            tool=case["tool"],
            target_object=case["target_object"],
            choose_action=choose,
            stop_on_outside=True,
            expected_periods=expected,
            on_completed=completed,
            progress=progress,
        )
        if outcome["stop"] is not None or outcome["steps_completed"] != len(expected):
            raise ValueError("Every registered prefix period must complete and match")
        previous = np.asarray(outcome["periods"][-1]["command"], dtype=float).copy()
        return outcome

    initial = run_recorded_window(
        0,
        terminal,
        np.asarray(case["chunk"]["effective_targets_tcp"][:8]),
        np.asarray(case["chunk"]["openings"][:8]),
        initial_periods,
    )
    current = terminal
    poses = openings = None
    for index, window in enumerate(windows):
        start, expected = window["start_step"], window["periods"]
        sample, endpoint = cache(env, start)
        current = dict(sample=sample, endpoint_receipt=endpoint)
        before = measured_snapshot(env, start)
        poses, openings = load_window(env, current, index, start)
        poses, openings = validate_transform(np.asarray(poses, dtype=float)), np.asarray(openings, dtype=float)
        if (
            poses.shape != (40, 4, 4)
            or openings.shape != (40,)
            or not np.isfinite(openings).all()
            or np.any((openings < 0) | (openings > 1))
            or measured_snapshot(env, start) != before
        ):
            raise ValueError("A current-state-bound complete generated reference is required")
        if phase == "producer":
            anchor = np.asarray(sample["source_pose"]) @ np.asarray(case["tool"])
            policy = advance_segmented_reset_policy(
                policy, owner, poses, openings, anchor, generated_step=start, runtime_receipt=receipt
            )
        outcome = run_recorded_window(start, current, poses[:8], openings[:8], expected)
        completed_windows.append(dict(start_step=start, continuation=outcome))

    count = len(windows[-1]["periods"])
    if not 1 <= count < 8:
        raise ValueError("A refresh probe requires a partially completed current reference")
    sample, endpoint = cache(env, expected_step)
    current = dict(sample=sample, endpoint_receipt=endpoint)
    last = completed_windows[-1]["continuation"]["periods"][-1]
    if last["step"] != expected_step - 1 or endpoint != last["endpoint_receipt"]:
        raise ValueError("The probe must start at its own actual completed endpoint")
    original_stop = None
    if phase == "producer":
        from s2a.research.joint_control.preview.causal_reference_replan import completed_quality_margin
        from s2a.research.native_runtime.reference_choice_comparison import quality_stop

        original_stop = quality_stop(last, expected_step, completed_quality_margin)
        if (
            original_stop is None
            or original_stop["stop"]["trigger"]["reason"] != expected_trigger["reason"]
            or original_stop["stop"]["trigger"]["current_step"] != expected_step
        ):
            raise ValueError("The recomputed trigger must match the recorded completed-state signal")
        if activate_probe is not None:
            before = measured_snapshot(env, expected_step)
            activate_probe(policy, owner, receipt, expected_step)
            if measured_snapshot(env, expected_step) != before:
                raise ValueError("Probe activation cannot advance or change the physical state")

    def choose_probe(q, limits, jacobian, prior, offset):
        before = measured_snapshot(env, expected_step + offset)
        if phase == "checker":
            return dict(command=np.asarray(recorded_probe["periods"][offset]["command"]), newly_selected=False)
        if offset:
            completed = receipt["continuation_partial"]["periods"][-1]
            reason = (
                "observed_joint_buffer_loss"
                if completed["inner_below_buffer_samples"]
                else "observed_pose_cap_exceeded"
                if not completed["pose_caps_satisfied"]
                else "single_probe_complete"
            )
            return {"stop": {"reason": reason}}
        if last["inner_below_buffer_samples"]:
            return {"stop": {"reason": "observed_joint_buffer_loss"}}
        if not last["pose_caps_satisfied"]:
            return {"stop": {"reason": "observed_pose_cap_exceeded"}}
        stop = gate(last, expected_step, completed_quality_margin)
        decision = stop if stop is not None else policy.select(q, limits, jacobian, prior, count)
        if measured_snapshot(env, expected_step + offset) != before:
            raise ValueError("Probe selection cannot advance or change the physical state")
        return decision

    def complete_probe(actual_env, command, offset):
        before = measured_snapshot(actual_env, expected_step + offset + 1)
        period = receipt["continuation_partial"]["periods"][-1]
        history = None
        if phase == "producer":
            history = policy.complete(
                actual_env, command, count + offset, [item["q"] for item in period["actual_inner"]]
            )
        score = score_gripper(actual_env, openings[count + offset])
        if phase == "checker":
            compare_score(score, recorded_probe["periods"][offset]["completed_feedback"]["score"])
        if measured_snapshot(actual_env, expected_step + offset + 1) != before:
            raise ValueError("Probe feedback cannot advance or change the physical state")
        return dict(score=score, policy_history=history, policy_history_recomputed=phase == "producer")

    outcome = run_continuation(
        env,
        current,
        wrapped,
        receipt,
        start_step=expected_step,
        previous_command=previous,
        targets=poses[count : count + 8],
        tool=case["tool"],
        target_object=case["target_object"],
        choose_action=choose_probe,
        stop_on_outside=True,
        expected_periods=None if phase == "producer" else recorded_probe["periods"],
        on_completed=complete_probe,
        progress=progress,
    )
    if outcome["steps_completed"] > 1:
        raise ValueError("Only one additional action may be dispatched")
    return dict(
        initial=initial,
        windows=completed_windows,
        probe=outcome,
        original_trigger=copy.deepcopy(expected_trigger),
        original_stop=original_stop,
        deferred_events=None if gate is None else gate.events,
        stop_recomputed=phase == "producer",
    )
