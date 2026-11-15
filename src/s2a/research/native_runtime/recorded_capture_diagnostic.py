"""Compare a recorded current-image capture after a bounded causal prefix."""

import copy
from types import SimpleNamespace

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.streaming_continuation import measured_snapshot

EXACT_CAPTURE_FIELDS = (
    "frame_index",
    "current_endpoint",
    "render_before",
    "render_after",
    "future_rgb_frames",
    "camera_keys",
    "canvas_shape",
    "canvas_dtype",
    "canvas_rgb_sha256",
)


def complete_windows_before(segments, *, capture_step):
    """Copy complete eight-period windows without touching later segments."""
    if not isinstance(segments, list) or type(capture_step) is not int or capture_step < 16 or capture_step % 8:
        raise ValueError("A complete eight-period prefix and boundary capture are required")
    result, step = [], 8
    for segment in segments:
        if step == capture_step:
            break
        periods = segment["continuation"]["periods"]
        if (
            segment["start_step"] != step
            or segment["boundary"] != "window_end"
            or [item["step"] for item in periods] != list(range(step, step + 8))
            or any(item.get("replan") is True for item in segment["signals"])
        ):
            raise ValueError("Every earlier reference window must be complete and consecutive")
        result.append(dict(start_step=step, periods=copy.deepcopy(periods)))
        step += 8
    if step != capture_step:
        raise ValueError("The recorded windows must reach the capture without a gap")
    return result


def _different_paths(expected, actual, path):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            name = f"{path}.{key}"
            if key not in expected or key not in actual:
                result.append(name)
            else:
                result.extend(_different_paths(expected[key], actual[key], name))
        return result
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [path]
        return [
            item
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
            for item in _different_paths(left, right, f"{path}[{index}]")
        ]
    return [] if expected == actual else [path]


def capture_differences(expected, actual):
    """Explain the same exact and 1e-10 comparisons as the existing capture guard."""
    paths, results = [], {}
    for key in EXACT_CAPTURE_FIELDS:
        differences = (
            [key] if key not in expected or key not in actual else _different_paths(expected[key], actual[key], key)
        )
        results[key] = not differences
        paths.extend(differences)
    expected_sha = expected.get("canvas_png", {}).get("sha256")
    actual_sha = actual.get("canvas_png", {}).get("sha256")
    results["canvas_png.sha256"] = expected_sha is not None and expected_sha == actual_sha
    if not results["canvas_png.sha256"]:
        paths.append("canvas_png.sha256")
    for key in ("current_source_pose", "current_opening"):
        try:
            np.testing.assert_allclose(expected[key], actual[key], atol=1e-10, rtol=0)
            results[key] = True
        except (AssertionError, KeyError, TypeError, ValueError):
            results[key] = False
            paths.append(key)
    return dict(passed=all(results.values()), fields=results, differing_paths=paths)


def replay_until_capture(
    env,
    terminal,
    observers,
    receipt,
    *,
    phase,
    case,
    initial_periods,
    windows,
    capture_step,
    load_window,
    capture_final,
    score_gripper,
    compare_score,
    progress=None,
):
    """Recompute or replay the prior prefix, then capture without another action."""
    if (
        phase not in ("producer", "checker")
        or [item["step"] for item in initial_periods] != list(range(8))
        or [item["start_step"] for item in windows] != list(range(8, capture_step, 8))
        or any(
            [item["step"] for item in window["periods"]] != list(range(window["start_step"], window["start_step"] + 8))
            for window in windows
        )
        or not all(callable(item) for item in (load_window, capture_final, score_gripper, compare_score))
    ):
        raise ValueError("The fixed complete prefix, capture boundary and callbacks are required")
    if phase == "producer":
        from s2a.research.joint_control.preview.reset_feedback_policy import policy_inputs
        from s2a.research.joint_control.preview.segmented_reset_policy import (
            advance_segmented_reset_policy,
            build_segmented_reset_policy,
        )

        policy, owner, previous = build_segmented_reset_policy(env, policy_inputs(case, terminal), receipt)
    else:
        policy = owner = None
        previous = np.zeros(7)

    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    wrapped = SimpleNamespace(
        endpoint=cache,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )

    def execute_window(start, current, targets, openings, expected):
        nonlocal previous

        def choose(q, limits, jacobian, prior, offset):
            before = measured_snapshot(env, start + offset)
            if phase == "checker":
                return dict(command=np.asarray(expected[offset]["command"], dtype=float), newly_selected=False)
            decision = policy.select(q, limits, jacobian, prior, offset)
            if measured_snapshot(env, start + offset) != before:
                raise ValueError("Prefix selection cannot change the measured state")
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
                raise ValueError("Prefix feedback cannot change the measured state")
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

    initial = execute_window(
        0,
        terminal,
        np.asarray(case["chunk"]["effective_targets_tcp"][:8]),
        np.asarray(case["chunk"]["openings"][:8]),
        initial_periods,
    )
    completed_windows = []
    for index, window in enumerate(windows):
        start = window["start_step"]
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
        outcome = execute_window(start, current, poses[:8], openings[:8], window["periods"])
        completed_windows.append(dict(start_step=start, continuation=outcome))

    sample, endpoint = cache(env, capture_step)
    current = dict(sample=sample, endpoint_receipt=endpoint)
    before = measured_snapshot(env, capture_step)
    target = capture_final(env, current, capture_step)
    if measured_snapshot(env, capture_step) != before:
        raise ValueError("The diagnostic capture cannot advance or change the measured state")
    return dict(
        initial=initial,
        windows=completed_windows,
        capture=target,
        capture_step=capture_step,
        stop="captured_before_next_action",
        next_action_dispatched=False,
    )
