"""Independently replay the actual fixed-H40 commands from reset."""

from types import SimpleNamespace

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.references.finite_horizon_window import remaining_h40
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


def run_checker(
    env,
    terminal,
    observers,
    receipt,
    *,
    case,
    producer,
    expected_capture,
    load_reference,
    runtime_output,
    progress=None,
    state_reader=None,
    task_observer=None,
    gripper_score=None,
    render_scope=render_with_two_torch_threads,
):
    """Reconstruct saved targets, rerender the anchor and replay only dispatched actions."""
    isolation.checker_isolation()
    gripper_score = original.measured_gripper if gripper_score is None else gripper_score
    runtime = producer["runtime"]
    original_run = runtime["continuation"]
    first, fixed = original_run["first_window"]["periods"], original_run["fixed"]
    segments = fixed["segments"]
    prefix.require(
        len(first) == 8
        and len(segments) <= 5
        and fixed["actual_model_requests"] == 0
        and fixed["automatic_retry"] is False
        and (
            (fixed["complete"] is True and fixed["stop"] is None)
            or (fixed["complete"] is False and fixed["stop"] is not None)
        ),
        "The producer must record one bounded fixed-reference run",
    )
    cache = SameStateEndpointCache(observers.endpoint)
    receipt["endpoint_cache_events"] = cache.events
    wrapped = SimpleNamespace(endpoint=cache, current=observers.current, render_cameras=observers.render_cameras)

    def score_expected(expected, opening):
        def complete(actual_env, _command, offset):
            period = receipt["continuation_partial"]["periods"][-1]
            endpoint = period["endpoint_receipt"]
            step = period["step"] + 1
            snapshot = measured_snapshot(actual_env, step)
            prefix.require(
                endpoint == expected[offset]["endpoint_receipt"]
                and endpoint["endpoint_step"] == step
                and endpoint["sim_time"] == snapshot["sim_time_s"]
                and endpoint["sim_state_sha256"] == snapshot["state_sha256"]
                and state_sha256(period["actual_state"]) == snapshot["state_sha256"],
                "The independently measured endpoint must match the saved dispatch",
            )
            score = gripper_score(actual_env, opening[offset])
            original.compare_score(score, expected[offset]["completed_feedback"]["score"])
            return dict(score=score, policy_history=None, policy_history_recomputed=False)

        return complete

    first_replay = run_continuation(
        env,
        terminal,
        wrapped,
        receipt,
        start_step=0,
        previous_command=np.zeros(7),
        targets=np.asarray(case["chunk"]["effective_targets_tcp"][:8]),
        tool=case["tool"],
        target_object=case["target_object"],
        choose_action=original.checker_action(first),
        stop_on_outside=True,
        expected_periods=first,
        on_completed=score_expected(first, case["chunk"]["openings"]),
        state_reader=state_reader,
        task_observer=task_observer,
        progress=progress,
    )
    receipt["reset_first_window"] = first_replay
    if not segments:
        prefix.require(fixed["stop"] is not None, "An empty fixed reference requires an explicit stop")
        return dict(
            first_window=first_replay,
            replays=[],
            reported_producer_stop=fixed["stop"],
            stop_reason_independently_verified=False,
        )
    sample, endpoint = cache(env, 8)
    current = measured_snapshot(env, 8)
    scope = {}
    capture = render_scope(
        observers.render_anchor, scope, env, 8, sample, minimal_renderer_endpoint(endpoint, 8), runtime_output
    )
    receipt["checker_fixed_capture"] = capture
    receipt["checker_fixed_canvas_scope"] = scope.get("canvas_thread_scope")
    capture_check.compare_capture(expected_capture, capture)
    capture_check.compare_capture(runtime["fixed_reference_capture"], capture)
    prefix.require(measured_snapshot(env, 8) == current, "Independent image rendering cannot advance physics")
    response, envelope = load_reference()
    arrays = repro.response_arrays(response)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    prefix.require(
        envelope["step"] == 8
        and envelope["state_sha256"] == current["state_sha256"]
        and envelope["sim_time_s"] == current["sim_time_s"]
        and envelope["image_sha256"] == capture["canvas_png"]["sha256"]
        and envelope["denoise_calls"] == 35,
        "The saved reference must bind the independently reproduced current observation",
    )
    anchor = np.asarray(sample["source_pose"], dtype=float) @ np.asarray(case["tool"], dtype=float)
    poses, openings = decode_action_targets(anchor, arrays["action"], horizon=40)
    previous_command = first[-1]["command"]
    replays = []
    for index, segment in enumerate(segments):
        offset, step = 8 * index, 8 + 8 * index
        expected = segment["continuation"]["periods"]
        window_poses, window_openings, metadata = remaining_h40(poses, openings, consumed=offset)
        prefix.require(
            segment["start_step"] == step
            and segment["reference_offset"] == offset
            and segment["generated_endpoints"] == metadata["generated_endpoints"]
            and segment["terminal_hold_endpoints"] == metadata["terminal_hold_endpoints"]
            and segment["continuation"]["steps_completed"] == len(expected)
            and 0 <= len(expected) <= 8
            and (index == len(segments) - 1 or len(expected) == 8),
            "Each replay must follow the producer's actual generated-reference offsets",
        )
        sample, endpoint = cache(env, step)
        replay = run_continuation(
            env,
            {"sample": sample, "endpoint_receipt": endpoint},
            wrapped,
            receipt,
            start_step=step,
            previous_command=previous_command,
            targets=window_poses[:8],
            tool=case["tool"],
            target_object=case["target_object"],
            choose_action=original.checker_action(expected),
            stop_on_outside=True,
            expected_periods=expected,
            on_completed=score_expected(expected, window_openings),
            state_reader=state_reader,
            task_observer=task_observer,
            progress=progress,
        )
        replays.append(replay)
        if expected:
            previous_command = expected[-1]["command"]
    prefix.require(
        sum(len(row["periods"]) for row in replays) <= 40
        and (not fixed["complete"] or sum(len(row["periods"]) for row in replays) == 40),
        "The checker cannot replay more than the original H40",
    )
    if fixed["stop"] is not None:
        stop = fixed["stop"]
        next_step = 8 + sum(len(row["periods"]) for row in replays)
        prefix.require(
            type(stop) is dict
            and type(stop.get("reason")) is str
            and bool(stop["reason"])
            and stop.get("next_action_dispatched") is False
            and (
                ("at_step" in stop and "after_step" not in stop and stop["at_step"] == next_step)
                or ("after_step" in stop and "at_step" not in stop and stop["after_step"] + 1 == next_step)
            ),
            "The saved stop must immediately follow the actual dispatched periods",
        )
    isolation.checker_isolation()
    return dict(
        first_window=first_replay,
        replays=replays,
        reported_producer_stop=fixed["stop"],
        stop_reason_independently_verified=False,
    )
