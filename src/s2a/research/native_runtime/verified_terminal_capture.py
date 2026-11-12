"""Capture a reproduced current endpoint without advancing physics."""

import copy

import numpy as np

from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.training.native.native_training import state_sha256


def capture_current_endpoint(
    env,
    render,
    sample,
    endpoint,
    *,
    step,
    expected_endpoint,
    runtime_output,
    render_scope=render_with_two_torch_threads,
):
    """Render only a measured endpoint that exactly matches the recorded one."""
    if (
        type(step) is not int
        or not 0 <= step <= 600
        or type(sample) is not dict
        or type(endpoint) is not dict
        or type(expected_endpoint) is not dict
        or not callable(render)
        or not callable(render_scope)
    ):
        raise ValueError("A bounded current endpoint, recorded identity and renderer are required")
    current = measured_snapshot(env, step)
    binding = {"sim_time": current["sim_time_s"], "sim_state_sha256": current["state_sha256"]}
    identity = {"endpoint_step": step, **binding}
    if (
        endpoint != expected_endpoint
        or any(endpoint.get(key) != value for key, value in identity.items())
        or state_sha256(sample["sim_states"]) != current["state_sha256"]
    ):
        raise ValueError("The observed endpoint must match the recorded current state")
    pose = np.asarray(sample["source_pose"]).copy()
    opening = sample["opening"]
    scope = {}
    capture = render_scope(render, scope, env, step, sample, copy.deepcopy(identity), runtime_output)
    if (
        measured_snapshot(env, step) != current
        or capture["frame_index"] != step
        or capture["current_endpoint"] != identity
        or capture["render_before"] != binding
        or capture["render_after"] != binding
        or capture["future_rgb_frames"] != 0
        or capture["current_opening"] != opening
    ):
        raise ValueError("The rendered image must preserve the current endpoint without future frames")
    np.testing.assert_array_equal(capture["current_source_pose"], pose)
    return dict(snapshot=current, capture=capture, canvas_thread_scope=scope.get("canvas_thread_scope"))
