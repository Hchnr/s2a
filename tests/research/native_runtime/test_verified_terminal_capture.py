"""Terminal capture binds an immutable recorded endpoint and current pixels."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime.verified_terminal_capture import capture_current_endpoint
from tests.research.native_runtime.test_streaming_continuation import fake_case


def setup_capture(tmp_path, *, mutation=None):
    case = fake_case(tmp_path)
    sample, endpoint = case.observers.endpoint(case.env, 88)

    def render(env, step, sample, identity, _output):
        binding = {"sim_time": identity["sim_time"], "sim_state_sha256": identity["sim_state_sha256"]}
        result = dict(
            frame_index=step,
            current_endpoint=identity,
            render_before=dict(binding),
            render_after=dict(binding),
            current_source_pose=sample["source_pose"].copy(),
            current_opening=sample["opening"],
            future_rgb_frames=0,
        )
        if mutation == "endpoint":
            result["current_endpoint"]["sim_state_sha256"] = "a" * 64
        elif mutation == "state":
            env.sim.data.time += 0.002
        elif mutation == "future":
            result["future_rgb_frames"] = 1
        elif mutation == "pose":
            result["current_source_pose"][0, 3] += 1
        return result

    options = dict(
        step=88,
        expected_endpoint=copy.deepcopy(endpoint),
        runtime_output=tmp_path,
        render_scope=lambda actual, _scope, *args, **kwargs: actual(*args, **kwargs),
    )
    return SimpleNamespace(case=case, sample=sample, endpoint=endpoint, render=render, options=options)


def execute(value):
    return capture_current_endpoint(value.case.env, value.render, value.sample, value.endpoint, **value.options)


def test_current_endpoint_capture_preserves_simulation(tmp_path):
    value = setup_capture(tmp_path)
    result = execute(value)
    assert result["snapshot"]["step"] == result["capture"]["frame_index"] == 88
    assert result["capture"]["current_endpoint"]["sim_state_sha256"] == result["snapshot"]["state_sha256"]
    assert value.case.sim.inner == 0


@pytest.mark.parametrize("mutation", ["endpoint", "state", "future"])
def test_rejects_mutable_endpoint_changed_state_or_future_image(tmp_path, mutation):
    value = setup_capture(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match="preserve the current endpoint"):
        execute(value)
    assert value.case.sim.inner == 0


def test_rejects_mutated_pose_and_different_recorded_state(tmp_path):
    value = setup_capture(tmp_path, mutation="pose")
    with pytest.raises(AssertionError):
        execute(value)
    value = setup_capture(tmp_path)
    value.options["expected_endpoint"]["sim_state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="recorded current state"):
        execute(value)
    assert np.isfinite(value.case.sim.data.time)
