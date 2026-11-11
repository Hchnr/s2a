"""A richer checker receipt must render under the producer's endpoint schema."""

import copy

import pytest

from scripts.simulation.runtime.reset_live_two_window_replay import compare_capture
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint
from tests.research.native_runtime.test_reset_live_two_window_replay import capture


def test_renderer_receives_the_actual_identity_not_additional_checker_metadata():
    producer = capture()
    checker = copy.deepcopy(producer)
    actual = dict(checker["current_endpoint"], sim_state_dtype="float64", native_success=False, motor_sha256="f" * 64)
    checker["current_endpoint"] = minimal_renderer_endpoint(actual, 8)
    compare_capture(producer, checker)
    assert checker["current_endpoint"]["sim_state_sha256"] == actual["sim_state_sha256"]
    assert actual["native_success"] is False
    actual["sim_state_sha256"] = "d" * 64
    checker["current_endpoint"] = minimal_renderer_endpoint(actual, 8)
    with pytest.raises(ValueError, match="current image or endpoint"):
        compare_capture(producer, checker)
    with pytest.raises(ValueError, match="matching step"):
        minimal_renderer_endpoint(actual, 9)
