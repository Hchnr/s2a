"""Decode actual saved exchanges without loading a candidate policy."""

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.simulation.runtime import causal_stream_live_48_replay as replay
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import causal_stream_live_48_admission as admission


def actual_prior():
    inputs = admission.load_inputs()
    summary = prefix.read(inputs["long_refs"]["prior_live_producer"])
    branch = prefix.read(summary["branch_ref"])
    root = Path(summary["branch_ref"]["path"]).parent / "runtime"
    return inputs, branch, root


def test_saved_live_exchanges_decode_to_the_audited_two_reference_chunks():
    inputs, branch, root = actual_prior()
    chunks, captures = replay.decode_actual_references(inputs, branch, image_root=root, max_windows=2)
    assert len(chunks) == len(captures) == 2
    for index, chunk in enumerate(chunks):
        np.testing.assert_allclose(chunk, inputs["old"]["targets"][index][:8], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(
            captures[index]["source_pose"], branch["runtime"]["streaming_captures"][index]["current_source_pose"]
        )


@pytest.mark.parametrize("field", ["image", "endpoint", "exchange", "period"])
def test_changed_capture_or_physical_endpoint_is_rejected(field):
    inputs, original, root = actual_prior()
    branch = copy.deepcopy(original)
    runtime = branch["runtime"]
    if field == "image":
        runtime["streaming_captures"][0]["canvas_png"]["sha256"] = "a" * 64
    elif field == "endpoint":
        runtime["streaming_captures"][0]["current_endpoint"]["sim_time"] += 0.1
    elif field == "exchange":
        runtime["streaming_model_exchanges"][0]["response_accepted"] = False
    else:
        runtime["continuation"]["segments"][0]["continuation"]["periods"][0]["actual_state"][0] += 1
    with pytest.raises(ValueError):
        replay.decode_actual_references(inputs, branch, image_root=root, max_windows=2)


def test_zero_request_stop_decodes_no_targets():
    inputs, branch, root = actual_prior()
    runtime = branch["runtime"]
    runtime["streaming_captures"] = []
    runtime["streaming_model_exchanges"] = []
    runtime["continuation"] = {
        "segments": [],
        "requests_issued": 0,
        "responses_accepted": 0,
        "feedback_periods_committed": 0,
        "complete": False,
        "stop": {"reason": "observed_prefix_joint_buffer_loss", "at_step": 80, "next_action_dispatched": False},
    }
    assert replay.decode_actual_references(inputs, branch, image_root=root, max_windows=6) == ([], [])
