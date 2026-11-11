"""CPU admission and transport tests; no simulator or model is launched."""

import copy
import os
import shutil
import socket
import subprocess
import sys
import threading

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.simulation.runtime import causal_stream_recorded_admission as admission
from scripts.simulation.runtime import run_causal_stream_recorded_physical as runner


def test_recorded_reference_service_exchanges_both_actual_saved_responses(tmp_path):
    inputs = admission.load_inputs()
    images = inputs["old"]["images"]
    for image in images:
        directory = tmp_path / f"anchor{image['frame_index']:04d}"
        directory.mkdir()
        shutil.copyfile(image["canvas_png"]["path"], directory / "current.png")
    sim, model = socket.socketpair()
    channel = FramedJSONChannel(sim, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    service_channel = FramedJSONChannel(model, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    ledger = CausalStreamLedger(**runner.model_config(inputs, tmp_path))
    result = {"local_reference_requests": 0, "responses_sent": 0, "records": []}
    errors = []

    def serve():
        try:
            runner.serve_recorded(service_channel, inputs, result, image_root=tmp_path)
        except BaseException as error:
            errors.append(error)
        finally:
            service_channel.close()

    thread = threading.Thread(target=serve)
    try:
        thread.start()
        for index, record in enumerate(inputs["records"]):
            current = ledger.current
            image = {"path": f"anchor{record['step']:04d}/current.png", "sha256": record["image_sha256"]}
            request = ledger.issue(before_render=current, after_render=current, image=image)
            channel.send(request)
            response = channel.receive()
            assert admission.gate_response(inputs["records"], request, response)
            np.testing.assert_array_equal(ledger.accept(response, after_wait=ledger.current), record["action"])
            for offset in range(8):
                before = ledger.current
                after = {
                    "step": before["step"] + 1,
                    "sim_time_s": before["sim_time_s"] + 0.05,
                    "state_sha256": "a" * 64,
                }
                if index == 0 and offset == 7:
                    after["sim_time_s"] = inputs["records"][1]["sim_time_s"]
                    after["state_sha256"] = inputs["records"][1]["state_sha256"]
                ledger.complete_period(before=before, after=after, inner_samples=25)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors
    assert ledger.phase == "done" and result["local_reference_requests"] == result["responses_sent"] == 2
    assert [row["historical_response_sha256"] for row in result["records"]] == [
        record["response_sha256"] for record in inputs["records"]
    ]
    assert all(row["new_model_requests"] == row["new_denoise_calls"] == 0 for row in result["records"])


def converted_branch(branch, *, producer):
    branch = copy.deepcopy(branch)
    runtime = branch["runtime"]
    periods = runtime["continuation"]["periods"]
    runtime["continuation"] = {
        "complete": True,
        "segments": [
            {"continuation": {"periods": periods[:8]}},
            {"continuation": {"periods": periods[8:]}},
        ],
        "requests_issued": 2,
        "responses_accepted": 2,
        "feedback_periods_committed": 16,
    }
    if producer:
        runtime["streaming_captures"] = runtime["current_wam_images"]
        runtime["streaming_model_exchanges"] = [
            {"request_sent": True, "response_received": True, "response_accepted": True, "policy_ready": True}
            for _ in range(2)
        ]
    return branch


def test_both_phase_summaries_require_complete_safe_recorded_periods():
    inputs = admission.load_inputs()
    producer = converted_branch(inputs["producer"], producer=True)
    checker = converted_branch(inputs["checker"], producer=False)
    for branch, phase in ((producer, "producer"), (checker, "checker")):
        assert runner.summarize_branch(branch, phase=phase) == {
            "control_steps": 96,
            "native_steps": 2400,
            "continuation_steps": 16,
            "prefix_inner_samples": 2000,
            "continuation_inner_samples": 400,
        }
    producer["runtime"]["continuation"]["segments"][1]["continuation"]["periods"][0]["inner_below_buffer_samples"] = 1
    with pytest.raises(ValueError, match=r"safe 80\+16"):
        runner.summarize_branch(producer, phase="producer")


def test_model_registration_binds_only_the_recorded_history_and_current_capture_root(tmp_path):
    inputs = admission.load_inputs()
    config = runner.model_config(inputs, tmp_path)
    assert config["initial"] == {
        "step": 80,
        "sim_time_s": inputs["records"][0]["sim_time_s"],
        "state_sha256": inputs["records"][0]["state_sha256"],
    }
    assert config["max_requests"] == 2 and config["denoise_steps"] == 35
    assert config["model_sha256"] == inputs["model_sha256"]
    assert runner.source_refs()


def test_checker_entry_imports_no_candidate_policy():
    code = """
from scripts.simulation.runtime import run_causal_stream_recorded_physical as runner
from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation
runner.prior.checker_isolation()
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
