"""Connect the adaptive control and model loops over their actual local channel."""

import threading
from types import SimpleNamespace

import pytest

from s2a.research.wam.tracking.runtime.adaptive_stream_control import close_adaptive_model_stream
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from scripts.wam.tracking.preview.adaptive_live_requests import serve_adaptive_requests
from tests.research.native_runtime.test_adaptive_stream_reference import run_driver, setup_adapter
from tests.research.native_runtime.test_streaming_continuation import response_for
from tests.research.wam.test_reset_live_request import Counter, Overrides, Torch
from tests.research.wam.test_reset_multitask_live_requests import Service


def test_control_and_real_request_loop_exchange_three_images_and_clean_stop(tmp_path):
    case, adapter, channel, model_channel, events, owner = setup_adapter(tmp_path, windows=4)
    old = case.ledger
    registration = {
        "session": old.session,
        "initial": old.current,
        "image_root": old.image_root,
        "task": "put the bowl on the plate",
        "seed": old.seed,
        "model_sha256": old.model_sha256,
        "max_requests": old.max_requests,
        "max_periods": old.max_periods,
        "denoise_steps": old.denoise_steps,
    }
    case.ledger = adapter.ledger = AdaptiveStreamLedger(**registration)
    directory = tmp_path / "model_loop"
    directory.mkdir()
    counter, evidence, model_result = Counter(), [], {}
    service = Service(registration["task"], counter, evidence)
    original_execute = service.execute

    def execute(sample, batch):
        response = original_execute(sample, batch).model_dump(mode="json")
        for row in response["outputs"][0]["content"]["action"]:
            row[3], row[7], row[9] = 1.0, 1.0, 0.7
        return SimpleNamespace(model_dump=lambda mode: response)

    service.execute = execute
    errors = []

    def model_loop():
        try:
            serve_adaptive_requests(
                registration=registration,
                channel=model_channel,
                image_root=tmp_path,
                directory=directory,
                template={},
                checkpoint={},
                service=service,
                torch=Torch,
                counter=counter,
                overrides=Overrides,
                domain_reader=lambda *_args: SimpleNamespace(is_libero=True),
                mask_evidence=evidence,
                result=model_result,
                request_builder=lambda _template, image, task: {
                    "name": f"synthetic_{image['step']}",
                    "vision_path": image["path"],
                    "task": task,
                },
                response_validator=lambda *_args, **_kwargs: None,
            )
        except BaseException as error:
            errors.append(error)
        finally:
            model_channel.close()

    thread = threading.Thread(target=model_loop)
    try:
        thread.start()
        stream = run_driver(case, adapter)
        close_adaptive_model_stream(channel, case.ledger, stream, case.receipt)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors
    assert stream["complete"] and stream["control_steps_completed"] == 12
    assert events == [("initial", 80, 0), ("advance", 83, 3), ("advance", 87, 7)]
    assert owner["completed"] == 12 and case.sim.inner == 300
    assert model_result["model_requests_attempted"] == model_result["responses_sent"] == 3
    assert model_result["denoise_calls_attempted"] == model_result["denoise_calls_completed"] == 105
    assert not model_result["model_sequence_complete"]
    assert model_result["model_sequence_stop"]["next_step"] == 92
    assert case.receipt["adaptive_stop_frame"]["sent"]


def test_stop_cannot_hide_an_unfinished_dispatched_action(tmp_path):
    case, _adapter, channel, model_channel, _events, _owner = setup_adapter(tmp_path)
    current = case.ledger.current
    request = case.ledger.issue(before_render=current, after_render=current, image=case.image)
    case.ledger.accept(response_for(request), after_wait=current)
    case.ledger.begin_period(before=current)
    case.ledger.abort()
    stream = {
        "stop": {"reason": "failed", "at_step": current["step"], "next_action_dispatched": False},
        "requests_issued": 1,
        "responses_accepted": 1,
        "control_steps_attempted": 1,
        "control_steps_completed": 0,
        "feedback_periods_committed": 0,
        "segments": [{"inflight": {"inner_completed": 3}}],
    }
    try:
        with pytest.raises(ValueError, match="completed dispatches"):
            close_adaptive_model_stream(channel, case.ledger, stream, case.receipt)
    finally:
        channel.close()
        model_channel.close()
    assert "adaptive_stop_frame" not in case.receipt and case.ledger.inflight == current


def test_no_stop_frame_after_final_model_response_budget(tmp_path):
    case, _adapter, channel, model_channel, _events, _owner = setup_adapter(tmp_path, windows=1)
    current = case.ledger.current
    request = case.ledger.issue(before_render=current, after_render=current, image=case.image)
    case.ledger.accept(response_for(request), after_wait=current)
    before = case.ledger.current
    case.ledger.begin_period(before=before)
    case.env.step([0.0] * 7)
    from s2a.research.native_runtime.streaming_continuation import measured_snapshot

    after = measured_snapshot(case.env, before["step"] + 1)
    case.ledger.complete_period(before=before, after=after, inner_samples=25)
    assert not case.ledger.refresh_early(trigger={"replan": True, "current_step": 81, "reason": "quality"})
    stream = {
        "stop": {"reason": "request_budget_exhausted", "at_step": 81, "next_action_dispatched": False},
        "requests_issued": 1,
        "responses_accepted": 1,
        "control_steps_attempted": 1,
        "control_steps_completed": 1,
        "feedback_periods_committed": 1,
        "segments": [{"inflight": None}],
    }
    try:
        close_adaptive_model_stream(channel, case.ledger, stream, case.receipt)
    finally:
        channel.close()
        model_channel.close()
    assert case.receipt["adaptive_stop_frame"] == {"required": False, "reason": "model_request_budget_exhausted"}
