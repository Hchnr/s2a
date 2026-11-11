"""Exercise six current-image requests and explicit stops with a fake service."""

import copy
import hashlib
from types import SimpleNamespace

import pytest

from scripts.wam.tracking.preview import live_stream_requests_48 as loop
from tests.research.wam.test_live_stream_requests import FakeOverrides, FakeTorch, setup_loop


def run_loop(items, *, max_requests=None):
    inputs, image_root, directory, channel, counter, evidence, result, service = items
    loop.execute_live_requests(
        inputs,
        channel=channel,
        image_root=image_root,
        directory=directory,
        service=service,
        torch=FakeTorch,
        counter=counter,
        overrides=FakeOverrides,
        domain_reader=lambda *_args: SimpleNamespace(is_libero=True),
        mask_evidence=evidence,
        result=result,
        max_requests=max_requests,
    )


def setup_six_windows(tmp_path):
    items = setup_loop(tmp_path)
    inputs, _root, _directory, channel, _counter, _evidence, _result, service = items
    inputs["live_config"] = dict(inputs["live_config"], max_model_requests=6, max_windows=6)
    first, second = channel.requests
    for index in range(2, 6):
        current = copy.deepcopy(second)
        current.update(
            request_id=index,
            step=80 + 8 * index,
            sim_time_s=first["sim_time_s"] + 0.4 * index,
            state_sha256=hashlib.sha256(f"actual-current-{index}".encode()).hexdigest(),
        )
        channel.requests.append(current)
    original_execute = service.execute

    def execute(sample, batch):
        result = original_execute(sample, batch)
        if service.request["name"] not in ("s2a_correct08_t80_s42", "s2a_correct08_t88_s42"):
            response = result.model_dump(mode="json")
            response["outputs"][0]["content"]["action"][0][0] += 0.01
            return SimpleNamespace(model_dump=lambda mode: response)
        return result

    service.execute = execute
    return items


def test_six_window_loop_gates_only_the_first_two_historical_outputs(tmp_path):
    items = setup_six_windows(tmp_path)
    run_loop(items, max_requests=6)
    inputs, _root, _directory, channel, _counter, evidence, result, service = items
    assert result["model_sequence_complete"] and result["model_sequence_stop"] is None
    assert result["model_requests_attempted"] == result["responses_sent"] == service.prepare_calls == 6
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 210
    assert len(channel.responses) == len(evidence) == 6
    assert all(row["comparison"] is not None for row in result["responses"][:2])
    assert all(row["comparison"] is row["historical_response_sha256"] is None for row in result["responses"][2:])
    assert channel.responses[2]["action"][0][0] != inputs["records"][1]["action"][0][0]
    assert [response["step"] for response in channel.responses] == [80, 88, 96, 104, 112, 120]


@pytest.mark.parametrize(("requests", "stop_step"), [(0, 80), (1, 83), (2, 96), (3, 101), (5, 119)])
def test_explicit_stop_preserves_actual_model_work_and_sends_no_more(tmp_path, requests, stop_step):
    from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop

    items = setup_six_windows(tmp_path)
    channel, result = items[3], items[6]
    session = channel.requests[0]["session"]
    channel.requests = channel.requests[:requests] + [make_stop(session=session, next_step=stop_step)]
    run_loop(items, max_requests=6)
    assert not result["model_sequence_complete"]
    assert result["model_sequence_stop"]["next_step"] == stop_step
    assert (
        result["local_requests_received"] == result["model_requests_completed"] == result["responses_sent"] == requests
    )
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 35 * requests
    assert len(channel.responses) == requests


@pytest.mark.parametrize("corruption", ["session", "step", "boolean_step", "version", "future_label"])
def test_invalid_stop_frame_cannot_end_the_model_loop(tmp_path, corruption):
    from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop

    items = setup_six_windows(tmp_path)
    channel, result = items[3], items[6]
    stop = make_stop(session=channel.requests[0]["session"], next_step=80)
    if corruption == "session":
        stop["session"] = "wrong"
    elif corruption == "step":
        stop["next_step"] = 81
    elif corruption == "boolean_step":
        stop["next_step"] = True
    elif corruption == "version":
        stop["version"] = True
    else:
        stop["future_label"] = "success"
    channel.requests = [stop]
    with pytest.raises(ValueError, match="Stop frame"):
        run_loop(items, max_requests=6)
    assert result["model_requests_attempted"] == result["responses_sent"] == 0
    assert result["model_sequence_stop"] is None


def test_six_window_budget_cannot_be_implicitly_expanded(tmp_path):
    items = setup_six_windows(tmp_path)
    with pytest.raises(ValueError, match="six-window registration"):
        run_loop(items, max_requests=7)
    assert len(items[3].requests) == 6 and items[7].prepare_calls == 0
