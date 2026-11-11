"""Saved responses can be relayed only for their bound current observation."""

import copy

import pytest

from s2a.research.wam.tracking.runtime.recorded_reference_relay import RecordedReferenceRelay
from tests.research.native_runtime.test_streaming_continuation import fake_case, response_for


def setup_relay(tmp_path):
    case = fake_case(tmp_path, windows=1)
    current = case.ledger.current
    request = case.ledger.issue(before_render=current, after_render=current, image=case.image)
    response = response_for(request)
    record = {
        "step": current["step"],
        "sim_time_s": current["sim_time_s"],
        "state_sha256": current["state_sha256"],
        "image_sha256": case.image["sha256"],
        "response_sha256": "c" * 64,
        "historical_denoise_calls": 35,
        "action": response["action"],
        "normalized_action": response["normalized_action"],
    }
    return case, request, record


def test_historical_reference_remains_distinct_from_new_model_work(tmp_path):
    case, request, record = setup_relay(tmp_path)
    relay = RecordedReferenceRelay(case.service, [record])
    response, inputs = relay.respond(request)
    assert inputs == {"image_bytes": b"synthetic current image", "task": "place the object"}
    assert response["action"] == record["action"] and response["normalized_action"] == record["normalized_action"]
    assert relay.used == [
        {
            "request_id": 0,
            "step": 80,
            "historical_response_sha256": "c" * 64,
            "historical_denoise_calls": 35,
            "new_model_requests": 0,
            "new_denoise_calls": 0,
        }
    ]
    assert case.service.phase == "done"
    with pytest.raises(ValueError):
        relay.respond(request)
    assert relay.failed


@pytest.mark.parametrize("changed", ["request_id", "step", "sim_time_s", "state_sha256", "image_sha256"])
def test_stale_or_mismatched_binding_fails_before_reading_image(tmp_path, changed):
    case, request, record = setup_relay(tmp_path)
    relay = RecordedReferenceRelay(case.service, [record])
    wrong = copy.deepcopy(request)
    if changed == "image_sha256":
        wrong["image"]["sha256"] = "d" * 64
    elif changed == "state_sha256":
        wrong[changed] = "d" * 64
    else:
        wrong[changed] += 1
    (tmp_path / "frame.png").unlink()
    with pytest.raises(ValueError, match="saved reference binding"):
        relay.respond(wrong)
    assert relay.failed and case.service.admitted_requests == 0 and not relay.used


def test_changed_image_bytes_are_rejected_even_with_matching_metadata(tmp_path):
    case, request, record = setup_relay(tmp_path)
    relay = RecordedReferenceRelay(case.service, [record])
    (tmp_path / "frame.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        relay.respond(request)
    assert relay.failed and not relay.used


@pytest.mark.parametrize("changed", ["historical_denoise_calls", "action", "inactive", "response_sha256"])
def test_invalid_saved_roster_is_rejected_before_admission(tmp_path, changed):
    case, _request, record = setup_relay(tmp_path)
    if changed == "historical_denoise_calls":
        record[changed] = 34
    elif changed == "action":
        record[changed].pop()
    elif changed == "inactive":
        record["normalized_action"][2][12] = 1.0
    else:
        record[changed] = "bad"
    with pytest.raises(ValueError, match="registered causal roster"):
        RecordedReferenceRelay(case.service, [record])
    assert case.service.admitted_requests == 0
