"""Audit a previously measured physical replay without running physics again."""

import copy

import pytest

from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_feedback_audit as audit

PRODUCER = {
    "path": str(prefix.REPO / "outputs/reset_live_second_window_20261111/producer/simulation/result.json"),
    "sha256": "37a4ad8399b46f236a70503aff993c2c9173c0442948d6bb3236b0423cc17ed6",
}
CHECKER = {
    "path": str(prefix.REPO / "outputs/reset_live_checker_repair_20261111/simulation/result.json"),
    "sha256": "2c90bb990f5a484f0298ddb6b0891ad2b97d019847d5b0046f3ca7bc55f28757",
}
MODEL = {
    "path": str(prefix.REPO / "outputs/reset_live_second_window_20261111/producer/model/result.json"),
    "sha256": "ced2c6970c170a8ed16b30bf26e686ad3d0d281f2a582159a60935d1a4316415",
}
BUNDLE = {
    "path": str(prefix.REPO / "outputs/reset_live_second_window_20261111/producer/model/registered_inputs.json"),
    "sha256": "8b17585dfb9811c4e377b4ca50285be6b8787bb82d42d384eadbd320b5a5ea86",
}
TOOL = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


@pytest.fixture(scope="module")
def recorded():
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    config = dict(
        case_id=producer["case"]["case_id"],
        first_window_steps=8,
        live_window_steps=8,
        max_control_steps_per_phase=16,
        max_native_steps_per_phase=400,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
    )
    return producer, checker, config


def test_actual_historical_receipts_match_without_new_physics(recorded):
    producer, checker, config = recorded
    counts = audit.summarize_branch(producer, "producer", config)
    checked = audit.summarize_branch(checker, "checker", config, producer=producer)
    difference = audit.compare_branches(producer, checker)
    assert counts["control_steps"] == checked["control_steps"] == 16
    assert counts["native_steps"] == checked["native_steps"] == 400
    assert counts["models"] == counts["contexts"] == 3
    assert counts["live_control_steps"] == 8 and counts["model_requests"] == 1
    assert counts["task_success_periods"] == counts["live_task_success_periods"] == 0
    assert all(value == 0 for value in difference.values())


@pytest.mark.parametrize("change", ["command", "task_success", "image", "stop"])
def test_independent_audit_rejects_changed_evidence(recorded, change):
    producer, original_checker, _config = recorded
    checker = copy.deepcopy(original_checker)
    if change == "command":
        checker["runtime"]["continuation"]["first_window"]["periods"][0]["command"][0] += 0.1
    elif change == "task_success":
        checker["runtime"]["continuation"]["replay"]["periods"][0]["task_success"] = True
    elif change == "image":
        checker["runtime"]["checker_stream_capture"]["canvas_png"]["sha256"] = "0" * 64
    else:
        checker["runtime"]["continuation"]["stop_reason_independently_verified"] = True
    with pytest.raises(ValueError):
        audit.compare_branches(producer, checker)


def test_mismatched_case_or_physical_limit_is_rejected(recorded):
    producer, _checker, config = recorded
    for change in (dict(case_id="different"), dict(max_native_steps_per_phase=401)):
        with pytest.raises(ValueError, match="registered case"):
            audit.summarize_branch(producer, "producer", dict(config, **change))


def test_pre_request_stop_preserves_empty_live_window(recorded):
    original_producer, original_checker, config = recorded
    producer, checker = copy.deepcopy(original_producer), copy.deepcopy(original_checker)
    stop = dict(reason="synthetic_pre_request_stop", at_step=8, next_action_dispatched=False)
    runtime = producer["runtime"]
    stream = runtime["continuation"]["stream"]
    stream.update(
        segments=[],
        requests_issued=0,
        responses_accepted=0,
        control_steps_attempted=0,
        control_steps_completed=0,
        feedback_periods_committed=0,
        complete=False,
        stop=stop,
    )
    runtime["streaming_captures"] = []
    runtime["segment_receipts"] = runtime["segment_receipts"][:1]
    runtime["newly_selected_commands"] = 8
    replay = checker["runtime"]["continuation"]
    replay.update(replay=None, reported_producer_stop=stop)
    checker["runtime"].pop("checker_stream_capture")
    for item in (runtime, checker["runtime"]):
        item["control_steps_attempted"] = item["control_steps_completed"] = 8
        item["counters"]["mj_step"] = dict(attempted=200, completed=200)
    counts = audit.summarize_branch(producer, "producer", config)
    checked = audit.summarize_branch(checker, "checker", config, producer=producer)
    assert counts["control_steps"] == checked["control_steps"] == 8
    assert counts["model_requests"] == counts["live_control_steps"] == checked["live_control_steps"] == 0
    assert all(value == 0 for value in audit.compare_branches(producer, checker).values())


def test_saved_model_response_matches_actual_current_image_and_envelope(recorded):
    producer, _checker, _config = recorded
    model, bundle = prefix.read(MODEL), prefix.read(BUNDLE)
    model["model_sequence_complete"] = True  # Older worker predated this explicit field.
    assert audit.verify_model_exchange(producer, model, bundle, TOOL) == {"requests": 1, "denoise_calls": 35}


@pytest.mark.parametrize("change", ["seed", "sent", "denoise", "image"])
def test_model_exchange_rejects_mismatched_identity_or_evidence(recorded, change):
    original_producer, _checker, _config = recorded
    producer = copy.deepcopy(original_producer)
    model, bundle = prefix.read(MODEL), prefix.read(BUNDLE)
    model["model_sequence_complete"] = True
    if change == "seed":
        bundle["model_registration"]["seed"] = 43
    elif change == "sent":
        model["responses"][0]["reference_sent"] = False
    elif change == "denoise":
        model["denoise_evidence"].pop()
    else:
        producer["runtime"]["streaming_captures"][0]["canvas_png"]["sha256"] = "0" * 64
    with pytest.raises((ValueError, AssertionError)):
        audit.verify_model_exchange(producer, model, bundle, TOOL)


def test_model_pre_request_stop_is_not_counted_as_sampling(recorded):
    original_producer, _checker, _config = recorded
    producer = copy.deepcopy(original_producer)
    model, bundle = prefix.read(MODEL), prefix.read(BUNDLE)
    runtime = producer["runtime"]
    stream = runtime["continuation"]["stream"]
    stream.update(requests_issued=0, responses_accepted=0, stop=dict(reason="synthetic", next_action_dispatched=False))
    runtime.update(streaming_captures=[], streaming_model_exchanges=[], model_stop_sent=True)
    model.update(
        responses=[],
        responses_sent=0,
        local_requests_received=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        denoise_calls_attempted=0,
        denoise_calls_completed=0,
        denoise_evidence=[],
        mask_evidence=[],
        model_sequence_complete=False,
        model_sequence_stop=make_stop(session=bundle["model_registration"]["session"], next_step=8),
    )
    assert audit.verify_model_exchange(producer, model, bundle, TOOL) == {"requests": 0, "denoise_calls": 0}
    model["model_sequence_stop"]["next_step"] = 9
    with pytest.raises(ValueError, match="pre-request stop"):
        audit.verify_model_exchange(producer, model, bundle, TOOL)
