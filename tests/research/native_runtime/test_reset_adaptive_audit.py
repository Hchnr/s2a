"""Score full task budgets and audit adapted real reset/model receipts."""

import copy

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_audit as audit

ROOT = prefix.REPO / "outputs/reset_seed43_live_second_window_20261113"


def saved_receipts():
    producer = prefix.read(prefix.reference(ROOT / "producer/simulation/result.json"))
    checker = prefix.read(prefix.reference(ROOT / "checker/simulation/result.json"))
    model = prefix.read(prefix.reference(ROOT / "producer/model/result.json"))
    bundle = prefix.read(prefix.reference(ROOT / "producer/model/registered_inputs.json"))
    # Convert the actual fixed two-window receipt into the equivalent adaptive
    # shape; the real measurements, responses, and image files stay unchanged.
    runtime, replay = producer["runtime"], checker["runtime"]
    original, checked = runtime["continuation"], replay["continuation"]
    stop = {"reason": "period_budget_exhausted", "at_step": 16, "next_action_dispatched": False}
    original["stream"]["stop"] = stop
    runtime["continuation"] = {"first_window": original["first_window"], "adaptive": original["stream"]}
    runtime["adaptive_captures"] = runtime["streaming_captures"]
    runtime["adaptive_model_exchanges"] = runtime["streaming_model_exchanges"]
    frame = {"version": 1, "kind": "stop", "session": bundle["model_registration"]["session"], "next_step": 16}
    runtime["adaptive_stop_frame"] = {"required": True, "send_attempted": True, "sent": True, "frame": frame}
    replay["continuation"] = {
        "first_window": checked["first_window"],
        "adaptive": {
            "segments": [{"continuation": checked["replay"]}],
            "reported_producer_stop": stop,
            "complete": True,
            "stop_reason_independently_verified": False,
            "control_steps_attempted": 8,
            "control_steps_completed": 8,
        },
    }
    replay["adaptive_checker_captures"] = [replay["checker_stream_capture"]]
    bundle["model_registration"]["max_requests"] = 2
    model["model_sequence_complete"] = False
    model["model_sequence_stop"] = {"frame": frame}
    config = dict(
        case_id=producer["case"]["case_id"],
        session=frame["session"],
        prefix_steps=8,
        max_adaptive_periods=8,
        max_control_steps_per_phase=16,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
        max_model_requests=2,
    )
    return producer, checker, model, bundle, config


@pytest.fixture(scope="module")
def recorded():
    return saved_receipts()


def test_saved_real_periods_images_and_model_response_reconcile(recorded):
    producer, checker, model, bundle, config = recorded
    made = audit.summarize_branch(producer, "producer", config)
    replayed = audit.summarize_branch(checker, "checker", config)
    assert made["control_steps"] == replayed["control_steps"] == 16
    assert made["native_steps"] == replayed["native_steps"] == 400
    assert made["model_requests"] == 1 and replayed["model_requests"] == 0
    assert not made["stable_last40"] and made["task_success_periods"] == 0
    assert all(value == 0 for value in audit.compare_branches(producer, checker).values())
    assert audit.verify_model_exchange(producer, model, bundle) == {"requests": 1, "denoise_calls": 35}


@pytest.mark.parametrize("change", ["stop", "seed", "image", "commands"])
def test_audit_rejects_changed_stops_or_cross_case_evidence(recorded, change):
    producer, checker, model, bundle, config = copy.deepcopy(recorded)
    if change == "stop":
        producer["runtime"]["continuation"]["adaptive"]["stop"]["at_step"] = 15
        with pytest.raises(ValueError, match="stop"):
            audit.summarize_branch(producer, "producer", config)
    elif change == "seed":
        model["sampling_seed"] = 42
        with pytest.raises(ValueError, match="seed"):
            audit.verify_model_exchange(producer, model, bundle)
    elif change == "image":
        checker["runtime"]["adaptive_checker_captures"][0]["canvas_png"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="PNG"):
            audit.compare_branches(producer, checker)
    else:
        checker["runtime"]["continuation"]["first_window"]["periods"][0]["command"][0] += 0.1
        with pytest.raises(ValueError):
            audit.compare_branches(producer, checker)


@pytest.mark.parametrize(
    ("flags", "stable", "tail"),
    [
        ([True] * 599, False, 599),
        ([False] * 560 + [True] * 40, True, 40),
        ([False] * 560 + [True] * 39 + [False], False, 0),
        ([True] * 600, True, 600),
    ],
)
def test_stability_requires_full_600_periods_and_last_40_consecutive(flags, stable, tail):
    result = audit.task_outcomes(flags)
    assert result["stable_last40"] is stable and result["consecutive_success_tail"] == tail
