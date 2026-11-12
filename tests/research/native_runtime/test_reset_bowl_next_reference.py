"""A new H40 starts only after the verified 88-period history and image."""

import copy

import pytest

from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import reset_bowl_next_reference as next_reference
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint
from tests.research.native_runtime.test_reset_bowl_reference_tail import run_longer, setup_longer
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)


def setup_next(tmp_path, *, stop_at=None):
    original_dir, next_dir = tmp_path / "original88", tmp_path / "next"
    original_dir.mkdir()
    next_dir.mkdir()
    case, receipt, options, _, _, previous56 = setup_longer(original_dir)
    first = run_longer(case, receipt, options, previous56)
    previous88 = {"runtime": {**receipt, "continuation": first}}
    sample, endpoint = case.observers.endpoint(case.env, 88)
    capture = case.observers.render_anchor(
        case.env, 88, sample, minimal_renderer_endpoint(endpoint, 88), options["runtime_output"]
    )
    current = measured_snapshot(case.env, 88)
    response, old_envelope = options["load_second_reference"]()
    envelope = {
        **copy.deepcopy(old_envelope),
        **current,
        "image_sha256": capture["canvas_png"]["sha256"],
        "seed": 42,
        "model_sha256": next_reference.MODEL_SHA,
    }
    case, receipt, options, events, first_reference, previous56 = setup_longer(next_dir, stop_at=stop_at)
    options.update(
        previous_56_producer=previous56,
        previous_88_producer=previous88,
        expected_step88_capture=capture,
        load_step88_reference=lambda: (copy.deepcopy(response), copy.deepcopy(envelope)),
    )
    return case, receipt, options, events, first_reference


def run_next(case, receipt, options):
    return next_reference.run_producer(case.env, terminal_at(case, 0), case.observers, receipt, **options)


def test_executes_new_h40_with_same_history_and_independently_replays(tmp_path):
    case, receipt, options, events, first_reference = setup_next(tmp_path)
    result = run_next(case, receipt, options)
    assert case.sim.inner == 128 * 25 and receipt["control_steps_completed"] == 128
    assert result["new_reference"]["complete"] and result["new_reference"]["generated_periods"] == 40
    assert [event[1] for event in events if event[0] == "advance"] == list(range(8, 121, 8))

    checker_dir = tmp_path / "checker"
    checker_dir.mkdir()
    checker = fake_case(checker_dir, windows=15)
    checker.observers.render_anchor = case.observers.render_anchor
    checker_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    producer = {"runtime": {**receipt, "continuation": result}}
    replayed = next_reference.run_checker(
        checker.env,
        terminal_at(checker, 0),
        checker.observers,
        checker_receipt,
        producer=producer,
        case=options["case"],
        expected_capture=options["expected_capture"],
        load_reference=lambda: first_reference,
        runtime_output=checker_dir,
        expected_second_capture=options["expected_second_capture"],
        load_second_reference=options["load_second_reference"],
        expected_step88_capture=options["expected_step88_capture"],
        load_step88_reference=options["load_step88_reference"],
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
        gripper_score=score,
        render_scope=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
    )
    assert checker.sim.inner == 128 * 25 and checker_receipt["newly_selected_commands"] == 0
    assert replayed["new_reference"]["generated_periods"] == 40
    assert replayed["new_reference"]["stop_recomputed"] is False


def test_changed_new_reference_cannot_dispatch_after_step88(tmp_path):
    case, receipt, options, _, _ = setup_next(tmp_path)
    response, envelope = options["load_step88_reference"]()
    envelope["image_sha256"] = "a" * 64
    options["load_step88_reference"] = lambda: (response, envelope)
    with pytest.raises(ValueError, match="match the current step-88 image"):
        run_next(case, receipt, options)
    assert case.sim.inner == 88 * 25 and receipt["control_steps_completed"] == 88


def test_new_reference_stop_keeps_only_dispatched_periods(tmp_path):
    case, receipt, options, _, _ = setup_next(tmp_path, stop_at=103)
    result = run_next(case, receipt, options)
    assert case.sim.inner == 103 * 25 and receipt["control_steps_completed"] == 103
    assert result["new_reference"]["stop"]["reason"] == "quality_no_witness"
    assert result["new_reference"]["stop"]["at_step"] == 103
    assert result["new_reference"]["generated_periods"] == 15
