"""Rebuild a 56-period history before executing a saved reference's tail."""

import copy

import pytest

from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from tests.research.native_runtime.test_reset_bowl_h40_second_window import produce, setup_second
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)


def setup_longer(tmp_path, *, stop_at=None):
    original_dir, next_dir = tmp_path / "original56", tmp_path / "longer"
    original_dir.mkdir()
    next_dir.mkdir()
    case, receipt, options, _, _ = setup_second(original_dir)
    original = produce(case, receipt, options)
    previous = {"runtime": {**receipt, "continuation": original}}
    case, receipt, options, events, first_reference = setup_second(next_dir, stop_at=stop_at)
    return case, receipt, options, events, first_reference, previous


def run_longer(case, receipt, options, previous):
    return tail.run_producer(
        case.env,
        terminal_at(case, 0),
        case.observers,
        receipt,
        previous_56_producer=previous,
        **options,
    )


def test_preserves_history_to_the_last_generated_endpoint_and_independently_replays(tmp_path):
    case, receipt, options, events, first_reference, previous = setup_longer(tmp_path)
    result = run_longer(case, receipt, options, previous)
    assert case.sim.inner == 88 * 25 and receipt["control_steps_completed"] == 88
    assert result["tail"]["complete"] and result["tail"]["generated_periods"] == 32
    assert [event[1] for event in events if event[0] == "advance"] == list(range(8, 81, 8))

    checker_dir = tmp_path / "checker"
    checker_dir.mkdir()
    checker = fake_case(checker_dir, windows=10)
    checker.observers.render_anchor = case.observers.render_anchor
    checker_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    producer = {"runtime": {**receipt, "continuation": result}}
    replayed = tail.run_checker(
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
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
        gripper_score=score,
        render_scope=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
    )
    assert checker.sim.inner == 88 * 25 and checker_receipt["newly_selected_commands"] == 0
    assert len(tail.tail_periods(replayed["tail"])) == 32 and replayed["tail"]["stop_recomputed"] is False


def test_changed_56_period_prefix_cannot_dispatch_a_tail_action(tmp_path):
    case, receipt, options, _, _, previous = setup_longer(tmp_path)
    changed = copy.deepcopy(previous)
    changed["runtime"]["continuation"]["second"]["periods"][-1]["command"][0] = 0.1
    with pytest.raises(ValueError, match="differ in command"):
        run_longer(case, receipt, options, changed)
    assert case.sim.inner == 56 * 25 and receipt["control_steps_completed"] == 56


def test_tail_stop_keeps_actual_dispatches_without_consuming_future_targets(tmp_path):
    case, receipt, options, _, _, previous = setup_longer(tmp_path, stop_at=73)
    result = run_longer(case, receipt, options, previous)
    assert case.sim.inner == 73 * 25 and receipt["control_steps_completed"] == 73
    assert result["tail"]["stop"]["reason"] == "quality_no_witness"
    assert result["tail"]["stop"]["at_step"] == 73
    assert len(tail.tail_periods(result["tail"])) == 17
