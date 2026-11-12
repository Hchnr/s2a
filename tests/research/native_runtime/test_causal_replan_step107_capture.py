import copy

from s2a.research.native_runtime.recorded_reference_prefix import capture_tail_prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime.run_causal_replan_step107_capture import capture_after_checker
from tests.research.native_runtime.test_reset_bowl_next_reference import run_next, setup_next
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


def test_capture_replays_only_the_registered_prefix_before_render(tmp_path, monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)
    case, receipt, options, _events, first_reference = setup_next(tmp_path, stop_at=108)
    result = run_next(case, receipt, options)
    assert case.sim.inner == 108 * 25
    expected = reference.prefix_periods(result["prefix88"]) + tail.tail_periods(result["new_reference"])
    producer = {"runtime": {**receipt, "continuation": result}}
    derived = copy.deepcopy(producer)
    derived["runtime"]["continuation"]["new_reference"] = capture_tail_prefix(
        result["new_reference"], reference_start=88, capture_step=107
    )

    checker_dir = tmp_path / "checker107"
    checker_dir.mkdir()
    checker = fake_case(checker_dir, windows=15)
    checker.observers.render_anchor = case.observers.render_anchor
    checked_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    replayed = capture_after_checker(
        checker.env,
        terminal_at(checker, 0),
        checker.observers,
        checked_receipt,
        checker=reference.run_checker,
        expected_periods=expected[:107],
        step=107,
        producer=derived,
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
    assert checker.sim.inner == 107 * 25 and checked_receipt["control_steps_completed"] == 107
    assert checked_receipt["newly_selected_commands"] == 0
    assert replayed["new_reference"]["stop"]["reason"] == "requested_observation"
    assert checked_receipt["terminal_capture"]["snapshot"]["step"] == 107
    assert result["new_reference"]["stop"]["reason"] == "quality_no_witness"
