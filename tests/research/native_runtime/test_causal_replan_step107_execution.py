import copy

import numpy as np
import pytest

from s2a.research.native_runtime.recorded_reference_prefix import capture_tail_prefix
from scripts.simulation.runtime import causal_replan_step107_execution as replan
from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime.run_causal_replan_step107_capture import capture_after_checker
from tests.research.native_runtime.test_reset_bowl_next_reference import run_next, setup_next
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)


def setup_capture(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    case, receipt, options, _events, first_reference = setup_next(baseline_dir, stop_at=108)
    result = run_next(case, receipt, options)
    periods = reference.prefix_periods(result["prefix88"]) + tail.tail_periods(result["new_reference"])
    producer = {"runtime": {**receipt, "continuation": result}}
    derived = copy.deepcopy(producer)
    derived["runtime"]["continuation"]["new_reference"] = capture_tail_prefix(
        result["new_reference"], reference_start=88, capture_step=107
    )
    checker_dir = tmp_path / "capture"
    checker_dir.mkdir()
    checker = fake_case(checker_dir, windows=15)
    checker.observers.render_anchor = case.observers.render_anchor
    checked = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    capture_after_checker(
        checker.env,
        terminal_at(checker, 0),
        checker.observers,
        checked,
        checker=reference.run_checker,
        expected_periods=periods[:107],
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
    captured = checked["terminal_capture"]
    raw, envelope = options["load_step88_reference"]()
    envelope = {
        **copy.deepcopy(envelope),
        **captured["snapshot"],
        "image_sha256": captured["capture"]["canvas_png"]["sha256"],
        "seed": 42,
        "model_sha256": replan.MODEL_SHA,
    }
    return producer, captured["capture"], (raw, envelope)


def test_replans_from_completed_partial_segment_and_replays_actual_actions(tmp_path):
    prior, captured, model_response = setup_capture(tmp_path)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    case, receipt, options, _events, first_reference = setup_next(candidate_dir)
    factory, old_advance = options.pop("policy_factory"), options.pop("advance_policy")
    switches = []

    class NewPolicy:
        def __init__(self, start):
            self.start = self.completed = start

        def select(self, _q, _limits, _jacobian, _previous, offset):
            assert self.completed - self.start == offset
            return {"command": np.zeros(7), "newly_selected": True}

        def complete(self, _env, _command, _offset, inner):
            assert len(inner) == 25
            self.completed += 1
            return {"completed": self.completed}

    def advance(previous, owner, *args, generated_step, runtime_receipt):
        if generated_step < 107:
            return old_advance(previous, owner, *args, generated_step=generated_step, runtime_receipt=runtime_receipt)
        assert previous.completed == generated_step
        switches.append(generated_step)
        return NewPolicy(generated_step)

    def trigger(_period, *, current_step):
        return {"replan": current_step == 107, "reason": "quality_margin_consumed", "current_step": current_step}

    result = replan.run_producer(
        case.env,
        terminal_at(case, 0),
        case.observers,
        receipt,
        prior_producer=prior,
        expected_replan_capture=captured,
        load_new_reference=lambda: model_response,
        prefix_runner=reference.run_producer,
        policy_factory=factory,
        advance_policy=advance,
        partial_advance=advance,
        trigger=trigger,
        **options,
    )
    assert case.sim.inner == 147 * 25 and receipt["control_steps_completed"] == 147
    assert switches == [107, 115, 123, 131, 139]
    assert result["prefix107"]["new_reference"]["stop"]["at_step"] == 107
    assert result["replanned"]["complete"] and result["replanned"]["generated_periods"] == 40

    checker_dir = tmp_path / "replay"
    checker_dir.mkdir()
    checker = fake_case(checker_dir, windows=19)
    checker.observers.render_anchor = case.observers.render_anchor
    checked = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    producer = {"runtime": {**receipt, "continuation": result}}
    replayed = replan.run_checker(
        checker.env,
        terminal_at(checker, 0),
        checker.observers,
        checked,
        producer=producer,
        expected_replan_capture=captured,
        load_new_reference=lambda: model_response,
        prefix_runner=reference.run_checker,
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
    assert checker.sim.inner == 147 * 25 and checked["newly_selected_commands"] == 0
    assert replayed["replanned"]["complete"] and replayed["replanned"]["stop_recomputed"] is False
