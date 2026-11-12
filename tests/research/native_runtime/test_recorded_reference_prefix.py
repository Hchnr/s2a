import copy

import numpy as np
import pytest

from s2a.research.native_runtime.recorded_reference_prefix import capture_tail_prefix
from tests.research.native_runtime.test_finite_reference_tail import execute, setup_tail
from tests.research.native_runtime.test_reset_live_two_window import score


def test_planned_observation_replays_only_the_completed_prefix(tmp_path):
    producer_dir, checker_dir = tmp_path / "producer", tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    producer, terminal, receipt, options, _ = setup_tail(producer_dir, stop_at=75)
    original = execute(producer, terminal, receipt, options)
    prefix = capture_tail_prefix(original, reference_start=48, capture_step=73)
    assert [len(row["continuation"]["periods"]) for row in original["segments"]] == [8, 8, 3]
    assert [len(row["continuation"]["periods"]) for row in prefix["segments"]] == [8, 8, 1]
    assert original["stop"]["reason"] == "quality_no_witness"
    assert prefix["stop"] == {"reason": "requested_observation", "at_step": 73, "next_action_dispatched": False}
    assert prefix["generated_periods"] == 17

    checker, terminal, checked_receipt, checked_options, _ = setup_tail(checker_dir)

    def open_replay(_poses, openings, _anchor, _step, expected):
        def choose(_q, _limits, _jacobian, _previous, offset):
            return {"command": np.asarray(expected[offset]["command"]), "newly_selected": False}

        def complete(env, _command, offset):
            measured = score(env, openings[offset])
            assert measured == expected[offset]["completed_feedback"]["score"]
            return {"score": measured, "policy_history": None}

        return choose, complete

    checked_options.update(open_segment=open_replay, recorded_segments=prefix["segments"], recorded_stop=prefix["stop"])
    replayed = execute(checker, terminal, checked_receipt, checked_options)
    assert checker.sim.inner == 73 * 25 and checked_receipt["control_steps_completed"] == 73
    assert checked_receipt["newly_selected_commands"] == 0 and replayed["stop"] == prefix["stop"]
    assert replayed["stop_recomputed"] is False


@pytest.mark.parametrize("corruption", ["step", "clock", "stop", "completed", "capture"])
def test_planned_observation_rejects_unrecorded_or_inconsistent_actions(tmp_path, corruption):
    producer, terminal, receipt, options, _ = setup_tail(tmp_path, stop_at=75)
    original = execute(producer, terminal, receipt, options)
    changed = copy.deepcopy(original)
    capture = 73
    if corruption == "step":
        changed["segments"][1]["continuation"]["periods"][0]["step"] += 1
    elif corruption == "clock":
        changed["segments"][1]["reference_offset"] += 8
    elif corruption == "stop":
        changed["stop"]["at_step"] += 1
    elif corruption == "completed":
        changed["segments"][0]["continuation"]["steps_completed"] -= 1
    else:
        capture = 76
    with pytest.raises(ValueError):
        capture_tail_prefix(changed, reference_start=48, capture_step=capture)
