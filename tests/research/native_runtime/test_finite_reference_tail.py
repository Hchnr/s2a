"""A saved H40 tail keeps generated indices, history and actual dispatches aligned."""

import copy

import numpy as np
import pytest

from s2a.research.native_runtime.finite_reference_tail import run_finite_reference_tail
from tests.research.native_runtime.test_reset_live_two_window import Policy, score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


def setup_tail(tmp_path, *, stop_at=None, render_changes=False):
    case = fake_case(tmp_path, windows=5, render_changes=render_changes)
    for _ in range(56 * 25):
        case.sim.step()
    terminal = terminal_at(case, 56)
    receipt = dict(control_steps_attempted=56, control_steps_completed=56, newly_selected_commands=0)
    prior = dict(
        step=55,
        command=np.zeros(7),
        actual_state=case.sim.get_state().copy(),
        endpoint_receipt=terminal["endpoint_receipt"],
        inner_below_buffer_samples=0,
    )
    poses = np.tile(np.eye(4), (40, 1, 1))
    poses[:, 0, 3] = np.arange(40) * 0.001
    openings = np.linspace(1.0, 0.0, 40)
    events = []

    class TailPolicy(Policy):
        def select(self, q, limits, jacobian, previous, offset):
            if self.completed == stop_at:
                return {"stop": {"reason": "quality_no_witness"}}
            return super().select(q, limits, jacobian, previous, offset)

    def open_segment(window, opening, anchor, step, expected):
        assert expected is None
        np.testing.assert_array_equal(anchor, np.eye(4))
        events.append((step, window.copy(), opening.copy()))
        policy = TailPolicy(completed=step)

        def complete(env, command, offset):
            period = receipt["continuation_partial"]["periods"][-1]
            history = policy.complete(env, command, offset, [item["q"] for item in period["actual_inner"]])
            return {"score": score(env, opening[offset]), "policy_history": history}

        return policy.select, complete

    options = dict(
        reference_poses=poses,
        reference_openings=openings,
        reference_start_step=48,
        consumed=8,
        max_segments=4,
        previous_command=np.zeros(7),
        previous_period=prior,
        tool=np.eye(4),
        target_object="target",
        open_segment=open_segment,
        state_reader=case.state_reader,
        task_observer=case.task_observer,
    )
    return case, terminal, receipt, options, events


def execute(case, terminal, receipt, options):
    return run_finite_reference_tail(case.env, terminal, case.observers, receipt, **options)


def test_consumes_only_remaining_targets_and_marks_terminal_preview_holds(tmp_path):
    case, terminal, receipt, options, events = setup_tail(tmp_path)
    result = execute(case, terminal, receipt, options)
    assert result["complete"] and result["stop"] is None and result["generated_periods"] == 32
    assert case.sim.inner == 88 * 25 and receipt["control_steps_completed"] == 88
    assert [event[0] for event in events] == [56, 64, 72, 80]
    assert [row["generated_endpoints"] for row in result["segments"]] == [32, 24, 16, 8]
    assert [row["terminal_hold_endpoints"] for row in result["segments"]] == [8, 16, 24, 32]
    for index, (_step, poses, openings) in enumerate(events):
        offset = 8 + 8 * index
        np.testing.assert_array_equal(poses[: 40 - offset], options["reference_poses"][offset:])
        np.testing.assert_array_equal(openings[: 40 - offset], options["reference_openings"][offset:])
        np.testing.assert_array_equal(openings[40 - offset :], np.repeat(openings[39 - offset], offset))


def test_early_stop_records_only_dispatched_periods(tmp_path):
    case, terminal, receipt, options, _events = setup_tail(tmp_path, stop_at=73)
    result = execute(case, terminal, receipt, options)
    assert not result["complete"] and result["stop"]["reason"] == "quality_no_witness"
    assert result["stop"]["at_step"] == 73 and result["stop"]["next_action_dispatched"] is False
    assert [len(row["continuation"]["periods"]) for row in result["segments"]] == [8, 8, 1]
    assert case.sim.inner == 73 * 25


def test_independent_replay_dispatches_only_recorded_actions(tmp_path):
    producer_dir, checker_dir = tmp_path / "producer", tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    case, terminal, receipt, options, _ = setup_tail(producer_dir, stop_at=73)
    produced = execute(case, terminal, receipt, options)
    checker, terminal, checked_receipt, checked_options, _ = setup_tail(checker_dir)

    def open_replay(_poses, openings, _anchor, _step, expected):
        assert expected is not None

        def select(_q, _limits, _jacobian, _previous, offset):
            return {"command": np.asarray(expected[offset]["command"]), "newly_selected": False}

        def complete(env, _command, offset):
            observed = score(env, openings[offset])
            assert observed == expected[offset]["completed_feedback"]["score"]
            return {"score": observed, "policy_history": None}

        return select, complete

    checked_options.update(
        open_segment=open_replay, recorded_segments=produced["segments"], recorded_stop=produced["stop"]
    )
    replayed = execute(checker, terminal, checked_receipt, checked_options)
    assert checker.sim.inner == 73 * 25 and checked_receipt["newly_selected_commands"] == 0
    assert replayed["stop"] == produced["stop"] and replayed["stop_recomputed"] is False
    assert [len(row["continuation"]["periods"]) for row in replayed["segments"]] == [8, 8, 1]


def test_wrong_previous_state_or_budget_cannot_dispatch(tmp_path):
    case, terminal, receipt, options, events = setup_tail(tmp_path)
    changed = copy.deepcopy(options["previous_period"])
    changed["actual_state"][0] += 1
    options["previous_period"] = changed
    with pytest.raises(ValueError, match="last completed period"):
        execute(case, terminal, receipt, options)
    assert case.sim.inner == 56 * 25 and not events
    options["max_segments"] = 5
    with pytest.raises(ValueError, match="bounded segment count"):
        execute(case, terminal, receipt, options)
    assert case.sim.inner == 56 * 25 and not events


def test_camera_rendering_cannot_advance_state_before_dispatch(tmp_path):
    case, terminal, receipt, options, _ = setup_tail(tmp_path, render_changes=True)
    with pytest.raises(ValueError, match="Camera rendering cannot advance"):
        execute(case, terminal, receipt, options)
    assert case.sim.inner == 56 * 25 and receipt["control_steps_completed"] == 56


def test_joint_buffer_loss_stops_before_opening_a_segment(tmp_path):
    case, _terminal, receipt, options, events = setup_tail(tmp_path)
    case.sim.data.qpos[0] = 0.96
    terminal = terminal_at(case, 56)
    options["previous_period"]["actual_state"] = case.sim.get_state().copy()
    options["previous_period"]["endpoint_receipt"] = terminal["endpoint_receipt"]
    result = execute(case, terminal, receipt, options)
    assert result["stop"]["reason"] == "observed_joint_buffer_loss"
    assert result["stop"]["at_step"] == 56 and not events and case.sim.inner == 56 * 25


def test_replay_rejects_unrecorded_action_before_physics(tmp_path):
    producer_dir, checker_dir = tmp_path / "producer", tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    case, terminal, receipt, options, _ = setup_tail(producer_dir, stop_at=57)
    produced = execute(case, terminal, receipt, options)
    checker, terminal, checked_receipt, checked_options, _ = setup_tail(checker_dir)

    def choose_wrong(_poses, _openings, _anchor, _step, _expected):
        return lambda *_args: {"command": np.ones(7) * 0.1, "newly_selected": False}, lambda *_args: None

    checked_options.update(
        open_segment=choose_wrong, recorded_segments=produced["segments"], recorded_stop=produced["stop"]
    )
    with pytest.raises(ValueError, match="only the recorded command"):
        execute(checker, terminal, checked_receipt, checked_options)
    assert checker.sim.inner == 56 * 25 and checked_receipt["control_steps_completed"] == 56


def test_inconsistent_recorded_stop_is_rejected_before_physics(tmp_path):
    producer_dir, checker_dir = tmp_path / "producer", tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    case, terminal, receipt, options, _ = setup_tail(producer_dir, stop_at=57)
    produced = execute(case, terminal, receipt, options)
    checker, terminal, checked_receipt, checked_options, events = setup_tail(checker_dir)
    bad_stop = {**produced["stop"], "at_step": 58}
    checked_options.update(recorded_segments=produced["segments"], recorded_stop=bad_stop)
    with pytest.raises(ValueError, match="stops and dispatched segment lengths"):
        execute(checker, terminal, checked_receipt, checked_options)
    assert checker.sim.inner == 56 * 25 and not events
