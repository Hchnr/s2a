"""A second reference starts only after the original history and image match."""

import copy

import pytest

from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import reset_bowl_h40_second_window as second
from scripts.simulation.runtime.reset_bowl_fixed_h40 import run_producer as first_producer
from scripts.simulation.runtime.reset_live_two_window_replay_v2 import minimal_renderer_endpoint
from tests.research.native_runtime.test_reset_bowl_fixed_h40 import setup_fixed
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)


def setup_second(tmp_path, *, stop_at=None):
    prior_dir, next_dir = tmp_path / "prior", tmp_path / "next"
    prior_dir.mkdir()
    next_dir.mkdir()
    prior, prior_receipt, prior_args, _ = setup_fixed(prior_dir)
    saved_reference = []
    old_loader = prior_args["load_reference"]

    def load_old():
        value = old_loader()
        saved_reference.append(value)
        return value

    prior_args["load_reference"] = load_old
    prior_run = first_producer(prior.env, terminal_at(prior, 0), prior.observers, prior_receipt, **prior_args)
    sample, endpoint = prior.observers.endpoint(prior.env, 48)
    capture = prior.observers.render_anchor(
        prior.env, 48, sample, minimal_renderer_endpoint(endpoint, 48), prior_args["runtime_output"]
    )
    current = measured_snapshot(prior.env, 48)
    response, old_envelope = saved_reference[0]
    envelope = {
        **copy.deepcopy(old_envelope),
        **current,
        "image_sha256": capture["canvas_png"]["sha256"],
        "seed": 42,
    }
    case, receipt, args, events = setup_fixed(next_dir, stop_at=stop_at)
    options = dict(
        **args,
        previous_producer={"runtime": {"continuation": prior_run}},
        expected_second_capture=capture,
        load_second_reference=lambda: (copy.deepcopy(response), copy.deepcopy(envelope)),
    )
    return case, receipt, options, events, saved_reference[0]


def produce(case, receipt, options):
    return second.run_producer(case.env, terminal_at(case, 0), case.observers, receipt, **options)


def test_second_reference_advances_the_same_history_and_replays_actual_actions(tmp_path):
    case, receipt, options, events, first_reference = setup_second(tmp_path)
    result = produce(case, receipt, options)
    assert case.sim.inner == 1400 and receipt["control_steps_completed"] == 56
    assert result["fixed"]["complete"] and result["second"]["attempted"]
    assert [row["step"] for row in result["second"]["periods"]] == list(range(48, 56))
    assert [event[1] for event in events if event[0] == "advance"] == [8, 16, 24, 32, 40, 48]

    checker_dir = tmp_path / "checker"
    checker_dir.mkdir()
    checker = fake_case(checker_dir, windows=6)
    checker.observers.render_anchor = case.observers.render_anchor
    checker_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    producer = {"runtime": {**receipt, "continuation": result}}
    replayed = second.run_checker(
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
    assert checker.sim.inner == 1400 and checker_receipt["newly_selected_commands"] == 0
    assert len(replayed["second"]["periods"]) == 8 and not replayed["second"]["stop_recomputed"]


def test_a_changed_step48_reference_stops_before_new_dispatch(tmp_path):
    case, receipt, options, _, _ = setup_second(tmp_path)
    response, envelope = options["load_second_reference"]()
    envelope["image_sha256"] = "a" * 64
    options["load_second_reference"] = lambda: (response, envelope)
    with pytest.raises(ValueError, match="must match the actual current image"):
        produce(case, receipt, options)
    assert case.sim.inner == 1200 and receipt["control_steps_completed"] == 48


def test_changed_original_history_stops_before_new_dispatch(tmp_path):
    case, receipt, options, _, _ = setup_second(tmp_path)
    previous = copy.deepcopy(options["previous_producer"])
    row = previous["runtime"]["continuation"]["fixed"]["segments"][2]["continuation"]["periods"][0]
    row["command"][0] = 0.1
    options["previous_producer"] = previous
    with pytest.raises(ValueError, match="differ in command"):
        produce(case, receipt, options)
    assert case.sim.inner == 1200 and receipt["control_steps_completed"] == 48


def test_original_reference_stop_does_not_start_the_second_window(tmp_path):
    case, receipt, options, _, _ = setup_second(tmp_path, stop_at=19)
    result = produce(case, receipt, options)
    assert case.sim.inner == 475 and receipt["control_steps_completed"] == 19
    assert not result["second"]["attempted"] and not result["second"]["periods"]
    assert result["second"]["stop"]["reason"] == "quality_no_witness"


def test_second_window_stop_preserves_only_dispatched_periods(tmp_path):
    case, receipt, options, _, _ = setup_second(tmp_path, stop_at=51)
    result = produce(case, receipt, options)
    assert case.sim.inner == 1275 and receipt["control_steps_completed"] == 51
    assert [row["step"] for row in result["second"]["periods"]] == [48, 49, 50]
    assert result["second"]["stop"]["reason"] == "quality_no_witness"
    assert result["second"]["stop"]["next_action_dispatched"] is False
