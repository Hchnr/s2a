"""Capture step 88 only after replaying every actual recorded action."""

import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import reset_bowl_reference_tail as tail
from scripts.simulation.runtime.run_reset_bowl_step88_capture import capture_after_checker
from tests.research.native_runtime.test_reset_bowl_reference_tail import run_longer, setup_longer
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)


def execute_capture(tmp_path, *, stop_at=None):
    producer_dir = tmp_path / "producer"
    checker_dir = tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    case, receipt, options, _, first_reference, previous = setup_longer(producer_dir, stop_at=stop_at)
    result = run_longer(case, receipt, options, previous)
    producer = {"runtime": {**receipt, "continuation": result}}
    checker = fake_case(checker_dir, windows=10)
    checker.observers.render_anchor = case.observers.render_anchor
    checker_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    kwargs = dict(
        checker=tail.run_checker,
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
        render_scope=lambda render, _scope, *args, **values: render(*args, **values),
        capture_render=lambda render, _scope, *args, **values: render(*args, **values),
    )
    return checker, checker_receipt, kwargs


def test_captures_the_actual_endpoint_after_full_independent_replay(tmp_path):
    checker, receipt, kwargs = execute_capture(tmp_path)
    result = capture_after_checker(checker.env, terminal_at(checker, 0), checker.observers, receipt, **kwargs)
    assert result["tail"]["complete"] and checker.sim.inner == 88 * 25
    assert receipt["newly_selected_commands"] == 0
    assert receipt["terminal_capture"]["snapshot"]["step"] == 88
    assert receipt["terminal_capture"]["capture"]["frame_index"] == 88


def test_early_producer_stop_cannot_be_presented_as_step88(tmp_path):
    checker, receipt, kwargs = execute_capture(tmp_path, stop_at=73)
    with pytest.raises(ValueError, match="entire original 88-period trajectory"):
        capture_after_checker(checker.env, terminal_at(checker, 0), checker.observers, receipt, **kwargs)
    assert checker.sim.inner == 73 * 25 and "terminal_capture" not in receipt


def test_capture_worker_import_does_not_load_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import run_reset_bowl_step88_capture as owner; "
        "owner.previous.isolation.checker_isolation(); owner.previous.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
