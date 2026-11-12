"""Replay only fixed-reference actions actually dispatched by the producer."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as checker_module
from scripts.simulation.runtime.reset_bowl_fixed_h40 import run_producer
from scripts.simulation.runtime.reset_bowl_fixed_h40_replay import run_checker
from tests.research.native_runtime.test_reset_bowl_fixed_h40 import setup_fixed
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    """The producer fixture shares this test process; a subprocess checks real isolation."""
    monkeypatch.setattr(checker_module.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(checker_module.original, "checker_isolation", lambda: None)


def setup_replay(tmp_path, *, stop_at=None):
    produced, receipt, args, _events = setup_fixed(tmp_path, stop_at=stop_at)
    saved, load = [], args["load_reference"]

    def remember():
        value = load()
        saved.append(value)
        return value

    args["load_reference"] = remember
    result = run_producer(produced.env, terminal_at(produced, 0), produced.observers, receipt, **args)
    producer = {"runtime": {**receipt, "continuation": result}}
    directory = tmp_path / "checker"
    directory.mkdir()
    checker = fake_case(directory, windows=5)
    checker.observers.render_anchor = produced.observers.render_anchor
    checked_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    options = dict(
        case=args["case"],
        producer=producer,
        expected_capture=args["expected_capture"],
        load_reference=lambda: saved[0],
        runtime_output=directory,
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
        gripper_score=score,
        render_scope=lambda render, _scope, *a, **kw: render(*a, **kw),
    )
    return checker, checked_receipt, options


def test_replays_all_generated_targets_and_1200_inner_steps(tmp_path):
    checker, receipt, options = setup_replay(tmp_path)
    result = run_checker(checker.env, terminal_at(checker, 0), checker.observers, receipt, **options)
    assert [row["steps_completed"] for row in result["replays"]] == [8] * 5
    assert checker.sim.inner == 1200 and receipt["newly_selected_commands"] == 0
    assert result["reported_producer_stop"] is None


def test_replays_only_completed_actions_after_early_stop(tmp_path):
    checker, receipt, options = setup_replay(tmp_path, stop_at=19)
    result = run_checker(checker.env, terminal_at(checker, 0), checker.observers, receipt, **options)
    assert [row["steps_completed"] for row in result["replays"]] == [8, 3]
    assert result["reported_producer_stop"]["reason"] == "quality_no_witness"
    assert result["stop_reason_independently_verified"] is False and checker.sim.inner == 475


def test_rejects_changed_recorded_state_before_replaying_past_it(tmp_path):
    checker, receipt, options = setup_replay(tmp_path)
    producer = copy.deepcopy(options["producer"])
    row = producer["runtime"]["continuation"]["fixed"]["segments"][1]["continuation"]["periods"][0]
    row["actual_state"][1] += 0.1
    options["producer"] = producer
    with pytest.raises(AssertionError):
        run_checker(checker.env, terminal_at(checker, 0), checker.observers, receipt, **options)
    assert checker.sim.inner == 425


def test_checker_import_does_not_load_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as s; "
        "s.isolation.checker_isolation(); s.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
