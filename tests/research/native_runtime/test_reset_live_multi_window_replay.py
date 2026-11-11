"""Replay multiple current-image windows without executing a candidate or model."""

import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_multi_window_replay as checker_module
from scripts.simulation.runtime.reset_live_multi_window_replay import run_checker
from tests.research.native_runtime.test_reset_live_multi_window import execute_fake
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    """Producer fixtures share this test process; the separate import test checks real isolation."""
    monkeypatch.setattr(checker_module.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(checker_module.original, "checker_isolation", lambda: None)


def setup_replay(tmp_path, *, stop_at=None, buffer_at=None):
    output = tmp_path / "producer"
    output.mkdir()
    result, receipt, _case, _events, context = execute_fake(
        output, stop_at=stop_at, buffer_at=buffer_at, include_context=True
    )
    producer = {"runtime": {**receipt, "continuation": result}}
    model_responses = []
    for exchange in receipt["streaming_model_exchanges"]:
        envelope = prefix.read(exchange["response_ref"])
        model_responses.append(
            {
                "status": "success",
                "outputs": [
                    {
                        "content": {
                            "action": envelope["action"],
                            "action128": {"normalized_action": envelope["normalized_action"]},
                        }
                    }
                ],
            }
        )
    checker_output = tmp_path / "checker"
    checker_output.mkdir()
    checker = fake_case(checker_output, windows=5, buffer_at=buffer_at)
    checker.observers.render_anchor = context["render"]
    checker_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    args = dict(
        case=context["case"],
        producer=producer,
        registration=context["registration"],
        model_responses=model_responses,
        runtime_output=checker_output,
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
        gripper_score=score,
        render_scope=lambda render, _scope, *a, **kw: render(*a, **kw),
    )
    return checker, checker_receipt, args


def replay(checker, receipt, args):
    return run_checker(checker.env, terminal_at(checker, 0), checker.observers, receipt, **args)


def test_replays_five_real_boundaries_and_every_dispatched_command(tmp_path):
    checker, receipt, args = setup_replay(tmp_path)
    result = replay(checker, receipt, args)
    assert result["actual_model_requests"] == 5 and result["actual_stream_periods"] == 40
    assert [row["steps_completed"] for row in result["replays"]] == [8] * 5
    assert [capture["frame_index"] for capture in receipt["checker_stream_captures"]] == [8, 16, 24, 32, 40]
    assert receipt["control_steps_completed"] == 48 and receipt["newly_selected_commands"] == 0
    assert checker.sim.inner == 1200


def test_replays_only_actual_partial_window_and_reports_decision_stop(tmp_path):
    checker, receipt, args = setup_replay(tmp_path, stop_at=19)
    result = replay(checker, receipt, args)
    assert result["actual_model_requests"] == 2 and result["actual_stream_periods"] == 11
    assert [row["steps_completed"] for row in result["replays"]] == [8, 3]
    assert result["reported_producer_stop"]["reason"] == "quality_no_witness"
    assert result["stop_reason_independently_verified"] is False
    assert checker.sim.inner == 475


def test_prefix_stop_replays_without_model_requests(tmp_path):
    checker, receipt, args = setup_replay(tmp_path, buffer_at=199)
    result = replay(checker, receipt, args)
    assert result["actual_model_requests"] == result["actual_stream_periods"] == 0
    assert result["replays"] == []
    assert result["reported_producer_stop"]["reason"] == "observed_prefix_joint_buffer_loss"
    assert checker.sim.inner == 200


@pytest.mark.parametrize("corruption", ["model", "capture", "state", "counts"])
def test_rejects_divergent_evidence_before_advancing_past_it(tmp_path, corruption):
    checker, receipt, args = setup_replay(tmp_path)
    if corruption == "model":
        args["model_responses"][1]["outputs"][0]["content"]["action"][0][0] = 0.1
    elif corruption == "capture":
        args["producer"]["runtime"]["streaming_captures"][1]["canvas_rgb_sha256"] = "d" * 64
    elif corruption == "state":
        row = args["producer"]["runtime"]["continuation"]["stream"]["segments"][1]["continuation"]["periods"][0]
        row["actual_state"] = row["actual_state"].copy()
        row["actual_state"][1] += 0.1
    else:
        args["producer"]["runtime"]["continuation"]["stream"]["requests_issued"] = 4
    with pytest.raises((ValueError, AssertionError)):
        replay(checker, receipt, args)
    assert checker.sim.inner <= (0 if corruption == "counts" else 425 if corruption == "state" else 400)


def test_rejects_missing_exchange_instead_of_inventing_a_window(tmp_path):
    checker, receipt, args = setup_replay(tmp_path)
    args["model_responses"].pop()
    with pytest.raises(ValueError, match="complete exchanges"):
        replay(checker, receipt, args)
    assert checker.sim.inner == 0


def test_checker_import_does_not_load_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import reset_live_multi_window_replay as s; "
        "s.isolation.checker_isolation(); s.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
