"""A fixed H40 is consumed causally while preserving actual feedback history."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime.reset_bowl_fixed_h40 import run_producer
from tests.research.native_runtime.test_reset_live_multi_window import execute_fake
from tests.research.native_runtime.test_reset_live_two_window import Policy, score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case


def setup_fixed(tmp_path, *, stop_at=None, corrupt=False):
    old_dir, new_dir = tmp_path / "old", tmp_path / "fixed"
    old_dir.mkdir()
    new_dir.mkdir()
    old, old_receipt, _old_case, _events, context = execute_fake(old_dir, include_context=True)
    exchange = old_receipt["streaming_model_exchanges"][0]
    envelope = prefix.read(exchange["response_ref"])
    raw = {
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
    if corrupt:
        envelope = copy.deepcopy(envelope)
        envelope["image_sha256"] = "d" * 64
    case = fake_case(new_dir, windows=5)
    case.observers.render_anchor = context["render"]
    receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    initial, owner, events = Policy(), {}, []

    class FixedPolicy(Policy):
        def select(self, q, limits, jacobian, previous, offset):
            if self.completed == stop_at:
                return {"stop": {"reason": "quality_no_witness"}}
            return super().select(q, limits, jacobian, previous, offset)

    def advance(previous, actual_owner, _poses, openings, _anchor, *, generated_step, runtime_receipt):
        assert actual_owner is owner and previous.completed == generated_step == case.sim.inner // 25
        events.append(("advance", generated_step, case.sim.inner))
        current = FixedPolicy(completed=generated_step)
        current.openings = openings
        return current

    def load():
        assert case.sim.inner == 200
        events.append(("load", case.sim.inner))
        return raw, envelope

    args = dict(
        case=context["case"],
        baseline=old["first_window"]["periods"],
        expected_capture=old_receipt["streaming_captures"][0],
        load_reference=load,
        runtime_output=new_dir,
        state_reader=case.state_reader,
        task_observer=case.task_observer,
        policy_factory=lambda *_args: (initial, owner, np.zeros(7)),
        advance_policy=advance,
        gripper_score=score,
        render_scope=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
    )
    return case, receipt, args, events


def test_complete_fixed_horizon_advances_one_history_without_model_calls(tmp_path):
    case, receipt, args, events = setup_fixed(tmp_path)
    result = run_producer(case.env, terminal_at(case, 0), case.observers, receipt, **args)
    fixed = result["fixed"]
    assert fixed["complete"] and fixed["actual_model_requests"] == 0 and fixed["stop"] is None
    assert [row["start_step"] for row in fixed["segments"]] == [8, 16, 24, 32, 40]
    assert [row["generated_endpoints"] for row in fixed["segments"]] == [40, 32, 24, 16, 8]
    assert [row["terminal_hold_endpoints"] for row in fixed["segments"]] == [0, 8, 16, 24, 32]
    assert events[0] == ("load", 200) and [event[1] for event in events[1:]] == [8, 16, 24, 32, 40]
    assert case.sim.inner == 1200 and receipt["control_steps_completed"] == 48


def test_partial_segment_stops_without_dispatching_the_next_action(tmp_path):
    case, receipt, args, events = setup_fixed(tmp_path, stop_at=19)
    result = run_producer(case.env, terminal_at(case, 0), case.observers, receipt, **args)
    fixed = result["fixed"]
    assert not fixed["complete"] and fixed["stop"]["reason"] == "quality_no_witness"
    assert fixed["stop"]["at_step"] == 19 and fixed["stop"]["next_action_dispatched"] is False
    assert [row["continuation"]["steps_completed"] for row in fixed["segments"]] == [8, 3]
    assert case.sim.inner == 475 and receipt["control_steps_completed"] == 19
    assert [event[1] for event in events[1:]] == [8, 16]


def test_saved_reference_must_match_current_image_before_new_dispatch(tmp_path):
    case, receipt, args, _events = setup_fixed(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="belong to this actual current image"):
        run_producer(case.env, terminal_at(case, 0), case.observers, receipt, **args)
    assert case.sim.inner == 200 and receipt["control_steps_completed"] == 8
