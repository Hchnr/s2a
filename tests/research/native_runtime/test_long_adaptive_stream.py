"""Exercise the registered long horizon against the real adaptive ledger."""

import pytest

from tests.research.native_runtime.test_adaptive_streaming_continuation import execute, make_case


@pytest.mark.parametrize(
    ("replan_steps", "expected_requests"),
    [(tuple(range(81, 120)), 40), ((), 5)],
)
def test_forty_completed_periods_keep_causal_request_and_action_bounds(tmp_path, replan_steps, expected_requests):
    case = make_case(tmp_path, windows=40, max_periods=40)
    result = execute(case, replan_steps=replan_steps)
    assert result["complete"] is True
    assert result["stop"] == {"reason": "period_budget_exhausted", "at_step": 120, "next_action_dispatched": False}
    assert result["requests_issued"] == result["responses_accepted"] == expected_requests
    assert result["feedback_periods_committed"] == result["control_steps_completed"] == 40
    assert case.sim.inner == 1000 and case.commits == 40 and case.ledger.phase == "done"
    requests = [event for event in case.events if event[0] == "request"]
    assert len(requests) == expected_requests
    assert all(inner == 25 * (step - 80) for _, step, inner in requests)
    assert all(segment["inflight"] is None for segment in result["segments"])
