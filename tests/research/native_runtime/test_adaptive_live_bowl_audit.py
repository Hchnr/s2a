"""Audit saved physical receipts rewrapped as a synthetic adaptive stream."""

import copy

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_audit as audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

CHECKER_BRANCH = {
    "path": str(admission.ROOT / "outputs/causal_replan_step107_execution_20261112/checker/case_00/result.json"),
    "sha256": "b18e7be755be47a012cbd712838865c87a14d474cb4125fa4e2ab63bd3a57260",
}


def rewrapped():
    config = prefix.read({"path": str(admission.CONFIG), "sha256": admission.CONFIG_SHA})
    producer = prefix.read(
        {"path": str(admission.ROOT / config["prior_execution"]), "sha256": config["prior_execution_sha256"]}
    )
    checker = prefix.read(CHECKER_BRANCH)
    actual = producer["runtime"]["continuation"]["replanned"]
    replay = checker["runtime"]["continuation"]["replanned"]
    count = sum(len(row["continuation"]["periods"]) for row in actual["segments"])
    assert count == 6
    producer["runtime"]["continuation"]["adaptive"] = dict(
        segments=[{**row, "inflight": None} for row in actual["segments"]],
        stop=actual["stop"],
        control_steps_attempted=count,
        control_steps_completed=count,
        feedback_periods_committed=count,
        requests_issued=len(actual["segments"]),
        responses_accepted=len(actual["segments"]),
    )
    checker["runtime"]["continuation"]["adaptive"] = dict(
        segments=replay["segments"],
        reported_producer_stop=actual["stop"],
        stop_reason_independently_verified=False,
        complete=True,
        control_steps_attempted=count,
        control_steps_completed=count,
    )
    producer["runtime"]["adaptive_captures"] = [producer["runtime"]["replan_capture"]["capture"]]
    producer["runtime"]["adaptive_stop_frame"] = {
        "required": True,
        "send_attempted": True,
        "sent": True,
        "frame": {"version": 1, "kind": "stop", "session": admission.SESSION, "next_step": 113},
    }
    checker["runtime"]["adaptive_checker_captures"] = [checker["runtime"]["replan_capture"]["capture"]]
    return config, producer, checker


def test_saved_producer_and_checker_receipts_are_counted_and_compared():
    config, producer, checker = rewrapped()
    left = audit.summarize_branch(producer, "producer", config)
    right = audit.summarize_branch(checker, "checker", config)
    assert left["control_steps"] == right["control_steps"] == 113
    assert left["adaptive_steps"] == right["adaptive_steps"] == 6
    assert left["adaptive_task_success_periods"] == right["adaptive_task_success_periods"] == 5
    assert all(value == 0 for value in audit.compare_branches(producer, checker).values())
    assert right["candidate_budget"] is None


@pytest.mark.parametrize("change", ["stop", "step_count", "inflight", "selected", "image", "buffer", "model_stop"])
def test_inconsistent_saved_evidence_is_rejected(change):
    config, producer, checker = rewrapped()
    if change == "stop":
        producer["runtime"]["continuation"]["adaptive"]["stop"] = {
            "reason": "quality_no_witness",
            "at_step": 114,
            "next_action_dispatched": False,
        }
    elif change == "step_count":
        producer["runtime"]["control_steps_completed"] -= 1
    elif change == "inflight":
        producer["runtime"]["continuation"]["adaptive"]["segments"][0]["inflight"] = {"inner_completed": 3}
    elif change == "selected":
        checker["runtime"]["newly_selected_commands"] = 1
    elif change == "buffer":
        audit.periods(producer, "producer")[0][3]["inner_below_buffer_samples"] = 1
    elif change == "model_stop":
        producer["runtime"]["adaptive_stop_frame"]["sent"] = False
    else:
        checker["runtime"]["adaptive_checker_captures"][0] = copy.deepcopy(
            checker["runtime"]["adaptive_checker_captures"][0]
        )
        checker["runtime"]["adaptive_checker_captures"][0]["canvas_png"]["sha256"] = "f" * 64
    with pytest.raises((ValueError, AssertionError)):
        if change == "selected":
            audit.summarize_branch(checker, "checker", config)
        elif change == "image":
            audit.compare_branches(producer, checker)
        else:
            audit.summarize_branch(producer, "producer", config)
