"""Replay binding and causal histories apply separately to each trajectory."""

import copy

import pytest

from scripts.tracking.response_models import cross_trajectory_tcp_innovation as cross


def test_fixed_replayed_roster_preserves_trajectory_boundaries():
    result = cross.analyze()
    assert len(result["trajectories"]) == 9 and result["total_periods"] == 103
    assert result["periods_with_history"] == 94
    assert sum(item["summary"] is None for item in result["trajectories"]) == 2
    assert all(item["rows"][0]["history_available"] is False for item in result["trajectories"])
    assert result["shared_initial_states_and_branches"] is True and result["independent_samples_claimed"] is False
    assert result["model_calls"] == result["physics_steps"] == result["fresh_cases"] == 0


def test_replay_comparison_rejects_changed_step_or_physics():
    result = cross.analyze()
    first = result["trajectories"][0]
    assert first["periods"] == 5
    consumed = {}
    config = cross.read_config(consumed)
    source = config["sources"][0]
    producer = cross.read_ref(source["producer"], source["producer_sha256"], consumed)
    checker = cross.read_ref(source["checker"], source["checker_sha256"], consumed)
    actual = cross.selected_branch(producer, source["branches"][0], consumed)["runtime"]["continuation"]["periods"]
    replay = cross.selected_branch(checker, source["branches"][0], consumed)["runtime"]["continuation"]["periods"]
    wrong = copy.deepcopy(replay)
    wrong[-1]["step"] += 1
    with pytest.raises(ValueError, match="period steps"):
        cross.compare_replay(actual, wrong)
    wrong = copy.deepcopy(replay)
    wrong[-1]["actual_inner"][-1]["q"][0] += 0.01
    with pytest.raises(AssertionError):
        cross.compare_replay(actual, wrong)
