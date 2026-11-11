"""Bounded static diagnosis branches retain causal inputs and audit witnesses."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as diagnostic


def setup_case():
    targets = np.tile(np.eye(4), (40, 1, 1))
    targets[:, 0, 3] = np.arange(40) * 0.001
    stopped = {
        "state": np.zeros(79),
        "previous": np.arange(7, dtype=float) / 10,
        "shifted_seed": np.full((8, 6), 0.1),
        "past_inner_error_rad": np.tile(np.arange(25)[:, None] / 1000, (1, 7)),
        "task_success": "deliberately not passed to the candidate",
    }
    context = {
        "model": object(),
        "velocity": np.zeros(7),
        "interface": object(),
        "site": np.eye(4),
        "site_from_tcp": np.eye(4),
        "limits": np.tile([-2.0, 2.0], (7, 1)),
        "mass": np.eye(7),
        "jacobian": np.zeros((6, 7)),
        "q": np.zeros(7),
        "contract": {"kp": np.ones(6), "kd": np.ones(6)},
        "initial_joint": np.zeros(7),
    }
    return context, stopped, targets, diagnostic.new_stats()


def no_witness(*_args, **_kwargs):
    return {"witness_found": False, "status": "no_witness", "response_evaluations": 0}


def test_no_joint_witness_skips_warm_start_and_uses_only_causal_arguments():
    context, stopped, targets, stats = setup_case()
    calls = []

    def solver(*args, **kwargs):
        calls.append((args, kwargs))
        return no_witness()

    result = diagnostic.analyze_candidates(
        context, stopped, targets, np.eye(4), stats, quality_solver=solver, joint_solver=solver, auditor=None
    )
    assert list(result) == ["quality_past", "quality_nominal", "joint_past", "quality_past_warm"]
    assert result["quality_past_warm"]["skipped"] is True
    assert stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 3
    assert stats["independent_endpoint_queries_charged"] == 0
    for args, kwargs in calls:
        np.testing.assert_array_equal(args[2], stopped["shifted_seed"])
        np.testing.assert_array_equal(args[3], stopped["previous"][:6])
        np.testing.assert_array_equal(args[7], targets[5:13])
        assert set(kwargs) == {"past_inner_error_rad"}
    np.testing.assert_array_equal(calls[0][1]["past_inner_error_rad"], stopped["past_inner_error_rad"])
    np.testing.assert_array_equal(calls[1][1]["past_inner_error_rad"], np.zeros((25, 7)))
    np.testing.assert_array_equal(calls[2][1]["past_inner_error_rad"], stopped["past_inner_error_rad"])


def test_joint_witness_is_independently_checked_before_warm_start():
    context, stopped, targets, stats = setup_case()
    events = []
    warm = np.full((8, 6), 0.2)

    def quality(*_args, **kwargs):
        events.append(("quality", kwargs.get("initial_guess")))
        return no_witness()

    def joint(*_args, **_kwargs):
        events.append(("joint", None))
        return {"witness_found": True, "planned_commands": warm, "rollout": {}, "response_evaluations": 0}

    def audit(*_args, **kwargs):
        events.append(("audit", kwargs["enforce_pose_caps"]))
        return {"passed": True, "independent_endpoint_queries": 200, "independent_transition_steps": 2600}

    result = diagnostic.analyze_candidates(
        context, stopped, targets, np.eye(4), stats, quality_solver=quality, joint_solver=joint, auditor=audit
    )
    assert [event[0] for event in events] == ["quality", "quality", "joint", "audit", "quality"]
    assert events[3][1] is False
    np.testing.assert_array_equal(events[4][1], warm)
    assert result["joint_past"]["independent_verification"]["passed"] is True
    assert stats["optimizer_calls_completed"] == 4
    assert stats["independent_endpoint_queries_charged"] == 200
    assert stats["independent_transition_steps_charged"] == 2600


def test_failed_independent_check_stops_warm_start_and_preserves_candidate():
    context, stopped, targets, stats = setup_case()
    partial = {}

    def witness(*_args, **_kwargs):
        return {"witness_found": True, "planned_commands": np.zeros((8, 6)), "rollout": {}, "response_evaluations": 0}

    def reject(*_args, **_kwargs):
        raise ValueError("independent audit failed")

    with pytest.raises(ValueError, match="independent audit failed"):
        diagnostic.analyze_candidates(
            context,
            stopped,
            targets,
            np.eye(4),
            stats,
            quality_solver=no_witness,
            joint_solver=witness,
            auditor=reject,
            results=partial,
        )
    assert partial["joint_past"]["witness_found"] is True
    assert partial["joint_past"]["independent_verification"] is None
    assert "quality_past_warm" not in partial
    assert stats["independent_endpoint_queries_charged"] == 200


def test_response_count_must_match_the_observed_model_calls():
    context, stopped, targets, stats = setup_case()

    def wrong_count(*_args, **_kwargs):
        return {"witness_found": False, "response_evaluations": 1}

    with pytest.raises(ValueError, match="Actual solver responses"):
        diagnostic.analyze_candidates(
            context,
            stopped,
            targets,
            np.eye(4),
            stats,
            quality_solver=wrong_count,
            joint_solver=wrong_count,
            auditor=None,
        )
