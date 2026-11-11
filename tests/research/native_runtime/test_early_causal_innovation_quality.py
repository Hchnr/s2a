"""Early dual-branch search keeps the original reference and audit gates."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base
from scripts.tracking.trajectory_search import run_early_causal_innovation_quality as run


def fixture():
    targets = np.tile(np.eye(4), (40, 1, 1))
    targets[:, 0, 3] = np.arange(40) * 0.001
    stopped = {
        "previous": np.zeros(7),
        "shifted_seed": np.full((8, 6), 0.1),
        "past_inner_error_rad": np.zeros((25, 7)),
        "translation_innovation": np.array([0.002, 0, 0]),
        "rotation_discrepancy": np.eye(3),
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
    return context, stopped, targets, base.new_stats()


def test_both_branches_use_cursor_two_and_require_independent_checks():
    context, stopped, targets, stats = fixture()
    observed = []

    def solve(*args, **kwargs):
        np.testing.assert_array_equal(args[2], stopped["shifted_seed"])
        np.testing.assert_array_equal(args[7], targets[2:10])
        observed.append(set(kwargs))
        stats["optimizer_responses_attempted"] += 1
        stats["optimizer_responses_completed"] += 1
        return {
            "witness_found": True,
            "response_evaluations": 1,
            "rollout": {},
            "innovation_corrected_first_tcp_error": np.array([0.002, 0, 0, 0, 0, 0]),
        }

    def auditor(*_args, **kwargs):
        assert kwargs["enforce_pose_caps"] is True
        return {
            "passed": True,
            "first_pose_caps_satisfied": True,
            "first_pose_caps_enforced": True,
            "independent_endpoint_queries": 200,
            "independent_transition_steps": 2600,
            "first_tcp_error": np.zeros(6),
        }

    results = run.run_branches(
        context, stopped, targets, np.eye(4), stats, raw=solve, innovation=solve, auditor=auditor, results={}
    )
    assert observed == [
        {"past_inner_error_rad"},
        {"past_inner_error_rad", "translation_innovation", "rotation_discrepancy"},
    ]
    assert results["raw_quality"]["certified_model_witness"] is True
    assert results["innovation_quality"]["certified_model_witness"] is True
    assert stats["independent_endpoint_queries_charged"] == 400


def test_branch_cannot_exceed_the_original_response_budget():
    context, stopped, targets, stats = fixture()

    def over_budget(*_args, **_kwargs):
        return {"witness_found": False, "response_evaluations": 161}

    with pytest.raises(ValueError, match="160-response budget"):
        run.run_branches(
            context,
            stopped,
            targets,
            np.eye(4),
            stats,
            raw=over_budget,
            innovation=over_budget,
            auditor=None,
            results={},
        )
