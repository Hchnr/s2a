"""Only a completed causal innovation may enter the static quality search."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base
from scripts.tracking.trajectory_search import run_causal_innovation_quality_search as run


def test_future_step125_cannot_change_the_selected_step124_innovation():
    config = run.FIXED
    saved = prefix.read(
        {"path": str(run.ROOT / config["innovation_audit"]), "sha256": config["innovation_audit_sha256"]}
    )
    first = run.completed_innovation(saved)
    changed = copy.deepcopy(saved)
    changed["periods"][-1]["actual_error"][0] += 1.0
    second = run.completed_innovation(changed)
    np.testing.assert_array_equal(first["translation_m"], second["translation_m"])
    np.testing.assert_array_equal(first["rotation_matrix"], second["rotation_matrix"])
    changed["periods"][-2]["completed_innovation"]["completed_history"] = 46
    with pytest.raises(ValueError, match="completed before step125"):
        run.completed_innovation(changed)


def test_independent_corrected_quality_gate_controls_witness_status(monkeypatch):
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
    stopped = {"previous": np.zeros(7), "shifted_seed": np.zeros((8, 6)), "past_inner_error_rad": np.zeros((25, 7))}
    innovation = {"translation_m": np.array([0.002, 0, 0]), "rotation_matrix": np.eye(3)}
    targets = np.tile(np.eye(4), (40, 1, 1))
    for accepted in (False, True):
        stats = base.new_stats()

        def fake_solver(*_args, **kwargs):
            np.testing.assert_array_equal(kwargs["translation_innovation"], innovation["translation_m"])
            stats["optimizer_responses_attempted"] += 1
            stats["optimizer_responses_completed"] += 1
            return {
                "witness_found": True,
                "response_evaluations": 1,
                "rollout": {},
                "innovation_corrected_first_tcp_error": np.array([0.002, 0, 0, 0, 0, 0]),
            }

        def auditor(*_args, **_kwargs):
            return {
                "passed": True,
                "first_pose_caps_satisfied": True,
                "first_pose_caps_enforced": True,
                "independent_endpoint_queries": 200,
                "independent_transition_steps": 2600,
                "first_tcp_error": np.zeros(6) if accepted else np.array([0.02, 0, 0, 0, 0, 0]),
            }

        monkeypatch.setattr(run.solver, "solve_interior_plan", fake_solver)
        result = run.single_search(innovation)(
            context,
            stopped,
            targets,
            np.eye(4),
            stats,
            quality_solver=None,
            joint_solver=None,
            auditor=auditor,
            results={},
        )
        assert result["innovation_quality"]["certified_model_witness"] is accepted
        assert stats["independent_endpoint_queries_charged"] == 200
